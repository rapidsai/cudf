/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "comp/compression.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <numeric>

namespace cudf::io::experimental {
namespace detail {

namespace {

static constexpr uint32_t magic_number = 0x4C425443;  ///< "CTBL" in little-endian

/**
 * @brief V1 binary file format header for CudfTable (24 bytes)
 *
 * Layout: [magic(4)] [version=1(4)] [metadata_length(8)] [data_length(8)]
 * Followed by: [metadata (variable)] [data (variable)]
 */
struct cudftable_header_v1 {
  static constexpr uint32_t version = 1;

  uint32_t magic{};
  uint32_t format_version{};
  uint64_t metadata_length{};
  uint64_t data_length{};

  cudftable_header_v1() = default;
  cudftable_header_v1(uint64_t metadata_size, uint64_t data_size)
    : magic{magic_number},
      format_version{version},
      metadata_length{metadata_size},
      data_length{data_size}
  {
  }
};

/**
 * @brief V2 binary file format header for CudfTable with block compression (48 bytes)
 *
 * Layout: [magic(4)] [version=2(4)] [compression(4)] [block_size(4)]
 *         [metadata_length(8)] [uncompressed_data_length(8)]
 *         [num_blocks(8)] [compressed_data_length(8)]
 * Followed by: [metadata (variable, uncompressed)]
 *              [block_index (num_blocks * 16 bytes)]
 *              [compressed blocks (variable)]
 */
struct cudftable_header_v2 {
  static constexpr uint32_t version = 2;

  uint32_t magic{};
  uint32_t format_version{};
  uint32_t compression{};
  uint32_t block_size{};
  uint64_t metadata_length{};
  uint64_t uncompressed_data_length{};
  uint64_t num_blocks{};
  uint64_t compressed_data_length{};

  cudftable_header_v2() = default;
  cudftable_header_v2(compression_type comp,
                      uint32_t blk_size,
                      uint64_t metadata_size,
                      uint64_t uncomp_data_size,
                      uint64_t n_blocks,
                      uint64_t comp_data_size)
    : magic{magic_number},
      format_version{version},
      compression{static_cast<uint32_t>(comp)},
      block_size{blk_size},
      metadata_length{metadata_size},
      uncompressed_data_length{uncomp_data_size},
      num_blocks{n_blocks},
      compressed_data_length{comp_data_size}
  {
  }
};

/**
 * @brief Block index entry for V2 format (16 bytes each)
 */
struct block_index_entry {
  uint64_t compressed_size;
  uint64_t uncompressed_size;
};

/**
 * @brief Common prefix shared by V1 and V2 headers, used for version detection.
 */
struct cudftable_header_prefix {
  uint32_t magic{};
  uint32_t format_version{};
};

}  // anonymous namespace

void write_cudftable(data_sink* sink,
                     table_view const& input,
                     compression_type compression,
                     uint32_t block_size,
                     rmm::cuda_stream_view stream)
{
  auto const packed = cudf::pack(input, stream, cudf::get_current_device_resource_ref());

  if (compression == compression_type::NONE) {
    // V1: uncompressed format
    auto const header = cudftable_header_v1{packed.metadata->size(), packed.gpu_data->size()};
    sink->host_write(&header, sizeof(cudftable_header_v1));

    sink->host_write(packed.metadata->data(), header.metadata_length);

    if (sink->is_device_write_preferred(header.data_length)) {
      sink->device_write(packed.gpu_data->data(), header.data_length, stream);
    } else {
      auto host_buffer = cudf::detail::make_host_vector(
        cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(packed.gpu_data->data()),
                                         header.data_length},
        stream);
      sink->host_write(host_buffer.data(), header.data_length);
    }

    sink->flush();
    return;
  }

  // V2: block-compressed format
  CUDF_EXPECTS(cudf::io::detail::is_compression_supported(compression),
               "Unsupported compression type for cudftable");

  auto const data_size  = packed.gpu_data->size();
  auto const num_blocks = data_size == 0 ? uint64_t{0} : (data_size + block_size - 1) / block_size;

  auto const mr = cudf::get_current_device_resource_ref();

  auto const max_comp_block_size = cudf::io::detail::max_compressed_size(compression, block_size);

  // Build input spans (views into packed.gpu_data) and output spans (into a padded output buffer)
  rmm::device_buffer d_compressed(max_comp_block_size * num_blocks, stream, mr);

  std::vector<cudf::device_span<uint8_t const>> h_inputs(num_blocks);
  std::vector<cudf::device_span<uint8_t>> h_outputs(num_blocks);
  for (uint64_t i = 0; i < num_blocks; ++i) {
    auto const offset      = i * block_size;
    auto const uncomp_size = std::min(static_cast<uint64_t>(block_size), data_size - offset);
    h_inputs[i]            = cudf::device_span<uint8_t const>(
      static_cast<uint8_t const*>(packed.gpu_data->data()) + offset, uncomp_size);
    h_outputs[i] = cudf::device_span<uint8_t>(
      static_cast<uint8_t*>(d_compressed.data()) + i * max_comp_block_size, max_comp_block_size);
  }

  auto d_inputs  = cudf::detail::make_device_uvector_async(h_inputs, stream, mr);
  auto d_outputs = cudf::detail::make_device_uvector_async(h_outputs, stream, mr);
  auto d_results = rmm::device_uvector<cudf::io::detail::codec_exec_result>(num_blocks, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    d_results.data(), 0, num_blocks * sizeof(cudf::io::detail::codec_exec_result), stream));

  cudf::io::detail::compress(compression, d_inputs, d_outputs, d_results, stream);

  auto h_results = cudf::detail::make_host_vector(d_results, stream);

  std::vector<block_index_entry> block_index(num_blocks);
  uint64_t total_compressed = 0;
  for (uint64_t i = 0; i < num_blocks; ++i) {
    CUDF_EXPECTS(h_results[i].status == cudf::io::detail::codec_status::SUCCESS,
                 "cudftable block compression failed");
    auto const offset      = i * block_size;
    auto const uncomp_size = std::min(static_cast<uint64_t>(block_size), data_size - offset);
    block_index[i]         = {h_results[i].bytes_written, uncomp_size};
    total_compressed += h_results[i].bytes_written;
  }

  // Compact compressed blocks into a contiguous device buffer, then use device_write
  rmm::device_buffer d_compacted(total_compressed, stream, mr);
  {
    std::vector<void*> h_dsts(num_blocks);
    std::vector<void const*> h_srcs(num_blocks);
    std::vector<std::size_t> h_sizes(num_blocks);
    uint64_t d_offset = 0;
    for (uint64_t i = 0; i < num_blocks; ++i) {
      h_srcs[i]  = static_cast<uint8_t const*>(d_compressed.data()) + i * max_comp_block_size;
      h_dsts[i]  = static_cast<uint8_t*>(d_compacted.data()) + d_offset;
      h_sizes[i] = block_index[i].compressed_size;
      d_offset += block_index[i].compressed_size;
    }

    CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
      h_dsts.data(), h_srcs.data(), h_sizes.data(), num_blocks, stream));
  }

  auto const header = cudftable_header_v2{
    compression, block_size, packed.metadata->size(), data_size, num_blocks, total_compressed};
  sink->host_write(&header, sizeof(cudftable_header_v2));
  sink->host_write(packed.metadata->data(), header.metadata_length);
  sink->host_write(block_index.data(), num_blocks * sizeof(block_index_entry));
  sink->device_write(d_compacted.data(), total_compressed, stream);

  sink->flush();
}

packed_table read_cudftable(datasource* source,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  auto const prefix_size = sizeof(cudftable_header_prefix);
  CUDF_EXPECTS(source->size() >= prefix_size, "File too small to contain a valid cudftable header");

  auto prefix = cudftable_header_prefix{};
  source->host_read(0, prefix_size, reinterpret_cast<uint8_t*>(&prefix));
  CUDF_EXPECTS(prefix.magic == magic_number, "Invalid magic number in cudftable header");

  if (prefix.format_version == cudftable_header_v1::version) {
    // V1: uncompressed format
    auto const header_size = sizeof(cudftable_header_v1);
    CUDF_EXPECTS(source->size() >= header_size,
                 "File too small to contain a valid V1 cudftable header");

    auto header = cudftable_header_v1{};
    source->host_read(0, header_size, reinterpret_cast<uint8_t*>(&header));

    auto const metadata_offset = header_size;
    auto const data_offset     = metadata_offset + header.metadata_length;
    CUDF_EXPECTS(source->size() == data_offset + header.data_length,
                 "File too small for the specified metadata and data sizes");

    auto packed = packed_columns{};
    packed.metadata->resize(header.metadata_length);
    source->host_read(metadata_offset, header.metadata_length, packed.metadata->data());

    packed.gpu_data = std::make_unique<rmm::device_buffer>(header.data_length, stream, mr);
    if (source->is_device_read_preferred(header.data_length)) {
      source->device_read(
        data_offset, header.data_length, static_cast<uint8_t*>(packed.gpu_data->data()), stream);
    } else {
      auto host_buffer = source->host_read(data_offset, header.data_length);
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        packed.gpu_data->data(), host_buffer->data(), header.data_length, stream));
      stream.synchronize();
    }

    auto unpacked_view = cudf::unpack(packed);
    return packed_table{unpacked_view, std::move(packed)};
  }

  if (prefix.format_version == cudftable_header_v2::version) {
    // V2: block-compressed format
    auto const header_size = sizeof(cudftable_header_v2);
    CUDF_EXPECTS(source->size() >= header_size,
                 "File too small to contain a valid V2 cudftable header");

    auto header = cudftable_header_v2{};
    source->host_read(0, header_size, reinterpret_cast<uint8_t*>(&header));

    auto const comp = static_cast<compression_type>(header.compression);
    CUDF_EXPECTS(cudf::io::detail::is_decompression_supported(comp),
                 "Unsupported compression type in cudftable V2 header");

    auto const metadata_offset    = header_size;
    auto const block_index_offset = metadata_offset + header.metadata_length;
    auto const block_index_size   = header.num_blocks * sizeof(block_index_entry);
    auto const blocks_offset      = block_index_offset + block_index_size;
    CUDF_EXPECTS(source->size() == blocks_offset + header.compressed_data_length,
                 "File size mismatch for V2 cudftable");

    // Read metadata
    auto packed = packed_columns{};
    packed.metadata->resize(header.metadata_length);
    source->host_read(metadata_offset, header.metadata_length, packed.metadata->data());

    // Read block index
    std::vector<block_index_entry> block_index(header.num_blocks);
    if (header.num_blocks > 0) {
      source->host_read(
        block_index_offset, block_index_size, reinterpret_cast<uint8_t*>(block_index.data()));
    }

    // Upload compressed data to device
    rmm::device_buffer d_compressed(header.compressed_data_length, stream, mr);
    if (header.compressed_data_length > 0) {
      if (source->is_device_read_preferred(header.compressed_data_length)) {
        source->device_read(blocks_offset,
                            header.compressed_data_length,
                            static_cast<uint8_t*>(d_compressed.data()),
                            stream);
      } else {
        auto host_buffer = source->host_read(blocks_offset, header.compressed_data_length);
        CUDF_CUDA_TRY(cudf::detail::memcpy_async(
          d_compressed.data(), host_buffer->data(), header.compressed_data_length, stream));
      }
    }

    // Allocate output buffer and build device spans for batched decompression
    packed.gpu_data =
      std::make_unique<rmm::device_buffer>(header.uncompressed_data_length, stream, mr);

    std::vector<cudf::device_span<uint8_t const>> h_inputs(header.num_blocks);
    std::vector<cudf::device_span<uint8_t>> h_outputs(header.num_blocks);
    uint64_t comp_offset   = 0;
    uint64_t uncomp_offset = 0;
    for (uint64_t i = 0; i < header.num_blocks; ++i) {
      h_inputs[i] = cudf::device_span<uint8_t const>(
        static_cast<uint8_t const*>(d_compressed.data()) + comp_offset,
        block_index[i].compressed_size);
      h_outputs[i] =
        cudf::device_span<uint8_t>(static_cast<uint8_t*>(packed.gpu_data->data()) + uncomp_offset,
                                   block_index[i].uncompressed_size);
      comp_offset += block_index[i].compressed_size;
      uncomp_offset += block_index[i].uncompressed_size;
    }

    auto d_inputs  = cudf::detail::make_device_uvector_async(h_inputs, stream, mr);
    auto d_outputs = cudf::detail::make_device_uvector_async(h_outputs, stream, mr);
    auto d_results =
      rmm::device_uvector<cudf::io::detail::codec_exec_result>(header.num_blocks, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(d_results.data(),
                                  0,
                                  header.num_blocks * sizeof(cudf::io::detail::codec_exec_result),
                                  stream));

    cudf::io::detail::decompress(comp,
                                 d_inputs,
                                 d_outputs,
                                 d_results,
                                 header.block_size,
                                 header.uncompressed_data_length,
                                 stream);

    auto h_results = cudf::detail::make_host_vector(d_results, stream);
    for (uint64_t i = 0; i < header.num_blocks; ++i) {
      CUDF_EXPECTS(h_results[i].status == cudf::io::detail::codec_status::SUCCESS,
                   "cudftable block decompression failed");
    }

    auto unpacked_view = cudf::unpack(packed);
    return packed_table{unpacked_view, std::move(packed)};
  }

  CUDF_FAIL("Unsupported cudftable format version: " + std::to_string(prefix.format_version));
}

}  // namespace detail

cudftable_writer_options_builder cudftable_writer_options::builder(sink_info const& sink,
                                                                   table_view const& table)
{
  return cudftable_writer_options_builder(sink, table);
}

cudftable_reader_options_builder cudftable_reader_options::builder(source_info src)
{
  return cudftable_reader_options_builder(std::move(src));
}

}  // namespace cudf::io::experimental
