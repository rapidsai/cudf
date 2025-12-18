/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf/detail/nvtx/ranges.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstring>
#include <vector>

namespace cudf::io::experimental {
namespace detail {

namespace {

/**
 * @brief Simple binary file format header for CUDFTable
 *
 * The CUDFTable format stores a table in a simple binary layout:
 * - Magic number (4 bytes): "CDFT"
 * - Version (4 bytes): uint32_t format version (currently 1)
 * - Metadata length (8 bytes): uint64_t size of the metadata buffer in bytes
 * - Data length (8 bytes): uint64_t size of the data buffer in bytes
 * - Metadata (variable): serialized column metadata from pack()
 * - Data (variable): contiguous device data from pack()
 */
struct cudftable_header {
  static constexpr uint32_t magic_number = 0x54464443;  ///< "CDFT" in little-endian
  static constexpr uint32_t version      = 1;           ///< Format version

  uint32_t magic;            ///< Magic number for format validation
  uint32_t format_version;   ///< Format version number
  uint64_t metadata_length;  ///< Length of metadata buffer in bytes
  uint64_t data_length;      ///< Length of data buffer in bytes
};

}  // anonymous namespace

void write_cudftable(data_sink* sink,
                     table_view const& input,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const packed = cudf::pack(input, stream, mr);

  auto const header = cudftable_header{cudftable_header::magic_number,
                                       cudftable_header::version,
                                       packed.metadata->size(),
                                       packed.gpu_data->size()};
  sink->host_write(&header, sizeof(cudftable_header));

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
}

packed_table read_cudftable(datasource* source,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const header_size = sizeof(cudftable_header);
  CUDF_EXPECTS(source->size() >= header_size, "File too small to contain a valid cudftable header");

  auto header = cudftable_header{};
  source->host_read(0, header_size, reinterpret_cast<uint8_t*>(&header));
  CUDF_EXPECTS(header.magic == cudftable_header::magic_number,
               "Invalid magic number in cudftable header");
  CUDF_EXPECTS(header.format_version == cudftable_header::version,
               "Unsupported cudftable format version");

  auto const metadata_offset = header_size;
  auto const data_offset     = metadata_offset + header.metadata_length;
  CUDF_EXPECTS(source->size() >= data_offset + header.data_length,
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
    CUDF_CUDA_TRY(cudaMemcpyAsync(packed.gpu_data->data(),
                                  host_buffer->data(),
                                  header.data_length,
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    stream.synchronize();
  }

  auto unpacked_view = cudf::unpack(packed);

  return packed_table{unpacked_view, std::move(packed)};
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
