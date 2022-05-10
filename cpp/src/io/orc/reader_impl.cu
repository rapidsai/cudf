/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file reader_impl.cu
 * @brief cuDF-IO ORC reader class implementation
 */

#include "orc.h"
#include "orc_gpu.h"
#include "reader_impl.hpp"
#include "timezone.cuh"

#include <io/comp/gpuinflate.h>
#include <io/comp/nvcomp_adapter.hpp>
#include <io/utilities/config_utils.hpp>
#include <io/utilities/time_utils.cuh>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <iterator>

namespace cudf {
namespace io {
namespace detail {
namespace orc {
using namespace cudf::io::orc;

namespace {
/**
 * @brief Function that translates ORC data kind to cuDF type enum
 */
constexpr type_id to_type_id(const orc::SchemaType& schema,
                             bool use_np_dtypes,
                             type_id timestamp_type_id,
                             type_id decimal_type_id)
{
  switch (schema.kind) {
    case orc::BOOLEAN: return type_id::BOOL8;
    case orc::BYTE: return type_id::INT8;
    case orc::SHORT: return type_id::INT16;
    case orc::INT: return type_id::INT32;
    case orc::LONG: return type_id::INT64;
    case orc::FLOAT: return type_id::FLOAT32;
    case orc::DOUBLE: return type_id::FLOAT64;
    case orc::STRING:
    case orc::BINARY:
    case orc::VARCHAR:
    case orc::CHAR:
      // Variable-length types can all be mapped to STRING
      return type_id::STRING;
    case orc::TIMESTAMP:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    case orc::DATE:
      // There isn't a (DAYS -> np.dtype) mapping
      return (use_np_dtypes) ? type_id::TIMESTAMP_MILLISECONDS : type_id::TIMESTAMP_DAYS;
    case orc::DECIMAL: return decimal_type_id;
    // Need to update once cuDF plans to support map type
    case orc::MAP:
    case orc::LIST: return type_id::LIST;
    case orc::STRUCT: return type_id::STRUCT;
    default: break;
  }

  return type_id::EMPTY;
}

constexpr std::pair<gpu::StreamIndexType, uint32_t> get_index_type_and_pos(
  const orc::StreamKind kind, uint32_t skip_count, bool non_child)
{
  switch (kind) {
    case orc::DATA:
      skip_count += 1;
      skip_count |= (skip_count & 0xff) << 8;
      return std::pair(gpu::CI_DATA, skip_count);
    case orc::LENGTH:
    case orc::SECONDARY:
      skip_count += 1;
      skip_count |= (skip_count & 0xff) << 16;
      return std::pair(gpu::CI_DATA2, skip_count);
    case orc::DICTIONARY_DATA: return std::pair(gpu::CI_DICTIONARY, skip_count);
    case orc::PRESENT:
      skip_count += (non_child ? 1 : 0);
      return std::pair(gpu::CI_PRESENT, skip_count);
    case orc::ROW_INDEX: return std::pair(gpu::CI_INDEX, skip_count);
    default:
      // Skip this stream as it's not strictly required
      return std::pair(gpu::CI_NUM_STREAMS, 0);
  }
}

/**
 * @brief struct to store buffer data and size of list buffer
 */
struct list_buffer_data {
  size_type* data;
  size_type size;
};

// Generates offsets for list buffer from number of elements in a row.
void generate_offsets_for_list(rmm::device_uvector<list_buffer_data> const& buff_data,
                               rmm::cuda_stream_view stream)
{
  auto transformer = [] __device__(list_buffer_data list_data) {
    thrust::exclusive_scan(
      thrust::seq, list_data.data, list_data.data + list_data.size, list_data.data);
  };
  thrust::for_each(rmm::exec_policy(stream), buff_data.begin(), buff_data.end(), transformer);
  stream.synchronize();
}

/**
 * @brief Struct that maps ORC streams to columns
 */
struct orc_stream_info {
  orc_stream_info() = default;
  explicit orc_stream_info(
    uint64_t offset_, size_t dst_pos_, uint32_t length_, uint32_t gdf_idx_, uint32_t stripe_idx_)
    : offset(offset_),
      dst_pos(dst_pos_),
      length(length_),
      gdf_idx(gdf_idx_),
      stripe_idx(stripe_idx_)
  {
  }
  uint64_t offset;      // offset in file
  size_t dst_pos;       // offset in memory relative to start of compressed stripe data
  size_t length;        // length in file
  uint32_t gdf_idx;     // column index
  uint32_t stripe_idx;  // stripe index
};

/**
 * @brief Function that populates column descriptors stream/chunk
 */
size_t gather_stream_info(const size_t stripe_index,
                          const orc::StripeInformation* stripeinfo,
                          const orc::StripeFooter* stripefooter,
                          const std::vector<int>& orc2gdf,
                          const std::vector<orc::SchemaType> types,
                          bool use_index,
                          size_t* num_dictionary_entries,
                          cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                          std::vector<orc_stream_info>& stream_info,
                          bool apply_struct_map)
{
  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;
  for (const auto& stream : stripefooter->streams) {
    if (!stream.column_id || *stream.column_id >= orc2gdf.size()) {
      dst_offset += stream.length;
      continue;
    }

    auto const column_id = *stream.column_id;
    auto col             = orc2gdf[column_id];

    if (col == -1 and apply_struct_map) {
      // A struct-type column has no data itself, but rather child columns
      // for each of its fields. There is only a PRESENT stream, which
      // needs to be included for the reader.
      const auto schema_type = types[column_id];
      if (schema_type.subtypes.size() != 0) {
        if (schema_type.kind == orc::STRUCT && stream.kind == orc::PRESENT) {
          for (const auto& idx : schema_type.subtypes) {
            auto child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
            if (child_idx >= 0) {
              col                             = child_idx;
              auto& chunk                     = chunks[stripe_index][col];
              chunk.strm_id[gpu::CI_PRESENT]  = stream_info.size();
              chunk.strm_len[gpu::CI_PRESENT] = stream.length;
            }
          }
        }
      }
    }
    if (col != -1) {
      if (src_offset >= stripeinfo->indexLength || use_index) {
        // NOTE: skip_count field is temporarily used to track index ordering
        auto& chunk = chunks[stripe_index][col];
        const auto idx =
          get_index_type_and_pos(stream.kind, chunk.skip_count, col == orc2gdf[column_id]);
        if (idx.first < gpu::CI_NUM_STREAMS) {
          chunk.strm_id[idx.first]  = stream_info.size();
          chunk.strm_len[idx.first] = stream.length;
          chunk.skip_count          = idx.second;

          if (idx.first == gpu::CI_DICTIONARY) {
            chunk.dictionary_start = *num_dictionary_entries;
            chunk.dict_len         = stripefooter->columns[column_id].dictionarySize;
            *num_dictionary_entries += stripefooter->columns[column_id].dictionarySize;
          }
        }
      }
      stream_info.emplace_back(
        stripeinfo->offset + src_offset, dst_offset, stream.length, col, stripe_index);
      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

/**
 * @brief Determines cuDF type of an ORC Decimal column.
 */
auto decimal_column_type(std::vector<std::string> const& decimal128_columns,
                         cudf::io::orc::detail::aggregate_orc_metadata const& metadata,
                         int column_index)
{
  if (metadata.get_col_type(column_index).kind != DECIMAL) { return type_id::EMPTY; }

  if (std::find(decimal128_columns.cbegin(),
                decimal128_columns.cend(),
                metadata.column_path(0, column_index)) != decimal128_columns.end()) {
    return type_id::DECIMAL128;
  }

  auto const precision = metadata.get_col_type(column_index)
                           .precision.value_or(cuda::std::numeric_limits<int64_t>::digits10);
  if (precision <= cuda::std::numeric_limits<int32_t>::digits10) { return type_id::DECIMAL32; }
  if (precision <= cuda::std::numeric_limits<int64_t>::digits10) { return type_id::DECIMAL64; }
  return type_id::DECIMAL128;
}

}  // namespace

__global__ void decompress_check_kernel(device_span<decompress_status const> stats,
                                        bool* any_block_failure)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < stats.size()) {
    if (stats[tid].status != 0) {
      *any_block_failure = true;  // Doesn't need to be atomic
    }
  }
}

void decompress_check(device_span<decompress_status> stats,
                      bool* any_block_failure,
                      rmm::cuda_stream_view stream)
{
  if (stats.empty()) { return; }  // early exit for empty stats

  dim3 block(128);
  dim3 grid(cudf::util::div_rounding_up_safe(stats.size(), static_cast<size_t>(block.x)));
  decompress_check_kernel<<<grid, block, 0, stream.value()>>>(stats, any_block_failure);
}

rmm::device_buffer reader::impl::decompress_stripe_data(
  cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
  const std::vector<rmm::device_buffer>& stripe_data,
  const OrcDecompressor* decompressor,
  std::vector<orc_stream_info>& stream_info,
  size_t num_stripes,
  cudf::detail::hostdevice_2dvector<gpu::RowGroup>& row_groups,
  size_t row_index_stride,
  bool use_base_stride,
  rmm::cuda_stream_view stream)
{
  // For checking whether we decompress successfully
  hostdevice_vector<bool> any_block_failure(1, stream);
  any_block_failure[0] = false;
  any_block_failure.host_to_device(stream);

  // Parse the columns' compressed info
  hostdevice_vector<gpu::CompressedStreamInfo> compinfo(0, stream_info.size(), stream);
  for (const auto& info : stream_info) {
    compinfo.insert(gpu::CompressedStreamInfo(
      static_cast<const uint8_t*>(stripe_data[info.stripe_idx].data()) + info.dst_pos,
      info.length));
  }
  compinfo.host_to_device(stream);

  gpu::ParseCompressedStripeData(compinfo.device_ptr(),
                                 compinfo.size(),
                                 decompressor->GetBlockSize(),
                                 decompressor->GetLog2MaxCompressionRatio(),
                                 stream);
  compinfo.device_to_host(stream, true);

  // Count the exact number of compressed blocks
  size_t num_compressed_blocks   = 0;
  size_t num_uncompressed_blocks = 0;
  size_t total_decomp_size       = 0;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    num_compressed_blocks += compinfo[i].num_compressed_blocks;
    num_uncompressed_blocks += compinfo[i].num_uncompressed_blocks;
    total_decomp_size += compinfo[i].max_uncompressed_size;
  }
  CUDF_EXPECTS(total_decomp_size > 0, "No decompressible data found");

  rmm::device_buffer decomp_data(total_decomp_size, stream);
  rmm::device_uvector<device_span<uint8_t const>> inflate_in(
    num_compressed_blocks + num_uncompressed_blocks, stream);
  rmm::device_uvector<device_span<uint8_t>> inflate_out(
    num_compressed_blocks + num_uncompressed_blocks, stream);
  rmm::device_uvector<decompress_status> inflate_stats(num_compressed_blocks, stream);

  // Parse again to populate the decompression input/output buffers
  size_t decomp_offset           = 0;
  uint32_t max_uncomp_block_size = 0;
  uint32_t start_pos             = 0;
  auto start_pos_uncomp          = (uint32_t)num_compressed_blocks;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    auto dst_base                 = static_cast<uint8_t*>(decomp_data.data());
    compinfo[i].uncompressed_data = dst_base + decomp_offset;
    compinfo[i].dec_in_ctl        = inflate_in.data() + start_pos;
    compinfo[i].dec_out_ctl       = inflate_out.data() + start_pos;
    compinfo[i].decstatus   = {inflate_stats.data() + start_pos, compinfo[i].num_compressed_blocks};
    compinfo[i].copy_in_ctl = inflate_in.data() + start_pos_uncomp;
    compinfo[i].copy_out_ctl = inflate_out.data() + start_pos_uncomp;

    stream_info[i].dst_pos = decomp_offset;
    decomp_offset += compinfo[i].max_uncompressed_size;
    start_pos += compinfo[i].num_compressed_blocks;
    start_pos_uncomp += compinfo[i].num_uncompressed_blocks;
    max_uncomp_block_size =
      std::max(max_uncomp_block_size, compinfo[i].max_uncompressed_block_size);
  }
  compinfo.host_to_device(stream);
  gpu::ParseCompressedStripeData(compinfo.device_ptr(),
                                 compinfo.size(),
                                 decompressor->GetBlockSize(),
                                 decompressor->GetLog2MaxCompressionRatio(),
                                 stream);

  // Dispatch batches of blocks to decompress
  if (num_compressed_blocks > 0) {
    device_span<device_span<uint8_t const>> inflate_in_view{inflate_in.data(),
                                                            num_compressed_blocks};
    device_span<device_span<uint8_t>> inflate_out_view{inflate_out.data(), num_compressed_blocks};
    switch (decompressor->GetKind()) {
      case orc::ZLIB:
        gpuinflate(
          inflate_in_view, inflate_out_view, inflate_stats, gzip_header_included::NO, stream);
        break;
      case orc::SNAPPY:
        if (nvcomp_integration::is_stable_enabled()) {
          nvcomp::batched_decompress(nvcomp::compression_type::SNAPPY,
                                     inflate_in_view,
                                     inflate_out_view,
                                     inflate_stats,
                                     max_uncomp_block_size,
                                     stream);
        } else {
          gpu_unsnap(inflate_in_view, inflate_out_view, inflate_stats, stream);
        }
        break;
      default: CUDF_FAIL("Unexpected decompression dispatch"); break;
    }
    decompress_check(inflate_stats, any_block_failure.device_ptr(), stream);
  }
  if (num_uncompressed_blocks > 0) {
    device_span<device_span<uint8_t const>> copy_in_view{inflate_in.data() + num_compressed_blocks,
                                                         num_uncompressed_blocks};
    device_span<device_span<uint8_t>> copy_out_view{inflate_out.data() + num_compressed_blocks,
                                                    num_uncompressed_blocks};
    gpu_copy_uncompressed_blocks(copy_in_view, copy_out_view, stream);
  }
  gpu::PostDecompressionReassemble(compinfo.device_ptr(), compinfo.size(), stream);

  any_block_failure.device_to_host(stream);

  compinfo.device_to_host(stream, true);

  // We can check on host after stream synchronize
  CUDF_EXPECTS(not any_block_failure[0], "Error during decompression");

  const size_t num_columns = chunks.size().second;

  // Update the stream information with the updated uncompressed info
  // TBD: We could update the value from the information we already
  // have in stream_info[], but using the gpu results also updates
  // max_uncompressed_size to the actual uncompressed size, or zero if
  // decompression failed.
  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto& chunk = chunks[i][j];
      for (int k = 0; k < gpu::CI_NUM_STREAMS; ++k) {
        if (chunk.strm_len[k] > 0 && chunk.strm_id[k] < compinfo.size()) {
          chunk.streams[k]  = compinfo[chunk.strm_id[k]].uncompressed_data;
          chunk.strm_len[k] = compinfo[chunk.strm_id[k]].max_uncompressed_size;
        }
      }
    }
  }

  if (row_groups.size().first) {
    chunks.host_to_device(stream);
    row_groups.host_to_device(stream);
    gpu::ParseRowGroupIndex(row_groups.base_device_ptr(),
                            compinfo.device_ptr(),
                            chunks.base_device_ptr(),
                            num_columns,
                            num_stripes,
                            row_groups.size().first,
                            row_index_stride,
                            use_base_stride,
                            stream);
  }

  return decomp_data;
}

/**
 * @brief Updates null mask of columns whose parent is a struct column.
 *        If struct column has null element, that row would be
 *        skipped while writing child column in ORC, so we need to insert the missing null
 *        elements in child column.
 *        There is another behavior from pyspark, where if the child column doesn't have any null
 *        elements, it will not have present stream, so in that case parent null mask need to be
 *        copied to child column.
 *
 * @param chunks Vector of list of column chunk descriptors
 * @param out_buffers Output columns' device buffers
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource to use for device memory allocation
 */
void update_null_mask(cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                      std::vector<column_buffer>& out_buffers,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
{
  const auto num_stripes = chunks.size().first;
  const auto num_columns = chunks.size().second;
  bool is_mask_updated   = false;

  for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    if (chunks[0][col_idx].parent_validity_info.valid_map_base != nullptr) {
      if (not is_mask_updated) {
        chunks.device_to_host(stream, true);
        is_mask_updated = true;
      }

      auto parent_valid_map_base = chunks[0][col_idx].parent_validity_info.valid_map_base;
      auto child_valid_map_base  = out_buffers[col_idx].null_mask();
      auto child_mask_len =
        chunks[0][col_idx].column_num_rows - chunks[0][col_idx].parent_validity_info.null_count;
      auto parent_mask_len = chunks[0][col_idx].column_num_rows;

      if (child_valid_map_base != nullptr) {
        rmm::device_uvector<uint32_t> dst_idx(child_mask_len, stream);
        // Copy indexes at which the parent has valid value.
        thrust::copy_if(rmm::exec_policy(stream),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(0) + parent_mask_len,
                        dst_idx.begin(),
                        [parent_valid_map_base] __device__(auto idx) {
                          return bit_is_set(parent_valid_map_base, idx);
                        });

        auto merged_null_mask = cudf::detail::create_null_mask(
          parent_mask_len, mask_state::ALL_NULL, rmm::cuda_stream_view(stream), mr);
        auto merged_mask      = static_cast<bitmask_type*>(merged_null_mask.data());
        uint32_t* dst_idx_ptr = dst_idx.data();
        // Copy child valid bits from child column to valid indexes, this will merge both child and
        // parent null masks
        thrust::for_each(rmm::exec_policy(stream),
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(0) + dst_idx.size(),
                         [child_valid_map_base, dst_idx_ptr, merged_mask] __device__(auto idx) {
                           if (bit_is_set(child_valid_map_base, idx)) {
                             cudf::set_bit(merged_mask, dst_idx_ptr[idx]);
                           };
                         });

        out_buffers[col_idx]._null_mask = std::move(merged_null_mask);

      } else {
        // Since child column doesn't have a mask, copy parent null mask
        auto mask_size = bitmask_allocation_size_bytes(parent_mask_len);
        out_buffers[col_idx]._null_mask =
          rmm::device_buffer(static_cast<void*>(parent_valid_map_base), mask_size, stream, mr);
      }
    }
  }

  thrust::counting_iterator<int> col_idx_it(0);
  thrust::counting_iterator<int> stripe_idx_it(0);

  if (is_mask_updated) {
    // Update chunks with pointers to column data which might have been changed.
    std::for_each(stripe_idx_it, stripe_idx_it + num_stripes, [&](auto stripe_idx) {
      std::for_each(col_idx_it, col_idx_it + num_columns, [&](auto col_idx) {
        auto& chunk          = chunks[stripe_idx][col_idx];
        chunk.valid_map_base = out_buffers[col_idx].null_mask();
      });
    });
    chunks.host_to_device(stream, true);
  }
}

/**
 * @brief Compute the per-stripe prefix sum of null count, for each struct column in the current
 * layer.
 */
void scan_null_counts(cudf::detail::hostdevice_2dvector<gpu::ColumnDesc> const& chunks,
                      cudf::host_span<rmm::device_uvector<uint32_t>> prefix_sums,
                      rmm::cuda_stream_view stream)
{
  auto const num_stripes = chunks.size().first;
  if (num_stripes == 0) return;

  auto const num_columns = chunks.size().second;
  std::vector<thrust::pair<size_type, cudf::device_span<uint32_t>>> prefix_sums_to_update;
  for (auto col_idx = 0ul; col_idx < num_columns; ++col_idx) {
    // Null counts sums are only needed for children of struct columns
    if (chunks[0][col_idx].type_kind == STRUCT) {
      prefix_sums_to_update.emplace_back(col_idx, prefix_sums[col_idx]);
    }
  }
  auto const d_prefix_sums_to_update =
    cudf::detail::make_device_uvector_async(prefix_sums_to_update, stream);

  thrust::for_each(rmm::exec_policy(stream),
                   d_prefix_sums_to_update.begin(),
                   d_prefix_sums_to_update.end(),
                   [chunks = cudf::detail::device_2dspan<gpu::ColumnDesc const>{chunks}] __device__(
                     auto const& idx_psums) {
                     auto const col_idx = idx_psums.first;
                     auto const psums   = idx_psums.second;

                     thrust::transform(
                       thrust::seq,
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(0) + psums.size(),
                       psums.begin(),
                       [&](auto stripe_idx) { return chunks[stripe_idx][col_idx].null_count; });

                     thrust::inclusive_scan(thrust::seq, psums.begin(), psums.end(), psums.begin());
                   });
  // `prefix_sums_to_update` goes out of scope, copy has to be done before we return
  stream.synchronize();
}

void reader::impl::decode_stream_data(cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                                      size_t num_dicts,
                                      size_t skip_rows,
                                      timezone_table_view tz_table,
                                      cudf::detail::hostdevice_2dvector<gpu::RowGroup>& row_groups,
                                      size_t row_index_stride,
                                      std::vector<column_buffer>& out_buffers,
                                      size_t level,
                                      rmm::cuda_stream_view stream)
{
  const auto num_stripes = chunks.size().first;
  const auto num_columns = chunks.size().second;
  thrust::counting_iterator<int> col_idx_it(0);
  thrust::counting_iterator<int> stripe_idx_it(0);

  // Update chunks with pointers to column data
  std::for_each(stripe_idx_it, stripe_idx_it + num_stripes, [&](auto stripe_idx) {
    std::for_each(col_idx_it, col_idx_it + num_columns, [&](auto col_idx) {
      auto& chunk            = chunks[stripe_idx][col_idx];
      chunk.column_data_base = out_buffers[col_idx].data();
      chunk.valid_map_base   = out_buffers[col_idx].null_mask();
    });
  });

  // Allocate global dictionary for deserializing
  rmm::device_uvector<gpu::DictionaryEntry> global_dict(num_dicts, stream);

  chunks.host_to_device(stream, true);
  gpu::DecodeNullsAndStringDictionaries(
    chunks.base_device_ptr(), global_dict.data(), num_columns, num_stripes, skip_rows, stream);

  if (level > 0) {
    // Update nullmasks for children if parent was a struct and had null mask
    update_null_mask(chunks, out_buffers, stream, _mr);
  }

  // Update the null map for child columns
  gpu::DecodeOrcColumnData(chunks.base_device_ptr(),
                           global_dict.data(),
                           row_groups,
                           num_columns,
                           num_stripes,
                           skip_rows,
                           tz_table,
                           row_groups.size().first,
                           row_index_stride,
                           level,
                           stream);
  chunks.device_to_host(stream, true);

  std::for_each(col_idx_it + 0, col_idx_it + num_columns, [&](auto col_idx) {
    out_buffers[col_idx].null_count() =
      std::accumulate(stripe_idx_it + 0,
                      stripe_idx_it + num_stripes,
                      0,
                      [&](auto null_count, auto const stripe_idx) {
                        return null_count + chunks[stripe_idx][col_idx].null_count;
                      });
  });
}

// Aggregate child column metadata per stripe and per column
void reader::impl::aggregate_child_meta(cudf::detail::host_2dspan<gpu::ColumnDesc> chunks,
                                        cudf::detail::host_2dspan<gpu::RowGroup> row_groups,
                                        std::vector<column_buffer>& out_buffers,
                                        std::vector<orc_column_meta> const& list_col,
                                        const size_type level)
{
  const auto num_of_stripes         = chunks.size().first;
  const auto num_of_rowgroups       = row_groups.size().first;
  const auto num_parent_cols        = selected_columns.levels[level].size();
  const auto num_child_cols         = selected_columns.levels[level + 1].size();
  const auto number_of_child_chunks = num_child_cols * num_of_stripes;
  auto& num_child_rows              = _col_meta.num_child_rows;
  auto& parent_column_data          = _col_meta.parent_column_data;

  // Reset the meta to store child column details.
  num_child_rows.resize(selected_columns.levels[level + 1].size());
  std::fill(num_child_rows.begin(), num_child_rows.end(), 0);
  parent_column_data.resize(number_of_child_chunks);
  _col_meta.parent_column_index.resize(number_of_child_chunks);
  _col_meta.child_start_row.resize(number_of_child_chunks);
  _col_meta.num_child_rows_per_stripe.resize(number_of_child_chunks);
  _col_meta.rwgrp_meta.resize(num_of_rowgroups * num_child_cols);

  auto child_start_row = cudf::detail::host_2dspan<uint32_t>(
    _col_meta.child_start_row.data(), num_of_stripes, num_child_cols);
  auto num_child_rows_per_stripe = cudf::detail::host_2dspan<uint32_t>(
    _col_meta.num_child_rows_per_stripe.data(), num_of_stripes, num_child_cols);
  auto rwgrp_meta = cudf::detail::host_2dspan<reader_column_meta::row_group_meta>(
    _col_meta.rwgrp_meta.data(), num_of_rowgroups, num_child_cols);

  int index = 0;  // number of child column processed

  // For each parent column, update its child column meta for each stripe.
  std::for_each(list_col.cbegin(), list_col.cend(), [&](const auto p_col) {
    const auto parent_col_idx = _col_meta.orc_col_map[level][p_col.id];
    auto start_row            = 0;
    auto processed_row_groups = 0;

    for (size_t stripe_id = 0; stripe_id < num_of_stripes; stripe_id++) {
      // Aggregate num_rows and start_row from processed parent columns per row groups
      if (num_of_rowgroups) {
        auto stripe_num_row_groups = chunks[stripe_id][parent_col_idx].num_rowgroups;
        auto processed_child_rows  = 0;

        for (size_t rowgroup_id = 0; rowgroup_id < stripe_num_row_groups;
             rowgroup_id++, processed_row_groups++) {
          const auto child_rows = row_groups[processed_row_groups][parent_col_idx].num_child_rows;
          for (size_type id = 0; id < p_col.num_children; id++) {
            const auto child_col_idx                                  = index + id;
            rwgrp_meta[processed_row_groups][child_col_idx].start_row = processed_child_rows;
            rwgrp_meta[processed_row_groups][child_col_idx].num_rows  = child_rows;
          }
          processed_child_rows += child_rows;
        }
      }

      // Aggregate start row, number of rows per chunk and total number of rows in a column
      const auto child_rows = chunks[stripe_id][parent_col_idx].num_child_rows;
      for (size_type id = 0; id < p_col.num_children; id++) {
        const auto child_col_idx = index + id;

        num_child_rows[child_col_idx] += child_rows;
        num_child_rows_per_stripe[stripe_id][child_col_idx] = child_rows;
        // start row could be different for each column when there is nesting at each stripe level
        child_start_row[stripe_id][child_col_idx] = (stripe_id == 0) ? 0 : start_row;
      }
      start_row += child_rows;
    }

    // Parent column null mask and null count would be required for child column
    // to adjust its nullmask.
    auto type              = out_buffers[parent_col_idx].type.id();
    auto parent_null_count = static_cast<uint32_t>(out_buffers[parent_col_idx].null_count());
    auto parent_valid_map  = out_buffers[parent_col_idx].null_mask();
    auto num_rows          = out_buffers[parent_col_idx].size;

    for (size_type id = 0; id < p_col.num_children; id++) {
      const auto child_col_idx                     = index + id;
      _col_meta.parent_column_index[child_col_idx] = parent_col_idx;
      if (type == type_id::STRUCT) {
        parent_column_data[child_col_idx] = {parent_valid_map, parent_null_count};
        // Number of rows in child will remain same as parent in case of struct column
        num_child_rows[child_col_idx] = num_rows;
      } else {
        parent_column_data[child_col_idx] = {nullptr, 0};
      }
    }
    index += p_col.num_children;
  });
}

std::string get_map_child_col_name(size_t const idx) { return (idx == 0) ? "key" : "value"; }

std::unique_ptr<column> reader::impl::create_empty_column(const size_type orc_col_id,
                                                          column_name_info& schema_info,
                                                          rmm::cuda_stream_view stream)
{
  schema_info.name = _metadata.column_name(0, orc_col_id);
  auto const type  = to_type_id(_metadata.get_schema(orc_col_id),
                               _use_np_dtypes,
                               _timestamp_type.id(),
                               decimal_column_type(decimal128_columns, _metadata, orc_col_id));
  int32_t scale    = 0;
  std::vector<std::unique_ptr<column>> child_columns;
  std::unique_ptr<column> out_col = nullptr;
  auto kind                       = _metadata.get_col_type(orc_col_id).kind;

  switch (kind) {
    case orc::LIST:
      schema_info.children.emplace_back("offsets");
      schema_info.children.emplace_back("");
      out_col = make_lists_column(
        0,
        make_empty_column(type_id::INT32),
        create_empty_column(
          _metadata.get_col_type(orc_col_id).subtypes[0], schema_info.children.back(), stream),
        0,
        rmm::device_buffer{0, stream},
        stream);
      break;
    case orc::MAP: {
      schema_info.children.emplace_back("offsets");
      schema_info.children.emplace_back("struct");
      const auto child_column_ids = _metadata.get_col_type(orc_col_id).subtypes;
      for (size_t idx = 0; idx < _metadata.get_col_type(orc_col_id).subtypes.size(); idx++) {
        auto& children_schema = schema_info.children.back().children;
        children_schema.emplace_back("");
        child_columns.push_back(create_empty_column(
          child_column_ids[idx], schema_info.children.back().children.back(), stream));
        auto name                 = get_map_child_col_name(idx);
        children_schema[idx].name = name;
      }
      auto struct_col =
        make_structs_column(0, std::move(child_columns), 0, rmm::device_buffer{0, stream}, stream);
      out_col = make_lists_column(0,
                                  make_empty_column(type_id::INT32),
                                  std::move(struct_col),
                                  0,
                                  rmm::device_buffer{0, stream},
                                  stream);
    } break;

    case orc::STRUCT:
      for (const auto col : _metadata.get_col_type(orc_col_id).subtypes) {
        schema_info.children.emplace_back("");
        child_columns.push_back(create_empty_column(col, schema_info.children.back(), stream));
      }
      out_col =
        make_structs_column(0, std::move(child_columns), 0, rmm::device_buffer{0, stream}, stream);
      break;

    case orc::DECIMAL:
      if (type == type_id::DECIMAL32 or type == type_id::DECIMAL64 or type == type_id::DECIMAL128) {
        scale = -static_cast<int32_t>(_metadata.get_types()[orc_col_id].scale.value_or(0));
      }
      out_col = make_empty_column(data_type(type, scale));
      break;

    default: out_col = make_empty_column(type);
  }

  return out_col;
}

// Adds child column buffers to parent column
column_buffer&& reader::impl::assemble_buffer(const size_type orc_col_id,
                                              std::vector<std::vector<column_buffer>>& col_buffers,
                                              const size_t level,
                                              rmm::cuda_stream_view stream)
{
  auto const col_id = _col_meta.orc_col_map[level][orc_col_id];
  auto& col_buffer  = col_buffers[level][col_id];

  col_buffer.name = _metadata.column_name(0, orc_col_id);
  auto kind       = _metadata.get_col_type(orc_col_id).kind;
  switch (kind) {
    case orc::LIST:
    case orc::STRUCT:
      for (auto const& col : selected_columns.children[orc_col_id]) {
        col_buffer.children.emplace_back(assemble_buffer(col, col_buffers, level + 1, stream));
      }

      break;
    case orc::MAP: {
      std::vector<column_buffer> child_col_buffers;
      // Get child buffers
      for (size_t idx = 0; idx < selected_columns.children[orc_col_id].size(); idx++) {
        auto name = get_map_child_col_name(idx);
        auto col  = selected_columns.children[orc_col_id][idx];
        child_col_buffers.emplace_back(assemble_buffer(col, col_buffers, level + 1, stream));
        child_col_buffers.back().name = name;
      }
      // Create a struct buffer
      auto num_rows = child_col_buffers[0].size;
      auto struct_buffer =
        column_buffer(cudf::data_type(type_id::STRUCT), num_rows, false, stream, _mr);
      struct_buffer.children = std::move(child_col_buffers);
      struct_buffer.name     = "struct";

      col_buffer.children.emplace_back(std::move(struct_buffer));
    } break;

    default: break;
  }

  return std::move(col_buffer);
}

// creates columns along with schema information for each column
void reader::impl::create_columns(std::vector<std::vector<column_buffer>>&& col_buffers,
                                  std::vector<std::unique_ptr<column>>& out_columns,
                                  std::vector<column_name_info>& schema_info,
                                  rmm::cuda_stream_view stream)
{
  std::transform(selected_columns.levels[0].begin(),
                 selected_columns.levels[0].end(),
                 std::back_inserter(out_columns),
                 [&](auto const col_meta) {
                   schema_info.emplace_back("");
                   auto col_buffer = assemble_buffer(col_meta.id, col_buffers, 0, stream);
                   return make_column(col_buffer, &schema_info.back(), stream, _mr);
                 });
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   orc_reader_options const& options,
                   rmm::mr::device_memory_resource* mr)
  : _mr(mr),
    _sources(std::move(sources)),
    _metadata{_sources},
    selected_columns{_metadata.select_columns(options.get_columns())}
{
  // Override output timestamp resolution if requested
  if (options.get_timestamp_type().id() != type_id::EMPTY) {
    _timestamp_type = options.get_timestamp_type();
  }

  // Enable or disable attempt to use row index for parsing
  _use_index = options.is_enabled_use_index();

  // Enable or disable the conversion to numpy-compatible dtypes
  _use_np_dtypes = options.is_enabled_use_np_dtypes();

  // Control decimals conversion
  decimal128_columns = options.get_decimal128_columns();
}

timezone_table reader::impl::compute_timezone_table(
  const std::vector<cudf::io::orc::metadata::stripe_source_mapping>& selected_stripes,
  rmm::cuda_stream_view stream)
{
  if (selected_stripes.empty()) return {};

  auto const has_timestamp_column = std::any_of(
    selected_columns.levels.cbegin(), selected_columns.levels.cend(), [&](auto& col_lvl) {
      return std::any_of(col_lvl.cbegin(), col_lvl.cend(), [&](auto& col_meta) {
        return _metadata.get_col_type(col_meta.id).kind == TypeKind::TIMESTAMP;
      });
    });
  if (not has_timestamp_column) return {};

  return build_timezone_transition_table(selected_stripes[0].stripe_info[0].second->writerTimezone,
                                         stream);
}

table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       const std::vector<std::vector<size_type>>& stripes,
                                       rmm::cuda_stream_view stream)
{
  // Selected columns at different levels of nesting are stored in different elements
  // of `selected_columns`; thus, size == 1 means no nested columns
  CUDF_EXPECTS(skip_rows == 0 or selected_columns.num_levels() == 1,
               "skip_rows is not supported by nested columns");

  std::vector<std::unique_ptr<column>> out_columns;
  // buffer and stripe data are stored as per nesting level
  std::vector<std::vector<column_buffer>> out_buffers(selected_columns.num_levels());
  std::vector<column_name_info> schema_info;
  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data(selected_columns.num_levels());
  std::vector<std::vector<rmm::device_uvector<uint32_t>>> null_count_prefix_sums;
  table_metadata out_metadata;

  // There are no columns in the table
  if (selected_columns.num_levels() == 0)
    return {std::make_unique<table>(), std::move(out_metadata)};

  // Select only stripes required (aka row groups)
  const auto selected_stripes = _metadata.select_stripes(stripes, skip_rows, num_rows);

  auto const tz_table = compute_timezone_table(selected_stripes, stream);

  // Iterates through levels of nested columns, child column will be one level down
  // compared to parent column.
  for (size_t level = 0; level < selected_columns.num_levels(); level++) {
    auto& columns_level = selected_columns.levels[level];
    // Association between each ORC column and its cudf::column
    _col_meta.orc_col_map.emplace_back(_metadata.get_num_cols(), -1);
    std::vector<orc_column_meta> nested_col;
    bool is_data_empty = false;

    // Get a list of column data types
    std::vector<data_type> column_types;
    for (auto& col : columns_level) {
      auto col_type = to_type_id(_metadata.get_col_type(col.id),
                                 _use_np_dtypes,
                                 _timestamp_type.id(),
                                 decimal_column_type(decimal128_columns, _metadata, col.id));
      CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
      if (col_type == type_id::DECIMAL32 or col_type == type_id::DECIMAL64 or
          col_type == type_id::DECIMAL128) {
        // sign of the scale is changed since cuDF follows c++ libraries like CNL
        // which uses negative scaling, but liborc and other libraries
        // follow positive scaling.
        auto const scale =
          -static_cast<size_type>(_metadata.get_col_type(col.id).scale.value_or(0));
        column_types.emplace_back(col_type, scale);
      } else {
        column_types.emplace_back(col_type);
      }

      // Map each ORC column to its column
      _col_meta.orc_col_map[level][col.id] = column_types.size() - 1;
      // TODO: Once MAP type is supported in cuDF, update this for MAP as well
      if (col_type == type_id::LIST or col_type == type_id::STRUCT) nested_col.emplace_back(col);
    }

    // If no rows or stripes to read, return empty columns
    if (num_rows <= 0 || selected_stripes.empty()) {
      std::transform(selected_columns.levels[0].begin(),
                     selected_columns.levels[0].end(),
                     std::back_inserter(out_columns),
                     [&](auto const col_meta) {
                       schema_info.emplace_back("");
                       return create_empty_column(col_meta.id, schema_info.back(), stream);
                     });
      break;
    } else {
      // Get the total number of stripes across all input files.
      size_t total_num_stripes =
        std::accumulate(selected_stripes.begin(),
                        selected_stripes.end(),
                        0,
                        [](size_t sum, auto& stripe_source_mapping) {
                          return sum + stripe_source_mapping.stripe_info.size();
                        });
      const auto num_columns = columns_level.size();
      cudf::detail::hostdevice_2dvector<gpu::ColumnDesc> chunks(
        total_num_stripes, num_columns, stream);
      memset(chunks.base_host_ptr(), 0, chunks.memory_size());

      const bool use_index =
        (_use_index == true) &&
        // Do stripes have row group index
        _metadata.is_row_grp_idx_present() &&
        // Only use if we don't have much work with complete columns & stripes
        // TODO: Consider nrows, gpu, and tune the threshold
        (num_rows > _metadata.get_row_index_stride() && !(_metadata.get_row_index_stride() & 7) &&
         _metadata.get_row_index_stride() > 0 && num_columns * total_num_stripes < 8 * 128) &&
        // Only use if first row is aligned to a stripe boundary
        // TODO: Fix logic to handle unaligned rows
        (skip_rows == 0);

      // Logically view streams as columns
      std::vector<orc_stream_info> stream_info;

      null_count_prefix_sums.emplace_back();
      null_count_prefix_sums.back().reserve(selected_columns.levels[level].size());
      std::generate_n(std::back_inserter(null_count_prefix_sums.back()),
                      selected_columns.levels[level].size(),
                      [&]() {
                        return cudf::detail::make_zeroed_device_uvector_async<uint32_t>(
                          total_num_stripes, stream);
                      });

      // Tracker for eventually deallocating compressed and uncompressed data
      auto& stripe_data = lvl_stripe_data[level];

      size_t stripe_start_row = 0;
      size_t num_dict_entries = 0;
      size_t num_rowgroups    = 0;
      int stripe_idx          = 0;

      std::vector<std::pair<std::future<size_t>, size_t>> read_tasks;
      for (auto const& stripe_source_mapping : selected_stripes) {
        // Iterate through the source files selected stripes
        for (auto const& stripe : stripe_source_mapping.stripe_info) {
          const auto stripe_info   = stripe.first;
          const auto stripe_footer = stripe.second;

          auto stream_count          = stream_info.size();
          const auto total_data_size = gather_stream_info(stripe_idx,
                                                          stripe_info,
                                                          stripe_footer,
                                                          _col_meta.orc_col_map[level],
                                                          _metadata.get_types(),
                                                          use_index,
                                                          &num_dict_entries,
                                                          chunks,
                                                          stream_info,
                                                          level == 0);

          if (total_data_size == 0) {
            CUDF_EXPECTS(stripe_info->indexLength == 0, "Invalid index rowgroup stream data");
            // In case ROW GROUP INDEX is not present and all columns are structs with no null
            // stream, there is nothing to read at this level.
            auto fn_check_dtype = [](auto dtype) { return dtype.id() == type_id::STRUCT; };
            CUDF_EXPECTS(std::all_of(column_types.begin(), column_types.end(), fn_check_dtype),
                         "Expected streams data within stripe");
            is_data_empty = true;
          }

          stripe_data.emplace_back(total_data_size, stream);
          auto dst_base = static_cast<uint8_t*>(stripe_data.back().data());

          // Coalesce consecutive streams into one read
          while (not is_data_empty and stream_count < stream_info.size()) {
            const auto d_dst  = dst_base + stream_info[stream_count].dst_pos;
            const auto offset = stream_info[stream_count].offset;
            auto len          = stream_info[stream_count].length;
            stream_count++;

            while (stream_count < stream_info.size() &&
                   stream_info[stream_count].offset == offset + len) {
              len += stream_info[stream_count].length;
              stream_count++;
            }
            if (_metadata.per_file_metadata[stripe_source_mapping.source_idx]
                  .source->is_device_read_preferred(len)) {
              read_tasks.push_back(
                std::pair(_metadata.per_file_metadata[stripe_source_mapping.source_idx]
                            .source->device_read_async(offset, len, d_dst, stream),
                          len));

            } else {
              const auto buffer =
                _metadata.per_file_metadata[stripe_source_mapping.source_idx].source->host_read(
                  offset, len);
              CUDF_EXPECTS(buffer->size() == len, "Unexpected discrepancy in bytes read.");
              CUDF_CUDA_TRY(cudaMemcpyAsync(
                d_dst, buffer->data(), len, cudaMemcpyHostToDevice, stream.value()));
              stream.synchronize();
            }
          }

          const auto num_rows_per_stripe = stripe_info->numberOfRows;
          const auto rowgroup_id         = num_rowgroups;
          auto stripe_num_rowgroups      = 0;
          if (use_index) {
            stripe_num_rowgroups = (num_rows_per_stripe + _metadata.get_row_index_stride() - 1) /
                                   _metadata.get_row_index_stride();
          }
          // Update chunks to reference streams pointers
          for (size_t col_idx = 0; col_idx < num_columns; col_idx++) {
            auto& chunk = chunks[stripe_idx][col_idx];
            // start row, number of rows in a each stripe and total number of rows
            // may change in lower levels of nesting
            chunk.start_row = (level == 0)
                                ? stripe_start_row
                                : _col_meta.child_start_row[stripe_idx * num_columns + col_idx];
            chunk.num_rows =
              (level == 0)
                ? stripe_info->numberOfRows
                : _col_meta.num_child_rows_per_stripe[stripe_idx * num_columns + col_idx];
            chunk.column_num_rows = (level == 0) ? num_rows : _col_meta.num_child_rows[col_idx];
            chunk.parent_validity_info =
              (level == 0) ? column_validity_info{} : _col_meta.parent_column_data[col_idx];
            chunk.parent_null_count_prefix_sums =
              (level == 0)
                ? nullptr
                : null_count_prefix_sums[level - 1][_col_meta.parent_column_index[col_idx]].data();
            chunk.encoding_kind = stripe_footer->columns[columns_level[col_idx].id].kind;
            chunk.type_kind     = _metadata.per_file_metadata[stripe_source_mapping.source_idx]
                                .ff.types[columns_level[col_idx].id]
                                .kind;
            // num_child_rows for a struct column will be same, for other nested types it will be
            // calculated.
            chunk.num_child_rows = (chunk.type_kind != orc::STRUCT) ? 0 : chunk.num_rows;
            chunk.dtype_id       = column_types[col_idx].id();
            chunk.decimal_scale  = _metadata.per_file_metadata[stripe_source_mapping.source_idx]
                                    .ff.types[columns_level[col_idx].id]
                                    .scale.value_or(0);

            chunk.rowgroup_id   = rowgroup_id;
            chunk.dtype_len     = (column_types[col_idx].id() == type_id::STRING)
                                    ? sizeof(string_index_pair)
                                  : ((column_types[col_idx].id() == type_id::LIST) or
                                 (column_types[col_idx].id() == type_id::STRUCT))
                                    ? sizeof(size_type)
                                    : cudf::size_of(column_types[col_idx]);
            chunk.num_rowgroups = stripe_num_rowgroups;
            if (chunk.type_kind == orc::TIMESTAMP) {
              chunk.timestamp_type_id = _timestamp_type.id();
            }
            if (not is_data_empty) {
              for (int k = 0; k < gpu::CI_NUM_STREAMS; k++) {
                chunk.streams[k] = dst_base + stream_info[chunk.strm_id[k]].dst_pos;
              }
            }
          }
          stripe_start_row += num_rows_per_stripe;
          num_rowgroups += stripe_num_rowgroups;

          stripe_idx++;
        }
      }
      for (auto& task : read_tasks) {
        CUDF_EXPECTS(task.first.get() == task.second, "Unexpected discrepancy in bytes read.");
      }

      // Process dataset chunk pages into output columns
      if (stripe_data.size() != 0) {
        auto row_groups =
          cudf::detail::hostdevice_2dvector<gpu::RowGroup>(num_rowgroups, num_columns, stream);
        if (level > 0 and row_groups.size().first) {
          cudf::host_span<gpu::RowGroup> row_groups_span(row_groups.base_host_ptr(),
                                                         num_rowgroups * num_columns);
          auto& rw_grp_meta = _col_meta.rwgrp_meta;

          // Update start row and num rows per row group
          std::transform(rw_grp_meta.begin(),
                         rw_grp_meta.end(),
                         row_groups_span.begin(),
                         rw_grp_meta.begin(),
                         [&](auto meta, auto& row_grp) {
                           row_grp.num_rows  = meta.num_rows;
                           row_grp.start_row = meta.start_row;
                           return meta;
                         });
        }
        // Setup row group descriptors if using indexes
        if (_metadata.per_file_metadata[0].ps.compression != orc::NONE and not is_data_empty) {
          auto decomp_data =
            decompress_stripe_data(chunks,
                                   stripe_data,
                                   _metadata.per_file_metadata[0].decompressor.get(),
                                   stream_info,
                                   total_num_stripes,
                                   row_groups,
                                   _metadata.get_row_index_stride(),
                                   level == 0,
                                   stream);
          stripe_data.clear();
          stripe_data.push_back(std::move(decomp_data));
        } else {
          if (row_groups.size().first) {
            chunks.host_to_device(stream);
            row_groups.host_to_device(stream);
            gpu::ParseRowGroupIndex(row_groups.base_device_ptr(),
                                    nullptr,
                                    chunks.base_device_ptr(),
                                    num_columns,
                                    total_num_stripes,
                                    num_rowgroups,
                                    _metadata.get_row_index_stride(),
                                    level == 0,
                                    stream);
          }
        }

        for (size_t i = 0; i < column_types.size(); ++i) {
          bool is_nullable = false;
          for (size_t j = 0; j < total_num_stripes; ++j) {
            if (chunks[j][i].strm_len[gpu::CI_PRESENT] != 0) {
              is_nullable = true;
              break;
            }
          }
          auto is_list_type = (column_types[i].id() == type_id::LIST);
          auto n_rows       = (level == 0) ? num_rows : _col_meta.num_child_rows[i];
          // For list column, offset column will be always size + 1
          if (is_list_type) n_rows++;
          out_buffers[level].emplace_back(column_types[i], n_rows, is_nullable, stream, _mr);
        }

        if (not is_data_empty) {
          decode_stream_data(chunks,
                             num_dict_entries,
                             skip_rows,
                             tz_table.view(),
                             row_groups,
                             _metadata.get_row_index_stride(),
                             out_buffers[level],
                             level,
                             stream);
        }

        // Extract information to process nested child columns
        if (nested_col.size()) {
          if (not is_data_empty) {
            scan_null_counts(chunks, null_count_prefix_sums[level], stream);
          }
          row_groups.device_to_host(stream, true);
          aggregate_child_meta(chunks, row_groups, out_buffers[level], nested_col, level);
        }

        // ORC stores number of elements at each row, so we need to generate offsets from that
        if (nested_col.size()) {
          std::vector<list_buffer_data> buff_data;
          std::for_each(
            out_buffers[level].begin(), out_buffers[level].end(), [&buff_data](auto& out_buffer) {
              if (out_buffer.type.id() == type_id::LIST) {
                auto data = static_cast<size_type*>(out_buffer.data());
                buff_data.emplace_back(list_buffer_data{data, out_buffer.size});
              }
            });

          if (buff_data.size()) {
            auto const dev_buff_data = cudf::detail::make_device_uvector_async(buff_data, stream);
            generate_offsets_for_list(dev_buff_data, stream);
          }
        }
      }
    }
  }

  // If out_columns is empty, then create columns from buffer.
  if (out_columns.empty()) {
    create_columns(std::move(out_buffers), out_columns, schema_info, stream);
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.reserve(schema_info.size());
  std::transform(schema_info.cbegin(),
                 schema_info.cend(),
                 std::back_inserter(out_metadata.column_names),
                 [](auto info) { return info.name; });

  out_metadata.schema_info = std::move(schema_info);

  std::transform(_metadata.per_file_metadata.cbegin(),
                 _metadata.per_file_metadata.cend(),
                 std::back_inserter(out_metadata.per_file_user_data),
                 [](auto& meta) {
                   std::unordered_map<std::string, std::string> kv_map;
                   std::transform(meta.ff.metadata.cbegin(),
                                  meta.ff.metadata.cend(),
                                  std::inserter(kv_map, kv_map.end()),
                                  [](auto const& kv) {
                                    return std::pair{kv.name, kv.value};
                                  });
                   return kv_map;
                 });
  out_metadata.user_data = {out_metadata.per_file_user_data[0].begin(),
                            out_metadata.per_file_user_data[0].end()};

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
               orc_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
{
  _impl = std::make_unique<impl>(std::move(sources), options, mr);
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(orc_reader_options const& options, rmm::cuda_stream_view stream)
{
  return _impl->read(
    options.get_skip_rows(), options.get_num_rows(), options.get_stripes(), stream);
}

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
