/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "io/comp/gpuinflate.hpp"
#include "io/comp/nvcomp_adapter.hpp"
#include "io/utilities/config_utils.hpp"
#include "reader_impl.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_helpers.hpp"

#include <cudf/detail/timezone.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/logger.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iterator>

namespace cudf::io::orc::detail {

namespace {

/**
 * @brief Struct that maps ORC streams to columns
 */
struct orc_stream_info {
  explicit orc_stream_info(uint64_t offset_,
                           std::size_t dst_pos_,
                           uint32_t length_,
                           uint32_t stripe_idx_)
    : offset(offset_), dst_pos(dst_pos_), length(length_), stripe_idx(stripe_idx_)
  {
  }
  uint64_t offset;      // offset in file
  std::size_t dst_pos;  // offset in memory relative to start of compressed stripe data
  std::size_t length;   // length in file
  uint32_t stripe_idx;  // stripe index
};

/**
 * @brief Function that populates column descriptors stream/chunk
 */
std::size_t gather_stream_info(std::size_t stripe_index,
                               orc::StripeInformation const* stripeinfo,
                               orc::StripeFooter const* stripefooter,
                               host_span<int const> orc2gdf,
                               host_span<orc::SchemaType const> types,
                               bool use_index,
                               bool apply_struct_map,
                               int64_t* num_dictionary_entries,
                               std::vector<orc_stream_info>& stream_info,
                               cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks)
{
  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;

  auto const get_stream_index_type = [](orc::StreamKind kind) {
    switch (kind) {
      case orc::DATA: return gpu::CI_DATA;
      case orc::LENGTH:
      case orc::SECONDARY: return gpu::CI_DATA2;
      case orc::DICTIONARY_DATA: return gpu::CI_DICTIONARY;
      case orc::PRESENT: return gpu::CI_PRESENT;
      case orc::ROW_INDEX: return gpu::CI_INDEX;
      default:
        // Skip this stream as it's not strictly required
        return gpu::CI_NUM_STREAMS;
    }
  };

  for (auto const& stream : stripefooter->streams) {
    if (!stream.column_id || *stream.column_id >= orc2gdf.size()) {
      // Ignore reading this stream from source.
      cudf::logger().warn("Unexpected stream in the input ORC source. The stream will be ignored.");
      src_offset += stream.length;
      continue;
    }

    auto const column_id = *stream.column_id;
    auto col             = orc2gdf[column_id];

    if (col == -1 and apply_struct_map) {
      // A struct-type column has no data itself, but rather child columns
      // for each of its fields. There is only a PRESENT stream, which
      // needs to be included for the reader.
      auto const schema_type = types[column_id];
      if (not schema_type.subtypes.empty()) {
        if (schema_type.kind == orc::STRUCT && stream.kind == orc::PRESENT) {
          for (auto const& idx : schema_type.subtypes) {
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
    } else if (col != -1) {
      if (src_offset >= stripeinfo->indexLength || use_index) {
        auto& chunk           = chunks[stripe_index][col];
        auto const index_type = get_stream_index_type(stream.kind);
        if (index_type < gpu::CI_NUM_STREAMS) {
          chunk.strm_id[index_type]  = stream_info.size();
          chunk.strm_len[index_type] = stream.length;
          // NOTE: skip_count field is temporarily used to track the presence of index streams
          chunk.skip_count |= 1 << index_type;

          if (index_type == gpu::CI_DICTIONARY) {
            chunk.dictionary_start = *num_dictionary_entries;
            chunk.dict_len         = stripefooter->columns[column_id].dictionarySize;
            *num_dictionary_entries += stripefooter->columns[column_id].dictionarySize;
          }
        }
      }
      stream_info.emplace_back(
        stripeinfo->offset + src_offset, dst_offset, stream.length, stripe_index);
      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

/**
 * @brief Decompresses the stripe data, at stream granularity.
 *
 * @param decompressor Block decompressor
 * @param stripe_data List of source stripe column data
 * @param stream_info List of stream to column mappings
 * @param chunks Vector of list of column chunk descriptors
 * @param row_groups Vector of list of row index descriptors
 * @param num_stripes Number of stripes making up column chunks
 * @param row_index_stride Distance between each row index
 * @param use_base_stride Whether to use base stride obtained from meta or use the computed value
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Device buffer to decompressed page data
 */
rmm::device_buffer decompress_stripe_data(
  OrcDecompressor const& decompressor,
  host_span<rmm::device_buffer const> stripe_data,
  host_span<orc_stream_info> stream_info,
  cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
  cudf::detail::hostdevice_2dvector<gpu::RowGroup>& row_groups,
  size_type num_stripes,
  size_type row_index_stride,
  bool use_base_stride,
  rmm::cuda_stream_view stream)
{
  // Parse the columns' compressed info
  cudf::detail::hostdevice_vector<gpu::CompressedStreamInfo> compinfo(
    0, stream_info.size(), stream);
  for (auto const& info : stream_info) {
    compinfo.push_back(gpu::CompressedStreamInfo(
      static_cast<uint8_t const*>(stripe_data[info.stripe_idx].data()) + info.dst_pos,
      info.length));
  }
  compinfo.host_to_device_async(stream);

  gpu::ParseCompressedStripeData(compinfo.device_ptr(),
                                 compinfo.size(),
                                 decompressor.GetBlockSize(),
                                 decompressor.GetLog2MaxCompressionRatio(),
                                 stream);
  compinfo.device_to_host_sync(stream);

  // Count the exact number of compressed blocks
  std::size_t num_compressed_blocks   = 0;
  std::size_t num_uncompressed_blocks = 0;
  std::size_t total_decomp_size       = 0;
  for (std::size_t i = 0; i < compinfo.size(); ++i) {
    num_compressed_blocks += compinfo[i].num_compressed_blocks;
    num_uncompressed_blocks += compinfo[i].num_uncompressed_blocks;
    total_decomp_size += compinfo[i].max_uncompressed_size;
  }
  CUDF_EXPECTS(
    not((num_uncompressed_blocks + num_compressed_blocks > 0) and (total_decomp_size == 0)),
    "Inconsistent info on compression blocks");

  // Buffer needs to be padded.
  // Required by `gpuDecodeOrcColumnData`.
  rmm::device_buffer decomp_data(
    cudf::util::round_up_safe(total_decomp_size, BUFFER_PADDING_MULTIPLE), stream);
  if (decomp_data.is_empty()) { return decomp_data; }

  rmm::device_uvector<device_span<uint8_t const>> inflate_in(
    num_compressed_blocks + num_uncompressed_blocks, stream);
  rmm::device_uvector<device_span<uint8_t>> inflate_out(
    num_compressed_blocks + num_uncompressed_blocks, stream);
  rmm::device_uvector<compression_result> inflate_res(num_compressed_blocks, stream);
  thrust::fill(rmm::exec_policy(stream),
               inflate_res.begin(),
               inflate_res.end(),
               compression_result{0, compression_status::FAILURE});

  // Parse again to populate the decompression input/output buffers
  std::size_t decomp_offset      = 0;
  uint32_t max_uncomp_block_size = 0;
  uint32_t start_pos             = 0;
  auto start_pos_uncomp          = (uint32_t)num_compressed_blocks;
  for (std::size_t i = 0; i < compinfo.size(); ++i) {
    auto dst_base                 = static_cast<uint8_t*>(decomp_data.data());
    compinfo[i].uncompressed_data = dst_base + decomp_offset;
    compinfo[i].dec_in_ctl        = inflate_in.data() + start_pos;
    compinfo[i].dec_out_ctl       = inflate_out.data() + start_pos;
    compinfo[i].dec_res      = {inflate_res.data() + start_pos, compinfo[i].num_compressed_blocks};
    compinfo[i].copy_in_ctl  = inflate_in.data() + start_pos_uncomp;
    compinfo[i].copy_out_ctl = inflate_out.data() + start_pos_uncomp;

    stream_info[i].dst_pos = decomp_offset;
    decomp_offset += compinfo[i].max_uncompressed_size;
    start_pos += compinfo[i].num_compressed_blocks;
    start_pos_uncomp += compinfo[i].num_uncompressed_blocks;
    max_uncomp_block_size =
      std::max(max_uncomp_block_size, compinfo[i].max_uncompressed_block_size);
  }
  compinfo.host_to_device_async(stream);
  gpu::ParseCompressedStripeData(compinfo.device_ptr(),
                                 compinfo.size(),
                                 decompressor.GetBlockSize(),
                                 decompressor.GetLog2MaxCompressionRatio(),
                                 stream);

  // Value for checking whether we decompress successfully.
  // It doesn't need to be atomic as there is no race condition: we only write `true` if needed.
  cudf::detail::hostdevice_vector<bool> any_block_failure(1, stream);
  any_block_failure[0] = false;
  any_block_failure.host_to_device_async(stream);

  // Dispatch batches of blocks to decompress
  if (num_compressed_blocks > 0) {
    device_span<device_span<uint8_t const>> inflate_in_view{inflate_in.data(),
                                                            num_compressed_blocks};
    device_span<device_span<uint8_t>> inflate_out_view{inflate_out.data(), num_compressed_blocks};
    switch (decompressor.compression()) {
      case compression_type::ZLIB:
        if (nvcomp::is_decompression_disabled(nvcomp::compression_type::DEFLATE)) {
          gpuinflate(
            inflate_in_view, inflate_out_view, inflate_res, gzip_header_included::NO, stream);
        } else {
          nvcomp::batched_decompress(nvcomp::compression_type::DEFLATE,
                                     inflate_in_view,
                                     inflate_out_view,
                                     inflate_res,
                                     max_uncomp_block_size,
                                     total_decomp_size,
                                     stream);
        }
        break;
      case compression_type::SNAPPY:
        if (nvcomp::is_decompression_disabled(nvcomp::compression_type::SNAPPY)) {
          gpu_unsnap(inflate_in_view, inflate_out_view, inflate_res, stream);
        } else {
          nvcomp::batched_decompress(nvcomp::compression_type::SNAPPY,
                                     inflate_in_view,
                                     inflate_out_view,
                                     inflate_res,
                                     max_uncomp_block_size,
                                     total_decomp_size,
                                     stream);
        }
        break;
      case compression_type::ZSTD:
        if (auto const reason = nvcomp::is_decompression_disabled(nvcomp::compression_type::ZSTD);
            reason) {
          CUDF_FAIL("Decompression error: " + reason.value());
        }
        nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                                   inflate_in_view,
                                   inflate_out_view,
                                   inflate_res,
                                   max_uncomp_block_size,
                                   total_decomp_size,
                                   stream);
        break;
      case compression_type::LZ4:
        if (auto const reason = nvcomp::is_decompression_disabled(nvcomp::compression_type::LZ4);
            reason) {
          CUDF_FAIL("Decompression error: " + reason.value());
        }
        nvcomp::batched_decompress(nvcomp::compression_type::LZ4,
                                   inflate_in_view,
                                   inflate_out_view,
                                   inflate_res,
                                   max_uncomp_block_size,
                                   total_decomp_size,
                                   stream);
        break;
      default: CUDF_FAIL("Unexpected decompression dispatch"); break;
    }

    // Check if any block has been failed to decompress.
    // Not using `thrust::any` or `thrust::count_if` to defer stream sync.
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(std::size_t{0}),
      thrust::make_counting_iterator(inflate_res.size()),
      [results           = inflate_res.begin(),
       any_block_failure = any_block_failure.device_ptr()] __device__(auto const idx) {
        if (results[idx].status != compression_status::SUCCESS) { *any_block_failure = true; }
      });
  }

  if (num_uncompressed_blocks > 0) {
    device_span<device_span<uint8_t const>> copy_in_view{inflate_in.data() + num_compressed_blocks,
                                                         num_uncompressed_blocks};
    device_span<device_span<uint8_t>> copy_out_view{inflate_out.data() + num_compressed_blocks,
                                                    num_uncompressed_blocks};
    gpu_copy_uncompressed_blocks(copy_in_view, copy_out_view, stream);
  }

  // Copy without stream sync, thus need to wait for stream sync below to access.
  any_block_failure.device_to_host_async(stream);

  gpu::PostDecompressionReassemble(compinfo.device_ptr(), compinfo.size(), stream);
  compinfo.device_to_host_sync(stream);  // This also sync stream for `any_block_failure`.

  // We can check on host after stream synchronize
  CUDF_EXPECTS(not any_block_failure[0], "Error during decompression");

  size_type const num_columns = chunks.size().second;

  // Update the stream information with the updated uncompressed info
  // TBD: We could update the value from the information we already
  // have in stream_info[], but using the gpu results also updates
  // max_uncompressed_size to the actual uncompressed size, or zero if
  // decompression failed.
  for (size_type i = 0; i < num_stripes; ++i) {
    for (size_type j = 0; j < num_columns; ++j) {
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
    chunks.host_to_device_async(stream);
    row_groups.host_to_device_async(stream);
    gpu::ParseRowGroupIndex(row_groups.base_device_ptr(),
                            compinfo.device_ptr(),
                            chunks.base_device_ptr(),
                            num_columns,
                            num_stripes,
                            row_index_stride,
                            use_base_stride,
                            stream);
  }

  return decomp_data;
}

/**
 * @brief Updates null mask of columns whose parent is a struct column.
 *
 * If struct column has null element, that row would be skipped while writing child column in ORC,
 * so we need to insert the missing null elements in child column. There is another behavior from
 * pyspark, where if the child column doesn't have any null elements, it will not have present
 * stream, so in that case parent null mask need to be copied to child column.
 *
 * @param chunks Vector of list of column chunk descriptors
 * @param out_buffers Output columns' device buffers
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource to use for device memory allocation
 */
void update_null_mask(cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                      host_span<column_buffer> out_buffers,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  auto const num_stripes = chunks.size().first;
  auto const num_columns = chunks.size().second;
  bool is_mask_updated   = false;

  for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    if (chunks[0][col_idx].parent_validity_info.valid_map_base != nullptr) {
      if (not is_mask_updated) {
        chunks.device_to_host_sync(stream);
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
        // Copy child valid bits from child column to valid indexes, this will merge both child
        // and parent null masks
        thrust::for_each(rmm::exec_policy(stream),
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(0) + dst_idx.size(),
                         [child_valid_map_base, dst_idx_ptr, merged_mask] __device__(auto idx) {
                           if (bit_is_set(child_valid_map_base, idx)) {
                             cudf::set_bit(merged_mask, dst_idx_ptr[idx]);
                           };
                         });

        out_buffers[col_idx].set_null_mask(std::move(merged_null_mask));

      } else {
        // Since child column doesn't have a mask, copy parent null mask
        auto mask_size = bitmask_allocation_size_bytes(parent_mask_len);
        out_buffers[col_idx].set_null_mask(
          rmm::device_buffer(static_cast<void*>(parent_valid_map_base), mask_size, stream, mr));
      }
    }
  }

  if (is_mask_updated) {
    // Update chunks with pointers to column data which might have been changed.
    for (std::size_t stripe_idx = 0; stripe_idx < num_stripes; ++stripe_idx) {
      for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        auto& chunk          = chunks[stripe_idx][col_idx];
        chunk.valid_map_base = out_buffers[col_idx].null_mask();
      }
    }
    chunks.host_to_device_sync(stream);
  }
}

/**
 * @brief Converts the stripe column data and outputs to columns.
 *
 * @param num_dicts Number of dictionary entries required
 * @param skip_rows Number of rows to offset from start
 * @param row_index_stride Distance between each row index
 * @param level Current nesting level being processed
 * @param tz_table Local time to UTC conversion table
 * @param chunks Vector of list of column chunk descriptors
 * @param row_groups Vector of list of row index descriptors
 * @param out_buffers Output columns' device buffers
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 */
void decode_stream_data(std::size_t num_dicts,
                        int64_t skip_rows,
                        size_type row_index_stride,
                        std::size_t level,
                        table_view const& tz_table,
                        cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                        cudf::detail::device_2dspan<gpu::RowGroup> row_groups,
                        std::vector<column_buffer>& out_buffers,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  auto const num_stripes = chunks.size().first;
  auto const num_columns = chunks.size().second;
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

  chunks.host_to_device_sync(stream);
  gpu::DecodeNullsAndStringDictionaries(
    chunks.base_device_ptr(), global_dict.data(), num_columns, num_stripes, skip_rows, stream);

  if (level > 0) {
    // Update nullmasks for children if parent was a struct and had null mask
    update_null_mask(chunks, out_buffers, stream, mr);
  }

  auto const tz_table_dptr = table_device_view::create(tz_table, stream);
  rmm::device_scalar<size_type> error_count(0, stream);
  // Update the null map for child columns
  gpu::DecodeOrcColumnData(chunks.base_device_ptr(),
                           global_dict.data(),
                           row_groups,
                           num_columns,
                           num_stripes,
                           skip_rows,
                           *tz_table_dptr,
                           row_groups.size().first,
                           row_index_stride,
                           level,
                           error_count.data(),
                           stream);
  chunks.device_to_host_async(stream);
  // `value` synchronizes
  auto const num_errors = error_count.value(stream);
  CUDF_EXPECTS(num_errors == 0, "ORC data decode failed");

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
  auto const d_prefix_sums_to_update = cudf::detail::make_device_uvector_async(
    prefix_sums_to_update, stream, rmm::mr::get_current_device_resource());

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

/**
 * @brief Aggregate child metadata from parent column chunks.
 */
void aggregate_child_meta(std::size_t level,
                          cudf::io::orc::detail::column_hierarchy const& selected_columns,
                          cudf::detail::host_2dspan<gpu::ColumnDesc> chunks,
                          cudf::detail::host_2dspan<gpu::RowGroup> row_groups,
                          host_span<orc_column_meta const> nested_cols,
                          host_span<column_buffer> out_buffers,
                          reader_column_meta& col_meta)
{
  auto const num_of_stripes         = chunks.size().first;
  auto const num_of_rowgroups       = row_groups.size().first;
  auto const num_child_cols         = selected_columns.levels[level + 1].size();
  auto const number_of_child_chunks = num_child_cols * num_of_stripes;
  auto& num_child_rows              = col_meta.num_child_rows;
  auto& parent_column_data          = col_meta.parent_column_data;

  // Reset the meta to store child column details.
  num_child_rows.resize(selected_columns.levels[level + 1].size());
  std::fill(num_child_rows.begin(), num_child_rows.end(), 0);
  parent_column_data.resize(number_of_child_chunks);
  col_meta.parent_column_index.resize(number_of_child_chunks);
  col_meta.child_start_row.resize(number_of_child_chunks);
  col_meta.num_child_rows_per_stripe.resize(number_of_child_chunks);
  col_meta.rwgrp_meta.resize(num_of_rowgroups * num_child_cols);

  auto child_start_row = cudf::detail::host_2dspan<int64_t>(
    col_meta.child_start_row.data(), num_of_stripes, num_child_cols);
  auto num_child_rows_per_stripe = cudf::detail::host_2dspan<int64_t>(
    col_meta.num_child_rows_per_stripe.data(), num_of_stripes, num_child_cols);
  auto rwgrp_meta = cudf::detail::host_2dspan<reader_column_meta::row_group_meta>(
    col_meta.rwgrp_meta.data(), num_of_rowgroups, num_child_cols);

  int index = 0;  // number of child column processed

  // For each parent column, update its child column meta for each stripe.
  std::for_each(nested_cols.begin(), nested_cols.end(), [&](auto const p_col) {
    auto const parent_col_idx = col_meta.orc_col_map[level][p_col.id];
    int64_t start_row         = 0;
    auto processed_row_groups = 0;

    for (std::size_t stripe_id = 0; stripe_id < num_of_stripes; stripe_id++) {
      // Aggregate num_rows and start_row from processed parent columns per row groups
      if (num_of_rowgroups) {
        auto stripe_num_row_groups = chunks[stripe_id][parent_col_idx].num_rowgroups;
        auto processed_child_rows  = 0;

        for (std::size_t rowgroup_id = 0; rowgroup_id < stripe_num_row_groups;
             rowgroup_id++, processed_row_groups++) {
          auto const child_rows = row_groups[processed_row_groups][parent_col_idx].num_child_rows;
          for (size_type id = 0; id < p_col.num_children; id++) {
            auto const child_col_idx                                  = index + id;
            rwgrp_meta[processed_row_groups][child_col_idx].start_row = processed_child_rows;
            rwgrp_meta[processed_row_groups][child_col_idx].num_rows  = child_rows;
          }
          processed_child_rows += child_rows;
        }
      }

      // Aggregate start row, number of rows per chunk and total number of rows in a column
      auto const child_rows = chunks[stripe_id][parent_col_idx].num_child_rows;
      for (size_type id = 0; id < p_col.num_children; id++) {
        auto const child_col_idx = index + id;

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
      auto const child_col_idx                    = index + id;
      col_meta.parent_column_index[child_col_idx] = parent_col_idx;
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

/**
 * @brief struct to store buffer data and size of list buffer
 */
struct list_buffer_data {
  size_type* data;
  size_type size;
};

// Generates offsets for list buffer from number of elements in a row.
void generate_offsets_for_list(host_span<list_buffer_data> buff_data, rmm::cuda_stream_view stream)
{
  for (auto& list_data : buff_data) {
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                           list_data.data,
                           list_data.data + list_data.size,
                           list_data.data);
  }
}

}  // namespace

void reader::impl::prepare_data(int64_t skip_rows,
                                std::optional<size_type> const& num_rows_opt,
                                std::vector<std::vector<size_type>> const& stripes)
{
  // Selected columns at different levels of nesting are stored in different elements
  // of `selected_columns`; thus, size == 1 means no nested columns
  CUDF_EXPECTS(skip_rows == 0 or _selected_columns.num_levels() == 1,
               "skip_rows is not supported by nested columns");

  // There are no columns in the table
  if (_selected_columns.num_levels() == 0) { return; }

  _file_itm_data = std::make_unique<file_intermediate_data>();

  // Select only stripes required (aka row groups)
  std::tie(
    _file_itm_data->rows_to_skip, _file_itm_data->rows_to_read, _file_itm_data->selected_stripes) =
    _metadata.select_stripes(stripes, skip_rows, num_rows_opt, _stream);
  auto const rows_to_skip      = _file_itm_data->rows_to_skip;
  auto const rows_to_read      = _file_itm_data->rows_to_read;
  auto const& selected_stripes = _file_itm_data->selected_stripes;

  // If no rows or stripes to read, return empty columns
  if (rows_to_read == 0 || selected_stripes.empty()) { return; }

  // Set up table for converting timestamp columns from local to UTC time
  auto const tz_table = [&, &selected_stripes = selected_stripes] {
    auto const has_timestamp_column = std::any_of(
      _selected_columns.levels.cbegin(), _selected_columns.levels.cend(), [&](auto const& col_lvl) {
        return std::any_of(col_lvl.cbegin(), col_lvl.cend(), [&](auto const& col_meta) {
          return _metadata.get_col_type(col_meta.id).kind == TypeKind::TIMESTAMP;
        });
      });

    return has_timestamp_column
             ? cudf::detail::make_timezone_transition_table(
                 {}, selected_stripes[0].stripe_info[0].second->writerTimezone, _stream)
             : std::make_unique<cudf::table>();
  }();

  auto& lvl_stripe_data        = _file_itm_data->lvl_stripe_data;
  auto& null_count_prefix_sums = _file_itm_data->null_count_prefix_sums;
  lvl_stripe_data.resize(_selected_columns.num_levels());

  _out_buffers.resize(_selected_columns.num_levels());

  // Iterates through levels of nested columns, child column will be one level down
  // compared to parent column.
  auto& col_meta = *_col_meta;
  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    auto& columns_level = _selected_columns.levels[level];
    // Association between each ORC column and its cudf::column
    col_meta.orc_col_map.emplace_back(_metadata.get_num_cols(), -1);
    std::vector<orc_column_meta> nested_cols;

    // Get a list of column data types
    std::vector<data_type> column_types;
    for (auto& col : columns_level) {
      auto col_type = to_cudf_type(_metadata.get_col_type(col.id).kind,
                                   _use_np_dtypes,
                                   _timestamp_type.id(),
                                   to_cudf_decimal_type(_decimal128_columns, _metadata, col.id));
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
      col_meta.orc_col_map[level][col.id] = column_types.size() - 1;
      if (col_type == type_id::LIST or col_type == type_id::STRUCT) {
        nested_cols.emplace_back(col);
      }
    }

    // Get the total number of stripes across all input files.
    std::size_t total_num_stripes =
      std::accumulate(selected_stripes.begin(),
                      selected_stripes.end(),
                      0,
                      [](std::size_t sum, auto& stripe_source_mapping) {
                        return sum + stripe_source_mapping.stripe_info.size();
                      });
    auto const num_columns = columns_level.size();
    cudf::detail::hostdevice_2dvector<gpu::ColumnDesc> chunks(
      total_num_stripes, num_columns, _stream);
    memset(chunks.base_host_ptr(), 0, chunks.size_bytes());

    const bool use_index =
      _use_index &&
      // Do stripes have row group index
      _metadata.is_row_grp_idx_present() &&
      // Only use if we don't have much work with complete columns & stripes
      // TODO: Consider nrows, gpu, and tune the threshold
      (rows_to_read > _metadata.get_row_index_stride() && !(_metadata.get_row_index_stride() & 7) &&
       _metadata.get_row_index_stride() != 0 && num_columns * total_num_stripes < 8 * 128) &&
      // Only use if first row is aligned to a stripe boundary
      // TODO: Fix logic to handle unaligned rows
      (rows_to_skip == 0);

    // Logically view streams as columns
    std::vector<orc_stream_info> stream_info;

    null_count_prefix_sums.emplace_back();
    null_count_prefix_sums.back().reserve(_selected_columns.levels[level].size());
    std::generate_n(std::back_inserter(null_count_prefix_sums.back()),
                    _selected_columns.levels[level].size(),
                    [&]() {
                      return cudf::detail::make_zeroed_device_uvector_async<uint32_t>(
                        total_num_stripes, _stream, rmm::mr::get_current_device_resource());
                    });

    // Tracker for eventually deallocating compressed and uncompressed data
    auto& stripe_data = lvl_stripe_data[level];

    int64_t stripe_start_row = 0;
    int64_t num_dict_entries = 0;
    int64_t num_rowgroups    = 0;
    size_type stripe_idx     = 0;

    std::vector<std::pair<std::future<std::size_t>, std::size_t>> read_tasks;
    for (auto const& stripe_source_mapping : selected_stripes) {
      // Iterate through the source files selected stripes
      for (auto const& stripe : stripe_source_mapping.stripe_info) {
        auto const stripe_info   = stripe.first;
        auto const stripe_footer = stripe.second;

        auto stream_count          = stream_info.size();
        auto const total_data_size = gather_stream_info(stripe_idx,
                                                        stripe_info,
                                                        stripe_footer,
                                                        col_meta.orc_col_map[level],
                                                        _metadata.get_types(),
                                                        use_index,
                                                        level == 0,
                                                        &num_dict_entries,
                                                        stream_info,
                                                        chunks);

        auto const is_stripe_data_empty = total_data_size == 0;
        CUDF_EXPECTS(not is_stripe_data_empty or stripe_info->indexLength == 0,
                     "Invalid index rowgroup stream data");

        // Buffer needs to be padded.
        // Required by `copy_uncompressed_kernel`.
        stripe_data.emplace_back(
          cudf::util::round_up_safe(total_data_size, BUFFER_PADDING_MULTIPLE), _stream);
        auto dst_base = static_cast<uint8_t*>(stripe_data.back().data());

        // Coalesce consecutive streams into one read
        while (not is_stripe_data_empty and stream_count < stream_info.size()) {
          auto const d_dst  = dst_base + stream_info[stream_count].dst_pos;
          auto const offset = stream_info[stream_count].offset;
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
                          .source->device_read_async(offset, len, d_dst, _stream),
                        len));

          } else {
            auto const buffer =
              _metadata.per_file_metadata[stripe_source_mapping.source_idx].source->host_read(
                offset, len);
            CUDF_EXPECTS(buffer->size() == len, "Unexpected discrepancy in bytes read.");
            CUDF_CUDA_TRY(
              cudaMemcpyAsync(d_dst, buffer->data(), len, cudaMemcpyDefault, _stream.value()));
            _stream.synchronize();
          }
        }

        auto const num_rows_per_stripe = stripe_info->numberOfRows;
        auto const rowgroup_id         = num_rowgroups;
        auto stripe_num_rowgroups      = 0;
        if (use_index) {
          stripe_num_rowgroups = (num_rows_per_stripe + _metadata.get_row_index_stride() - 1) /
                                 _metadata.get_row_index_stride();
        }
        // Update chunks to reference streams pointers
        for (std::size_t col_idx = 0; col_idx < num_columns; col_idx++) {
          auto& chunk = chunks[stripe_idx][col_idx];
          // start row, number of rows in a each stripe and total number of rows
          // may change in lower levels of nesting
          chunk.start_row = (level == 0)
                              ? stripe_start_row
                              : col_meta.child_start_row[stripe_idx * num_columns + col_idx];
          chunk.num_rows =
            (level == 0) ? stripe_info->numberOfRows
                         : col_meta.num_child_rows_per_stripe[stripe_idx * num_columns + col_idx];
          chunk.column_num_rows = (level == 0) ? rows_to_read : col_meta.num_child_rows[col_idx];
          chunk.parent_validity_info =
            (level == 0) ? column_validity_info{} : col_meta.parent_column_data[col_idx];
          chunk.parent_null_count_prefix_sums =
            (level == 0)
              ? nullptr
              : null_count_prefix_sums[level - 1][col_meta.parent_column_index[col_idx]].data();
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
          if (chunk.type_kind == orc::TIMESTAMP) { chunk.timestamp_type_id = _timestamp_type.id(); }
          if (not is_stripe_data_empty) {
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

    if (stripe_data.empty()) { continue; }

    // Process dataset chunk pages into output columns
    auto row_groups =
      cudf::detail::hostdevice_2dvector<gpu::RowGroup>(num_rowgroups, num_columns, _stream);
    if (level > 0 and row_groups.size().first) {
      cudf::host_span<gpu::RowGroup> row_groups_span(row_groups.base_host_ptr(),
                                                     num_rowgroups * num_columns);
      auto& rw_grp_meta = col_meta.rwgrp_meta;

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
    if (_metadata.per_file_metadata[0].ps.compression != orc::NONE) {
      auto decomp_data = decompress_stripe_data(*_metadata.per_file_metadata[0].decompressor,
                                                stripe_data,
                                                stream_info,
                                                chunks,
                                                row_groups,
                                                total_num_stripes,
                                                _metadata.get_row_index_stride(),
                                                level == 0,
                                                _stream);
      stripe_data.clear();
      stripe_data.push_back(std::move(decomp_data));
    } else {
      if (row_groups.size().first) {
        chunks.host_to_device_async(_stream);
        row_groups.host_to_device_async(_stream);
        row_groups.host_to_device_async(_stream);
        gpu::ParseRowGroupIndex(row_groups.base_device_ptr(),
                                nullptr,
                                chunks.base_device_ptr(),
                                num_columns,
                                total_num_stripes,
                                _metadata.get_row_index_stride(),
                                level == 0,
                                _stream);
      }
    }

    for (std::size_t i = 0; i < column_types.size(); ++i) {
      bool is_nullable = false;
      for (std::size_t j = 0; j < total_num_stripes; ++j) {
        if (chunks[j][i].strm_len[gpu::CI_PRESENT] != 0) {
          is_nullable = true;
          break;
        }
      }
      auto is_list_type = (column_types[i].id() == type_id::LIST);
      auto n_rows       = (level == 0) ? rows_to_read : col_meta.num_child_rows[i];
      // For list column, offset column will be always size + 1
      if (is_list_type) n_rows++;
      _out_buffers[level].emplace_back(column_types[i], n_rows, is_nullable, _stream, _mr);
    }

    decode_stream_data(num_dict_entries,
                       rows_to_skip,
                       _metadata.get_row_index_stride(),
                       level,
                       tz_table->view(),
                       chunks,
                       row_groups,
                       _out_buffers[level],
                       _stream,
                       _mr);

    if (nested_cols.size()) {
      // Extract information to process nested child columns
      scan_null_counts(chunks, null_count_prefix_sums[level], _stream);

      row_groups.device_to_host_sync(_stream);
      aggregate_child_meta(
        level, _selected_columns, chunks, row_groups, nested_cols, _out_buffers[level], col_meta);

      // ORC stores number of elements at each row, so we need to generate offsets from that
      std::vector<list_buffer_data> buff_data;
      std::for_each(
        _out_buffers[level].begin(), _out_buffers[level].end(), [&buff_data](auto& out_buffer) {
          if (out_buffer.type.id() == type_id::LIST) {
            auto data = static_cast<size_type*>(out_buffer.data());
            buff_data.emplace_back(list_buffer_data{data, out_buffer.size});
          }
        });

      if (not buff_data.empty()) { generate_offsets_for_list(buff_data, _stream); }
    }
  }  // end loop level
}

}  // namespace cudf::io::orc::detail
