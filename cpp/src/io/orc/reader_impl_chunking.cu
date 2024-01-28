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

// #define PRINT_DEBUG

#include "reader_impl.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_helpers.hpp"

#include <io/comp/gpuinflate.hpp>
#include <io/comp/nvcomp_adapter.hpp>
#include <io/utilities/config_utils.hpp>

#include <cudf/detail/timezone.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

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
 * @brief Function that populates column descriptors stream/chunk
 */
std::size_t gather_stream_info_and_update_chunks(
  std::size_t stripe_index,
  std::size_t level,
  orc::StripeInformation const* stripeinfo,
  orc::StripeFooter const* stripefooter,
  host_span<int const> orc2gdf,
  host_span<orc::SchemaType const> types,
  bool use_index,
  bool apply_struct_map,
  std::size_t* num_dictionary_entries,
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
      dst_offset += stream.length;
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
    }
    if (col != -1) {
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

      stream_info.emplace_back(stripeinfo->offset + src_offset,
                               dst_offset,
                               stream.length,
                               stripe_index,
                               level,
                               column_id,
                               stream.kind);
      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

}  // namespace

void reader::impl::query_stripe_compression_info()
{
  if (_file_itm_data->compinfo_ready) { return; }
  if (_selected_columns.num_levels() == 0) { return; }

  auto const rows_to_skip      = _file_itm_data->rows_to_skip;
  auto const rows_to_read      = _file_itm_data->rows_to_read;
  auto const& selected_stripes = _file_itm_data->selected_stripes;

  // If no rows or stripes to read, return empty columns
  // TODO : remove?
  if (rows_to_read == 0 || selected_stripes.empty()) { return; }

  auto& lvl_stripe_data      = _file_itm_data->lvl_stripe_data;
  auto& lvl_data_chunks      = _file_itm_data->lvl_data_chunks;
  auto& lvl_num_dict_entries = _file_itm_data->lvl_num_dict_entries;
  auto& lvl_stripe_is_empty  = _file_itm_data->lvl_stripe_is_empty;

  lvl_stripe_data.resize(_selected_columns.num_levels());
  lvl_data_chunks.resize(_selected_columns.num_levels());
  lvl_num_dict_entries.resize(_selected_columns.num_levels());
  lvl_stripe_is_empty.resize(_selected_columns.num_levels());

  // TODO: Don't have to keep it for all stripe/level. Can reset it after each iter.
  std::unordered_map<stream_id_info, gpu::CompressedStreamInfo*, stream_id_hash, stream_id_equal>
    stream_compinfo_map;

  // Logically view streams as columns
  _file_itm_data->lvl_stream_info.resize(_selected_columns.num_levels());

  // Iterates through levels of nested columns, child column will be one level down
  // compared to parent column.
  auto& col_meta = *_col_meta;
  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    auto& columns_level = _selected_columns.levels[level];
    // Association between each ORC column and its cudf::column
    col_meta.orc_col_map.emplace_back(_metadata.get_num_cols(), -1);

    size_type col_id{0};
    for (auto& col : columns_level) {
      // Map each ORC column to its column
      col_meta.orc_col_map[level][col.id] = col_id++;
    }
  }

  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    // Get the total number of stripes across all input files.
    std::size_t total_num_stripes =
      std::accumulate(selected_stripes.begin(),
                      selected_stripes.end(),
                      0,
                      [](std::size_t sum, auto& stripe_source_mapping) {
                        return sum + stripe_source_mapping.stripe_info.size();
                      });
    auto& columns_level    = _selected_columns.levels[level];
    auto const num_columns = columns_level.size();
    _file_itm_data->lvl_data_chunks[level] =
      cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>(total_num_stripes, num_columns, _stream);
    auto& chunks = _file_itm_data->lvl_data_chunks[level];
    memset(chunks.base_host_ptr(), 0, chunks.size_bytes());

    auto& stream_info = _file_itm_data->lvl_stream_info[level];
    stream_info.reserve(selected_stripes.size() * selected_stripes.front().stripe_info.size() *
                        num_columns);

    const bool use_index =
      _use_index &&
      // Do stripes have row group index
      _metadata.is_row_grp_idx_present() &&
      // Only use if we don't have much work with complete columns & stripes
      // TODO: Consider nrows, gpu, and tune the threshold
      (rows_to_read > _metadata.get_row_index_stride() && !(_metadata.get_row_index_stride() & 7) &&
       _metadata.get_row_index_stride() > 0 && num_columns * total_num_stripes < 8 * 128) &&
      // Only use if first row is aligned to a stripe boundary
      // TODO: Fix logic to handle unaligned rows
      (rows_to_skip == 0);

    // Tracker for eventually deallocating compressed and uncompressed data
    auto& stripe_data = lvl_stripe_data[level];

    lvl_stripe_is_empty[level].reserve(selected_stripes.size() *
                                       selected_stripes.front().stripe_info.size());

    std::size_t num_dict_entries = 0;
    std::size_t stripe_idx       = 0;

    std::vector<std::pair<std::future<std::size_t>, std::size_t>> read_tasks;
    for (auto const& stripe_source_mapping : selected_stripes) {
      // Iterate through the source files selected stripes
      for (auto const& stripe : stripe_source_mapping.stripe_info) {
        auto const stripe_info   = stripe.first;
        auto const stripe_footer = stripe.second;

        auto stream_count = stream_info.size();
        auto const total_data_size =
          gather_stream_info_and_update_chunks(stripe_idx,
                                               level,
                                               stripe_info,
                                               stripe_footer,
                                               col_meta.orc_col_map[level],
                                               _metadata.get_types(),
                                               use_index,
                                               level == 0,
                                               &num_dict_entries,
                                               stream_info,
                                               chunks);

        lvl_stripe_is_empty[level].push_back(total_data_size == 0);

        //        auto const total_data_size = gather_stream_info(stripe_idx,
        //                                                        level,
        //                                                        stripe_info,
        //                                                        stripe_footer,
        //                                                        col_meta.orc_col_map[level],
        //                                                        _metadata.get_types(),
        //                                                        level == 0,
        //                                                        stream_info);

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

        stripe_idx++;
      }
    }

    for (auto& task : read_tasks) {
      CUDF_EXPECTS(task.first.get() == task.second, "Unexpected discrepancy in bytes read.");
    }

    if (stripe_data.empty()) { continue; }

    lvl_num_dict_entries[level] = num_dict_entries;

    // Setup row group descriptors if using indexes
    if (_metadata.per_file_metadata[0].ps.compression != orc::NONE) {
      auto const& decompressor = *_metadata.per_file_metadata[0].decompressor;
      cudf::detail::hostdevice_vector<gpu::CompressedStreamInfo> compinfo(
        0, stream_info.size(), _stream);

      for (auto const& info : stream_info) {
        compinfo.push_back(gpu::CompressedStreamInfo(
          static_cast<uint8_t const*>(stripe_data[info.stripe_idx].data()) + info.dst_pos,
          info.length));
        stream_compinfo_map[stream_id_info{
          info.stripe_idx, info.level, info.orc_col_idx, info.kind}] =
          &compinfo[compinfo.size() - 1];
#ifdef PRINT_DEBUG
        printf("collec stream [%d, %d, %d, %d]: dst = %lu,  length = %lu\n",
               (int)info.stripe_idx,
               (int)info.level,
               (int)info.orc_col_idx,
               (int)info.kind,
               info.dst_pos,
               info.length);
        fflush(stdout);
#endif
      }

      compinfo.host_to_device_async(_stream);

      gpu::ParseCompressedStripeData(compinfo.device_ptr(),
                                     compinfo.size(),
                                     decompressor.GetBlockSize(),
                                     decompressor.GetLog2MaxCompressionRatio(),
                                     _stream);
      compinfo.device_to_host_sync(_stream);

      auto& compinfo_map = _file_itm_data->compinfo_map;
      for (auto& [stream_id, stream_compinfo] : stream_compinfo_map) {
        compinfo_map[stream_id] = {stream_compinfo->num_compressed_blocks,
                                   stream_compinfo->num_uncompressed_blocks,
                                   stream_compinfo->max_uncompressed_size};
#ifdef PRINT_DEBUG
        printf("cache info [%d, %d, %d, %d]:  %lu | %lu | %lu\n",
               (int)stream_id.stripe_idx,
               (int)stream_id.level,
               (int)stream_id.orc_col_idx,
               (int)stream_id.kind,
               (size_t)stream_compinfo->num_compressed_blocks,
               (size_t)stream_compinfo->num_uncompressed_blocks,
               stream_compinfo->max_uncompressed_size);
        fflush(stdout);
#endif
      }

      // Must clear so we will not overwrite the old compression info stream_id.
      stream_compinfo_map.clear();

    } else {
      // printf("no compression \n");
      // fflush(stdout);

      // Set decompressed data size equal to the input size.
      // TODO
    }

    // printf("  end level %d\n\n", (int)level);

  }  // end loop level

  // lvl_stripe_data.clear();
  _file_itm_data->compinfo_ready = true;
}

}  // namespace cudf::io::orc::detail
