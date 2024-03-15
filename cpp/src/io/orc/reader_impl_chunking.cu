/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "io/orc/reader_impl.hpp"
#include "io/orc/reader_impl_chunking.hpp"
#include "io/orc/reader_impl_helpers.hpp"
#include "io/utilities/config_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/timezone.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/logger.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iterator>

//
//
//
#include <cudf_test/debug_utilities.hpp>

#include <cudf/detail/utilities/linked_column.hpp>
//
//
//

namespace cudf::io::orc::detail {

std::size_t gather_stream_info_and_column_desc(
  std::size_t global_stripe_order,
  std::size_t level,
  orc::StripeInformation const* stripeinfo,
  orc::StripeFooter const* stripefooter,
  host_span<int const> orc2gdf,
  host_span<orc::SchemaType const> types,
  bool use_index,
  bool apply_struct_map,
  int64_t* num_dictionary_entries,
  std::size_t* local_stream_order,
  std::optional<std::vector<orc_stream_info>*> const& stream_info,
  std::optional<cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>*> const& chunks)
{
  CUDF_EXPECTS(stream_info.has_value() ^ chunks.has_value(),
               "Either stream_info or chunks must be provided, but not both.");

  std::size_t src_offset = 0;
  std::size_t dst_offset = 0;

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
      CUDF_LOG_WARN("Unexpected stream in the input ORC source. The stream will be ignored.");
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
      if (!schema_type.subtypes.empty() && schema_type.kind == orc::STRUCT &&
          stream.kind == orc::PRESENT) {
        for (auto const& idx : schema_type.subtypes) {
          auto const child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
          if (child_idx >= 0) {
            col = child_idx;
            if (chunks.has_value()) {
              auto& chunk                     = (*chunks.value())[global_stripe_order][col];
              chunk.strm_id[gpu::CI_PRESENT]  = *local_stream_order;
              chunk.strm_len[gpu::CI_PRESENT] = stream.length;
            }
          }
        }
      }
    } else if (col != -1) {
      if (chunks.has_value()) {
        if (src_offset >= stripeinfo->indexLength || use_index) {
          auto const index_type = get_stream_index_type(stream.kind);
          if (index_type < gpu::CI_NUM_STREAMS) {
            auto& chunk                = (*chunks.value())[global_stripe_order][col];
            chunk.strm_id[index_type]  = *local_stream_order;
            chunk.strm_len[index_type] = stream.length;
            // NOTE: skip_count field is temporarily used to track the presence of index streams
            chunk.skip_count |= 1 << index_type;

            if (index_type == gpu::CI_DICTIONARY) {
              chunk.dictionary_start = *num_dictionary_entries;
              chunk.dict_len         = stripefooter->columns[column_id].dictionarySize;
              *num_dictionary_entries +=
                static_cast<int64_t>(stripefooter->columns[column_id].dictionarySize);
            }
          }
        }

        (*local_stream_order)++;
      } else {  // not chunks.has_value()
        stream_info.value()->emplace_back(
          stripeinfo->offset + src_offset,
          dst_offset,
          stream.length,
          stream_source_info{global_stripe_order, level, column_id, stream.kind});
      }

      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

template <typename T>
std::vector<range> find_splits(host_span<T const> cumulative_sizes,
                               std::size_t total_count,
                               std::size_t size_limit)
{
  CUDF_EXPECTS(size_limit > 0, "Invalid size limit");

  std::vector<range> splits;
  std::size_t cur_count{0};
  int64_t cur_pos{0};
  std::size_t cur_cumulative_size{0};

  [[maybe_unused]] std::size_t cur_cumulative_rows{0};

  auto const start = thrust::make_transform_iterator(
    cumulative_sizes.begin(),
    [&](auto const& size) { return size.size_bytes - cur_cumulative_size; });
  auto const end = start + cumulative_sizes.size();

  while (cur_count < total_count) {
    int64_t split_pos = static_cast<int64_t>(
      thrust::distance(start, thrust::lower_bound(thrust::seq, start + cur_pos, end, size_limit)));

    // If we're past the end, or if the returned range has size exceeds the given size limit,
    // move back one position.
    if (split_pos >= static_cast<int64_t>(cumulative_sizes.size()) ||
        (cumulative_sizes[split_pos].size_bytes > cur_cumulative_size + size_limit)) {
      split_pos--;
    }

    if constexpr (std::is_same_v<T, cumulative_size_and_row>) {
      // Similarly, while the returned range has total number of rows exceeds column size limit,
      // move back one position.
      while (split_pos > 0 && cumulative_sizes[split_pos].rows >
                                cur_cumulative_rows +
                                  static_cast<std::size_t>(std::numeric_limits<size_type>::max())) {
        split_pos--;
      }
    }

    // In case we have moved back too much in the steps above, far beyond the last split point, that
    // means we could not find any range that has size fits within the given size limit.
    // In such situations, we need to move forward until we move pass the last output range.
    while (split_pos < (static_cast<int64_t>(cumulative_sizes.size()) - 1) &&
           (split_pos < 0 || cumulative_sizes[split_pos].count <= cur_count)) {
      split_pos++;
    }

    auto const start_count = cur_count;
    cur_count              = cumulative_sizes[split_pos].count;
    splits.emplace_back(range{start_count, cur_count});
    cur_pos             = split_pos;
    cur_cumulative_size = cumulative_sizes[split_pos].size_bytes;

    if constexpr (std::is_same_v<T, cumulative_size_and_row>) {
      cur_cumulative_rows = cumulative_sizes[split_pos].rows;
    }
  }

  // If the last range has size smaller than `merge_threshold` the size of the second last one,
  // merge it with the second last one.
  // This is to prevent having too small trailing range.
  if (splits.size() > 1) {
    double constexpr merge_threshold = 0.15;
    if (auto const last = splits.back(), second_last = splits[splits.size() - 2];
        (last.end - last.begin) <=
        static_cast<std::size_t>(merge_threshold * (second_last.end - second_last.begin))) {
      splits.pop_back();
      splits.back().end = last.end;
    }
  }

  return splits;
}

// Since `find_splits` is a template function, we need to explicitly instantiate it so it can be
// used outside of this TU.
template std::vector<range> find_splits<cumulative_size>(host_span<cumulative_size const> sizes,
                                                         std::size_t total_count,
                                                         std::size_t size_limit);
template std::vector<range> find_splits<cumulative_size_and_row>(
  host_span<cumulative_size_and_row const> sizes, std::size_t total_count, std::size_t size_limit);

range get_range(std::vector<range> const& input_ranges, range const& selected_ranges)
{
  // The first and last range.
  auto const& first_range = input_ranges[selected_ranges.begin];
  auto const& last_range  = input_ranges[selected_ranges.end - 1];

  // The range of data covered from the first to the last range.
  return {first_range.begin, last_range.end};
}

void reader::impl::global_preprocess(read_mode mode)
{
  if (_file_itm_data.global_preprocessed) { return; }
  _file_itm_data.global_preprocessed = true;

  //
  // Load stripes' metadata:
  //
  std::tie(
    _file_itm_data.rows_to_skip, _file_itm_data.rows_to_read, _file_itm_data.selected_stripes) =
    _metadata.select_stripes(
      _config.selected_stripes, _config.skip_rows, _config.num_read_rows, _stream);
  if (_file_itm_data.has_no_data()) { return; }

  CUDF_EXPECTS(
    mode == read_mode::CHUNKED_READ ||
      _file_itm_data.rows_to_read <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
    "READ_ALL mode does not support reading number of rows more than cudf's column size limit.",
    std::overflow_error);

  auto const& selected_stripes = _file_itm_data.selected_stripes;
  auto const num_total_stripes = selected_stripes.size();
  auto const num_levels        = _selected_columns.num_levels();

  //
  // Pre allocate necessary memory for data processed in the other reading steps:
  //
  auto& stripe_data_read_ranges = _file_itm_data.stripe_data_read_ranges;
  stripe_data_read_ranges.resize(num_total_stripes);

  auto& lvl_stripe_data          = _file_itm_data.lvl_stripe_data;
  auto& lvl_stripe_sizes         = _file_itm_data.lvl_stripe_sizes;
  auto& lvl_stream_info          = _file_itm_data.lvl_stream_info;
  auto& lvl_stripe_stream_ranges = _file_itm_data.lvl_stripe_stream_ranges;
  auto& lvl_column_types         = _file_itm_data.lvl_column_types;
  auto& lvl_nested_cols          = _file_itm_data.lvl_nested_cols;

  lvl_stripe_data.resize(num_levels);
  lvl_stripe_sizes.resize(num_levels);
  lvl_stream_info.resize(num_levels);
  lvl_stripe_stream_ranges.resize(num_levels);
  lvl_column_types.resize(num_levels);
  lvl_nested_cols.resize(num_levels);
  _out_buffers.resize(num_levels);

  auto& read_info = _file_itm_data.data_read_info;
  auto& col_meta  = *_col_meta;

  //
  // Collect columns' types.
  //

  for (std::size_t level = 0; level < num_levels; ++level) {
    lvl_stripe_sizes[level].resize(num_total_stripes);
    lvl_stripe_stream_ranges[level].resize(num_total_stripes);

    // Association between each ORC column and its cudf::column
    col_meta.orc_col_map.emplace_back(_metadata.get_num_cols(), -1);

    auto const& columns_level = _selected_columns.levels[level];
    size_type col_id{0};

    for (auto const& col : columns_level) {
      // Map each ORC column to its column
      col_meta.orc_col_map[level][col.id] = col_id++;

      auto const col_type =
        to_cudf_type(_metadata.get_col_type(col.id).kind,
                     _config.use_np_dtypes,
                     _config.timestamp_type.id(),
                     to_cudf_decimal_type(_config.decimal128_columns, _metadata, col.id));
      CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");

      auto& column_types = lvl_column_types[level];
      auto& nested_cols  = lvl_nested_cols[level];

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

      // Map each ORC column to its column.
      if (col_type == type_id::LIST or col_type == type_id::STRUCT) {
        nested_cols.emplace_back(col);
      }
    }

    // Try to reserve some memory, but the final size is unknown,
    // since each column may have more than one stream.
    auto const num_columns = columns_level.size();
    lvl_stream_info[level].reserve(num_total_stripes * num_columns);
    if (read_info.capacity() < num_total_stripes * num_columns) {
      read_info.reserve(num_total_stripes * num_columns);
    }
  }

  //
  // Collect all data streams' information:
  //

  // Accumulate data size for data streams in each stripe.
  cudf::detail::hostdevice_vector<cumulative_size> total_stripe_sizes(num_total_stripes, _stream);

  for (std::size_t stripe_global_idx = 0; stripe_global_idx < num_total_stripes;
       ++stripe_global_idx) {
    auto const& stripe       = selected_stripes[stripe_global_idx];
    auto const stripe_info   = stripe.stripe_info;
    auto const stripe_footer = stripe.stripe_footer;

    std::size_t this_stripe_size{0};
    auto const last_read_size = read_info.size();
    for (std::size_t level = 0; level < num_levels; ++level) {
      auto& stream_info = _file_itm_data.lvl_stream_info[level];

      auto stream_level_count = stream_info.size();
      auto const stripe_level_size =
        gather_stream_info_and_column_desc(stripe_global_idx,
                                           level,
                                           stripe_info,
                                           stripe_footer,
                                           col_meta.orc_col_map[level],
                                           _metadata.get_types(),
                                           false,  // use_index,
                                           level == 0,
                                           nullptr,  // num_dictionary_entries
                                           nullptr,  // local_stream_order
                                           &stream_info,
                                           std::nullopt  // chunks
        );

      auto const is_stripe_data_empty = stripe_level_size == 0;
      CUDF_EXPECTS(not is_stripe_data_empty or stripe_info->indexLength == 0,
                   "Invalid index rowgroup stream data");

      lvl_stripe_sizes[level][stripe_global_idx] = stripe_level_size;
      this_stripe_size += stripe_level_size;

      // Range of the streams in `stream_info` corresponding to this stripe at the current level.
      lvl_stripe_stream_ranges[level][stripe_global_idx] =
        range{stream_level_count, stream_info.size()};

      // Coalesce consecutive streams into one read.
      while (not is_stripe_data_empty and stream_level_count < stream_info.size()) {
        auto const d_dst  = stream_info[stream_level_count].dst_pos;
        auto const offset = stream_info[stream_level_count].offset;
        auto len          = stream_info[stream_level_count].length;
        stream_level_count++;

        while (stream_level_count < stream_info.size() &&
               stream_info[stream_level_count].offset == offset + len) {
          len += stream_info[stream_level_count].length;
          stream_level_count++;
        }
        read_info.emplace_back(offset, d_dst, len, stripe.source_idx, stripe_global_idx, level);
      }
    }  // end loop level

    total_stripe_sizes[stripe_global_idx] = {1, this_stripe_size};

    // Range of all stream reads in `read_info` corresponding to this stripe, in all levels.
    stripe_data_read_ranges[stripe_global_idx] = range{last_read_size, read_info.size()};
  }

  //
  // Split list of all stripes into subsets that be loaded separately without blowing up memory:
  //

  _chunk_read_data.curr_load_stripe_range = 0;

  // Load all stripes if there is no read limit.
  if (_chunk_read_data.data_read_limit == 0) {
    _chunk_read_data.load_stripe_ranges = {range{0UL, num_total_stripes}};
    return;
  }

  // TODO: exec_policy_nosync
  // Compute the prefix sum of stripes' data sizes.
  total_stripe_sizes.host_to_device_async(_stream);
  thrust::inclusive_scan(rmm::exec_policy(_stream),  // todo no sync
                         total_stripe_sizes.d_begin(),
                         total_stripe_sizes.d_end(),
                         total_stripe_sizes.d_begin(),
                         cumulative_size_sum{});
  total_stripe_sizes.device_to_host_sync(_stream);

  auto const load_limit = [&] {
    auto const tmp = static_cast<std::size_t>(_chunk_read_data.data_read_limit *
                                              chunk_read_data::load_limit_ratio);
    // Make sure not to pass 0 byte limit (due to round-off) to `find_splits`.
    return tmp > 0UL ? tmp : 1UL;
  }();

  _chunk_read_data.load_stripe_ranges =
    find_splits<cumulative_size>(total_stripe_sizes, num_total_stripes, load_limit);
}

// Load each chunk from `load_stripe_chunks`.
void reader::impl::load_data()
{
  if (_file_itm_data.has_no_data()) { return; }

  auto const load_stripe_range =
    _chunk_read_data.load_stripe_ranges[_chunk_read_data.curr_load_stripe_range++];
  auto const stripe_start = load_stripe_range.begin;
  auto const stripe_end   = load_stripe_range.end;
  auto const stripe_count = stripe_end - stripe_start;

  auto const num_levels = _selected_columns.num_levels();

#ifdef LOCAL_TEST
  printf("\n\nloading data from stripe %d -> %d\n", (int)stripe_start, (int)stripe_end);
#endif

  auto& lvl_stripe_data = _file_itm_data.lvl_stripe_data;

  // Prepare the buffer to read raw data onto.
  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    auto& stripe_data = lvl_stripe_data[level];
    stripe_data.resize(stripe_count);

    for (std::size_t idx = 0; idx < stripe_count; ++idx) {
      auto const stripe_size = _file_itm_data.lvl_stripe_sizes[level][idx + stripe_start];
      stripe_data[idx]       = rmm::device_buffer(
        cudf::util::round_up_safe(stripe_size, BUFFER_PADDING_MULTIPLE), _stream);
    }
  }

  //
  // Load stripe data into memory:
  //

  // After loading data from sources into host buffers, we need to transfer (async) data to device.
  // Such host buffers need to be kept alive until we sync device.
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> host_read_buffers;

  // If we load data directly from sources into device, we also need to the entire read tasks.
  // Thus, we need to keep all read tasks alive and sync all together.
  std::vector<std::pair<std::future<std::size_t>, std::size_t>> read_tasks;

  auto const [read_begin, read_end] =
    get_range(_file_itm_data.stripe_data_read_ranges, load_stripe_range);

  for (auto read_idx = read_begin; read_idx < read_end; ++read_idx) {
    auto const& read_info = _file_itm_data.data_read_info[read_idx];
    auto const source     = _metadata.per_file_metadata[read_info.source_idx].source;
    auto const dst_base   = static_cast<uint8_t*>(
      lvl_stripe_data[read_info.level][read_info.stripe_idx - stripe_start].data());

    if (source->is_device_read_preferred(read_info.length)) {
      read_tasks.push_back(
        std::pair(source->device_read_async(
                    read_info.offset, read_info.length, dst_base + read_info.dst_pos, _stream),
                  read_info.length));

    } else {
      auto buffer = source->host_read(read_info.offset, read_info.length);
      CUDF_EXPECTS(buffer->size() == read_info.length, "Unexpected discrepancy in bytes read.");
      CUDF_CUDA_TRY(cudaMemcpyAsync(dst_base + read_info.dst_pos,
                                    buffer->data(),
                                    read_info.length,
                                    cudaMemcpyDefault,
                                    _stream.value()));
      host_read_buffers.emplace_back(std::move(buffer));
    }
  }

  if (host_read_buffers.size() > 0) { _stream.synchronize(); }
  for (auto& task : read_tasks) {
    CUDF_EXPECTS(task.first.get() == task.second, "Unexpected discrepancy in bytes read.");
  }

  //
  // Split list of all stripes into subsets that be loaded separately without blowing up memory:
  //

  // A map from stripe source into `CompressedStreamInfo*` pointer.
  // These pointers are then used to retrieve stripe/level decompressed sizes for later
  // decompression and decoding.
  stream_source_map<gpu::CompressedStreamInfo*> stream_compinfo_map;

  // For estimating the decompressed sizes of the loaded stripes.
  cudf::detail::hostdevice_vector<cumulative_size_and_row> stripe_decomp_sizes(stripe_count,
                                                                               _stream);
  std::size_t num_loaded_stripes{0};
  for (std::size_t stripe_idx = 0; stripe_idx < stripe_count; ++stripe_idx) {
    auto const& stripe              = _file_itm_data.selected_stripes[stripe_idx];
    auto const stripe_info          = stripe.stripe_info;
    stripe_decomp_sizes[stripe_idx] = cumulative_size_and_row{1, 0, stripe_info->numberOfRows};
    num_loaded_stripes += stripe_info->numberOfRows;
  }

  auto& compinfo_map = _file_itm_data.compinfo_map;

  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    auto const& stream_info = _file_itm_data.lvl_stream_info[level];
    auto const num_columns  = _selected_columns.levels[level].size();

    auto& stripe_data = lvl_stripe_data[level];
    if (stripe_data.empty()) { continue; }

    auto const stream_range =
      get_range(_file_itm_data.lvl_stripe_stream_ranges[level], load_stripe_range);
    auto const num_streams = stream_range.end - stream_range.begin;

    if (_metadata.per_file_metadata[0].ps.compression != orc::NONE) {
      auto const& decompressor = *_metadata.per_file_metadata[0].decompressor;

      // Cannot be cached as-is, since this is for streams in a loaded stripe range, while
      // the latter decompression/decoding step will use a different stripe range.
      cudf::detail::hostdevice_vector<gpu::CompressedStreamInfo> compinfo(0, num_streams, _stream);

      for (auto stream_idx = stream_range.begin; stream_idx < stream_range.end; ++stream_idx) {
        auto const& info = stream_info[stream_idx];
        auto const dst_base =
          static_cast<uint8_t const*>(stripe_data[info.source.stripe_idx - stripe_start].data());

        compinfo.push_back(gpu::CompressedStreamInfo(dst_base + info.dst_pos, info.length));
        stream_compinfo_map[stream_source_info{
          info.source.stripe_idx, info.source.level, info.source.orc_col_idx, info.source.kind}] =
          &compinfo.back();

#ifdef LOCAL_TEST
        printf("collec stream [%d, %d, %d, %d]: dst = %lu,  length = %lu\n",
               (int)info.source.stripe_idx,
               (int)info.source.level,
               (int)info.source.orc_col_idx,
               (int)info.source.kind,
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

      for (auto& [stream_id, stream_compinfo] : stream_compinfo_map) {
        // Cache these parsed numbers so they can be reused in the decompression/decoding step.
        compinfo_map[stream_id] = {stream_compinfo->num_compressed_blocks,
                                   stream_compinfo->num_uncompressed_blocks,
                                   stream_compinfo->max_uncompressed_size};
        stripe_decomp_sizes[stream_id.stripe_idx - stripe_start].size_bytes +=
          stream_compinfo->max_uncompressed_size;

#ifdef LOCAL_TEST
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

      // Important: must clear this map since the next level will have similar keys.
      stream_compinfo_map.clear();

    } else {
#ifdef LOCAL_TEST
      printf("no compression \n");
      fflush(stdout);
#endif

      // Set decompression size equal to the input size.
      for (auto stream_idx = stream_range.begin; stream_idx < stream_range.end; ++stream_idx) {
        auto const& info = stream_info[stream_idx];
        stripe_decomp_sizes[info.source.stripe_idx - stripe_start].size_bytes += info.length;
      }
    }

    // printf("  end level %d\n\n", (int)level);

  }  // end loop level

  // Decoding range is reset to start from the first position in `decode_stripe_ranges`.
  _chunk_read_data.curr_decode_stripe_range = 0;

  // Decode all loaded stripes if there is no read limit.
  // In theory, we should just decode enough stripes for output one table chunk, instead of
  // decoding all stripes like this.
  // However, we do not know how many stripes are 'enough' because there is not any simple and
  // cheap way to compute the exact decoded sizes of stripes.
  if (_chunk_read_data.data_read_limit == 0 &&
      // In addition to not have any read limit, we also need to check if the the total number of
      // rows in the loaded stripes exceeds column size limit.
      num_loaded_stripes < static_cast<std::size_t>(std::numeric_limits<size_type>::max())) {
#ifdef LOCAL_TEST
    printf("0 limit: output decode stripe chunk unchanged\n");
#endif

    _chunk_read_data.decode_stripe_ranges = {load_stripe_range};
    return;
  }

#ifdef LOCAL_TEST
  // TODO: remove
  if (_chunk_read_data.data_read_limit == 0) { printf("0 limit but size overflow\n"); }

  {
    int count{0};
    for (auto& size : stripe_decomp_sizes) {
      printf("decomp stripe size: %ld, %zu, %zu\n", size.count, size.size_bytes, size.rows);
      if (count++ > 5) break;
    }
  }
#endif

  // TODO: exec_policy_nosync
  // Compute the prefix sum of stripe data sizes and rows.
  stripe_decomp_sizes.host_to_device_async(_stream);
  thrust::inclusive_scan(rmm::exec_policy(_stream),
                         stripe_decomp_sizes.d_begin(),
                         stripe_decomp_sizes.d_end(),
                         stripe_decomp_sizes.d_begin(),
                         cumulative_size_sum{});
  stripe_decomp_sizes.device_to_host_sync(_stream);

#ifdef LOCAL_TEST
  {
    int count{0};
    for (auto& size : stripe_decomp_sizes) {
      printf(
        "prefix sum decomp stripe size: %ld, %zu, %zu\n", size.count, size.size_bytes, size.rows);
      if (count++ > 5) break;
    }
  }
#endif

  auto const decode_limit = [&] {
    // In this case, we have no read limit but have to split due to having number of rows in loaded
    // stripes exceeds column size limit. So we will split based on row number, not data size.
    if (_chunk_read_data.data_read_limit == 0) { return std::numeric_limits<std::size_t>::max(); }

    // If `data_read_limit` is too small, make sure not to pass 0 byte limit to compute splits.
    auto const tmp = static_cast<std::size_t>(_chunk_read_data.data_read_limit *
                                              chunk_read_data::decode_limit_ratio);
    return tmp > 0UL ? tmp : 1UL;
  }();
  _chunk_read_data.decode_stripe_ranges =
    find_splits<cumulative_size_and_row>(stripe_decomp_sizes, stripe_count, decode_limit);

  // The split ranges always start from zero.
  // We need to update the ranges to start from `stripe_start` which is covererd by the current
  // range of loaded stripes.
  for (auto& range : _chunk_read_data.decode_stripe_ranges) {
    range.begin += stripe_start;
    range.end += stripe_start;
  }

#ifdef LOCAL_TEST
  auto& splits = _chunk_read_data.decode_stripe_ranges;
  printf("------------\nSplits decode_stripe_chunks (/%d): \n", (int)stripe_count);
  for (size_t idx = 0; idx < splits.size(); idx++) {
    printf("{%ld, %ld}\n", splits[idx].begin, splits[idx].end);
  }

  auto peak_mem = mem_stats_logger.peak_memory_usage();
  std::cout << "load, peak_memory_usage: " << peak_mem << "("
            << (peak_mem * 1.0) / (1024.0 * 1024.0) << " MB)" << std::endl;
  fflush(stdout);
#endif
}

}  // namespace cudf::io::orc::detail
