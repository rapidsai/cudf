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
std::size_t gather_stream_info(std::size_t stripe_index,
                               std::size_t level,
                               orc::StripeInformation const* stripeinfo,
                               orc::StripeFooter const* stripefooter,
                               host_span<int const> orc2gdf,
                               host_span<orc::SchemaType const> types,
                               bool apply_struct_map,
                               std::vector<orc_stream_info>& stream_info)
{
  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;

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
            if (child_idx >= 0) { col = child_idx; }
          }
        }
      }
    }

    if (col != -1) {
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

struct cumulative_size {
  std::size_t count;
  std::size_t size_bytes;
};

struct cumulative_size_sum {
  __device__ cumulative_size operator()(cumulative_size const& a, cumulative_size const& b) const
  {
    return cumulative_size{a.count + b.count, a.size_bytes + b.size_bytes};
  }
};

#if 0
std::vector<chunk> find_splits(host_span<cumulative_size const> sizes,
                               size_type num_rows,
                               size_t size_limit)
{
  std::vector<chunk> splits;

  uint32_t cur_count         = 0;
  int64_t cur_pos            = 0;
  size_t cur_cumulative_size = 0;
  auto const start           = thrust::make_transform_iterator(
    sizes.begin(), [&](auto const& size) { return size.size_bytes - cur_cumulative_size; });
  auto const end = start + static_cast<int64_t>(sizes.size());
  while (cur_count < static_cast<uint32_t>(num_rows)) {
    int64_t split_pos =
      thrust::distance(start, thrust::lower_bound(thrust::seq, start + cur_pos, end, size_limit));

    // If we're past the end, or if the returned bucket is bigger than the chunk_read_limit, move
    // back one.
    if (static_cast<size_t>(split_pos) >= sizes.size() ||
        (sizes[split_pos].size_bytes - cur_cumulative_size > size_limit)) {
      split_pos--;
    }

    // best-try. if we can't find something that'll fit, we have to go bigger. we're doing this in
    // a loop because all of the cumulative sizes for all the pages are sorted into one big list.
    // so if we had two columns, both of which had an entry {1000, 10000}, that entry would be in
    // the list twice. so we have to iterate until we skip past all of them.  The idea is that we
    // either do this, or we have to call unique() on the input first.
    while (split_pos < (static_cast<int64_t>(sizes.size()) - 1) &&
           (split_pos < 0 || sizes[split_pos].count == cur_count)) {
      split_pos++;
    }

    auto const start_row = cur_count;
    cur_count            = sizes[split_pos].count;
    splits.emplace_back(chunk{start_row, static_cast<size_type>(cur_count - start_row)});
    cur_pos             = split_pos;
    cur_cumulative_size = sizes[split_pos].size_bytes;
  }

  return splits;
}
#endif

}  // namespace

void reader::impl::query_stripe_compression_info()
{
  if (_file_itm_data->compinfo_ready) { return; }
  if (_selected_columns.num_levels() == 0) { return; }

  auto const rows_to_read      = _file_itm_data->rows_to_read;
  auto const& selected_stripes = _file_itm_data->selected_stripes;

  // If no rows or stripes to read, return empty columns
  // TODO : remove?
  if (rows_to_read == 0 || selected_stripes.empty()) { return; }

  auto& lvl_stripe_data  = _file_itm_data->lvl_stripe_data;
  auto& lvl_stripe_sizes = _file_itm_data->lvl_stripe_sizes;
  lvl_stripe_data.resize(_selected_columns.num_levels());
  lvl_stripe_sizes.resize(_selected_columns.num_levels());

  auto& read_info = _file_itm_data->read_info;

  // TODO: Don't have to keep it for all stripe/level. Can reset it after each iter.
  std::unordered_map<stream_id_info, gpu::CompressedStreamInfo*, stream_id_hash, stream_id_equal>
    stream_compinfo_map;

  // Logically view streams as columns
  _file_itm_data->lvl_stream_info.resize(_selected_columns.num_levels());

  // Get the total number of stripes across all input files.
  std::size_t num_stripes = selected_stripes.size();

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

    lvl_stripe_data[level].resize(num_stripes);

    auto& stream_info      = _file_itm_data->lvl_stream_info[level];
    auto const num_columns = _selected_columns.levels[level].size();
    auto& stripe_sizes     = lvl_stripe_sizes[level];
    stream_info.reserve(selected_stripes.size() * num_columns);  // final size is unknown

    stripe_sizes.resize(selected_stripes.size());
    if (read_info.capacity() < selected_stripes.size()) {
      read_info.reserve(selected_stripes.size() * num_columns);  // final size is unknown
    }
  }

  cudf::detail::hostdevice_vector<cumulative_size> total_stripe_sizes(num_stripes, _stream);

  // Compute input size for each stripe.
  for (std::size_t stripe_idx = 0; stripe_idx < num_stripes; ++stripe_idx) {
    auto const& stripe       = selected_stripes[stripe_idx];
    auto const stripe_info   = stripe.stripe_info;
    auto const stripe_footer = stripe.stripe_footer;

    std::size_t total_stripe_size{0};
    for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
      auto& stream_info  = _file_itm_data->lvl_stream_info[level];
      auto& stripe_sizes = lvl_stripe_sizes[level];

      auto stream_count        = stream_info.size();
      auto const stripe_size   = gather_stream_info(stripe_idx,
                                                  level,
                                                  stripe_info,
                                                  stripe_footer,
                                                  col_meta.orc_col_map[level],
                                                  _metadata.get_types(),
                                                  level == 0,
                                                  stream_info);
      stripe_sizes[stripe_idx] = stripe_size;
      total_stripe_size += stripe_size;

      auto const is_stripe_data_empty = stripe_size == 0;
      CUDF_EXPECTS(not is_stripe_data_empty or stripe_info->indexLength == 0,
                   "Invalid index rowgroup stream data");

      // Coalesce consecutive streams into one read
      while (not is_stripe_data_empty and stream_count < stream_info.size()) {
        auto const d_dst  = stream_info[stream_count].dst_pos;
        auto const offset = stream_info[stream_count].offset;
        auto len          = stream_info[stream_count].length;
        stream_count++;

        while (stream_count < stream_info.size() &&
               stream_info[stream_count].offset == offset + len) {
          len += stream_info[stream_count].length;
          stream_count++;
        }
        read_info.emplace_back(offset, len, d_dst, stripe.source_idx, stripe_idx, level);
      }
    }
    total_stripe_sizes[stripe_idx] = {1, total_stripe_size};
  }

  // Compute the prefix sum of stripe data sizes.
  total_stripe_sizes.host_to_device_async(_stream);
  thrust::inclusive_scan(rmm::exec_policy(_stream),
                         total_stripe_sizes.d_begin(),
                         total_stripe_sizes.d_end(),
                         total_stripe_sizes.d_begin(),
                         cumulative_size_sum{});

  total_stripe_sizes.device_to_host_sync(_stream);

  //  fix this:
  //  _file_itm_data->stripe_chunks =
  //    find_splits(total_stripe_sizes, _file_itm_data->rows_to_read, /*chunk_size_limit*/ 0);

  //  std::cout << "  total rows: " << _file_itm_data.rows_to_read << std::endl;
  //  print_cumulative_row_info(stripe_size_bytes, "  ", _chunk_read_info.chunks);

  // Prepare the buffer to read raw data onto.
  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    auto& stripe_data  = lvl_stripe_data[level];
    auto& stripe_sizes = lvl_stripe_sizes[level];
    for (std::size_t stripe_idx = 0; stripe_idx < num_stripes; ++stripe_idx) {
      stripe_data[stripe_idx] = rmm::device_buffer(
        cudf::util::round_up_safe(stripe_sizes[stripe_idx], BUFFER_PADDING_MULTIPLE), _stream);
    }
  }

  std::vector<std::pair<std::future<std::size_t>, std::size_t>> read_tasks;
  // Should not read all, but read stripe by stripe.
  // read_info should be limited by stripe.
  // Read level-by-level.
  // TODO: Test with read and parse/decode column by column.
  // This is future work.
  for (auto const& read : read_info) {
    auto& stripe_data = lvl_stripe_data[read.level];
    auto dst_base     = static_cast<uint8_t*>(stripe_data[read.stripe_idx].data());

    if (_metadata.per_file_metadata[read.source_idx].source->is_device_read_preferred(
          read.length)) {
      read_tasks.push_back(
        std::pair(_metadata.per_file_metadata[read.source_idx].source->device_read_async(
                    read.offset, read.length, dst_base + read.dst_pos, _stream),
                  read.length));

    } else {
      auto const buffer =
        _metadata.per_file_metadata[read.source_idx].source->host_read(read.offset, read.length);
      CUDF_EXPECTS(buffer->size() == read.length, "Unexpected discrepancy in bytes read.");
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        dst_base + read.dst_pos, buffer->data(), read.length, cudaMemcpyDefault, _stream.value()));
      _stream.synchronize();

#if 0
     // This in theory should be faster, but in practice it's slower. Why?
      read_tasks.push_back(
        std::pair(std::async(std::launch::async,
                             [&, read = read, dst_base = dst_base] {
                               auto const buffer =
                                 _metadata.per_file_metadata[read.source_idx].source->host_read(
                                   read.offset, read.length);
                               CUDF_EXPECTS(buffer->size() == read.length,
                                            "Unexpected discrepancy in bytes read.");
                               CUDF_CUDA_TRY(cudaMemcpyAsync(dst_base + read.dst_pos,
                                                             buffer->data(),
                                                             read.length,
                                                             cudaMemcpyDefault,
                                                             _stream.value()));
                               _stream.synchronize();
                               return read.length;
                             }),
                  read.length));
#endif
    }
  }
  for (auto& task : read_tasks) {
    CUDF_EXPECTS(task.first.get() == task.second, "Unexpected discrepancy in bytes read.");
  }

  // Parse the decompressed sizes for each stripe.
  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    auto& stream_info      = _file_itm_data->lvl_stream_info[level];
    auto const num_columns = _selected_columns.levels[level].size();

    // Tracker for eventually deallocating compressed and uncompressed data
    auto& stripe_data = lvl_stripe_data[level];
    if (stripe_data.empty()) { continue; }

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
