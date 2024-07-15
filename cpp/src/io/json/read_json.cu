/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "io/comp/io_uncomp.hpp"
#include "io/json/nested_json.hpp"
#include "read_json.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>

#include <numeric>

namespace cudf::io::json::detail {

size_t sources_size(host_span<std::unique_ptr<datasource>> const sources,
                    size_t range_offset,
                    size_t range_size)
{
  return std::accumulate(sources.begin(), sources.end(), 0ul, [=](size_t sum, auto& source) {
    auto const size = source->size();
    // TODO take care of 0, 0, or *, 0 case.
    return sum +
           (range_size == 0 or range_offset + range_size > size ? size - range_offset : range_size);
  });
}

/**
 * @brief Read from array of data sources into RMM buffer. The size of the returned device span
          can be larger than the number of bytes requested from the list of sources when
          the range to be read spans across multiple sources. This is due to the delimiter
          characters inserted after the end of each accessed source.
 *
 * @param buffer Device span buffer to which data is read
 * @param sources Array of data sources
 * @param compression Compression format of source
 * @param range_offset Number of bytes to skip from source start
 * @param range_size Number of bytes to read from source
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns A subspan of the input device span containing data read
 */
device_span<char> ingest_raw_input(device_span<char> buffer,
                                   host_span<std::unique_ptr<datasource>> sources,
                                   compression_type compression,
                                   size_t range_offset,
                                   size_t range_size,
                                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // We append a line delimiter between two files to make sure the last line of file i and the first
  // line of file i+1 don't end up on the same JSON line, if file i does not already end with a line
  // delimiter.
  auto constexpr num_delimiter_chars = 1;

  if (compression == compression_type::NONE) {
    auto delimiter_map = cudf::detail::make_empty_host_vector<size_t>(sources.size(), stream);
    std::vector<size_t> prefsum_source_sizes(sources.size());
    std::vector<std::unique_ptr<datasource::buffer>> h_buffers;
    size_t bytes_read = 0;
    std::transform_inclusive_scan(sources.begin(),
                                  sources.end(),
                                  prefsum_source_sizes.begin(),
                                  std::plus<size_t>{},
                                  [](std::unique_ptr<datasource> const& s) { return s->size(); });
    auto upper =
      std::upper_bound(prefsum_source_sizes.begin(), prefsum_source_sizes.end(), range_offset);
    size_t start_source = std::distance(prefsum_source_sizes.begin(), upper);

    auto const total_bytes_to_read =
      std::min(range_size, prefsum_source_sizes.back() - range_offset);
    range_offset -= start_source ? prefsum_source_sizes[start_source - 1] : 0;
    for (size_t i = start_source; i < sources.size() && bytes_read < total_bytes_to_read; i++) {
      if (sources[i]->is_empty()) continue;
      auto data_size =
        std::min(sources[i]->size() - range_offset, total_bytes_to_read - bytes_read);
      auto destination = reinterpret_cast<uint8_t*>(buffer.data()) + bytes_read +
                         (num_delimiter_chars * delimiter_map.size());
      if (sources[i]->is_device_read_preferred(data_size)) {
        bytes_read += sources[i]->device_read(range_offset, data_size, destination, stream);
      } else {
        h_buffers.emplace_back(sources[i]->host_read(range_offset, data_size));
        auto const& h_buffer = h_buffers.back();
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          destination, h_buffer->data(), h_buffer->size(), cudaMemcpyHostToDevice, stream.value()));
        bytes_read += h_buffer->size();
      }
      range_offset = 0;
      delimiter_map.push_back(bytes_read + (num_delimiter_chars * delimiter_map.size()));
    }
    // Removing delimiter inserted after last non-empty source is read
    if (!delimiter_map.empty()) { delimiter_map.pop_back(); }

    // If this is a multi-file source, we scatter the JSON line delimiters between files
    if (sources.size() > 1) {
      static_assert(num_delimiter_chars == 1,
                    "Currently only single-character delimiters are supported");
      auto const delimiter_source = thrust::make_constant_iterator('\n');
      auto const d_delimiter_map  = cudf::detail::make_device_uvector_async(
        delimiter_map, stream, rmm::mr::get_current_device_resource());
      thrust::scatter(rmm::exec_policy_nosync(stream),
                      delimiter_source,
                      delimiter_source + d_delimiter_map.size(),
                      d_delimiter_map.data(),
                      buffer.data());
    }
    stream.synchronize();
    return buffer.first(bytes_read + (delimiter_map.size() * num_delimiter_chars));
  }
  // TODO: allow byte range reading from multiple compressed files.
  auto remaining_bytes_to_read = std::min(range_size, sources[0]->size() - range_offset);
  auto hbuffer                 = std::vector<uint8_t>(remaining_bytes_to_read);
  // Single read because only a single compressed source is supported
  // Reading to host because decompression of a single block is much faster on the CPU
  sources[0]->host_read(range_offset, remaining_bytes_to_read, hbuffer.data());
  auto uncomp_data = decompress(compression, hbuffer);
  CUDF_CUDA_TRY(cudaMemcpyAsync(buffer.data(),
                                reinterpret_cast<char*>(uncomp_data.data()),
                                uncomp_data.size() * sizeof(char),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  stream.synchronize();
  return buffer.first(uncomp_data.size());
}

size_type find_first_delimiter_in_chunk(host_span<std::unique_ptr<cudf::io::datasource>> sources,
                                        json_reader_options const& reader_opts,
                                        char const delimiter,
                                        rmm::cuda_stream_view stream)
{
  auto total_source_size = sources_size(sources, 0, 0) + (sources.size() - 1);
  rmm::device_uvector<char> buffer(total_source_size, stream);
  auto readbufspan = ingest_raw_input(buffer,
                                      sources,
                                      reader_opts.get_compression(),
                                      reader_opts.get_byte_range_offset(),
                                      reader_opts.get_byte_range_size(),
                                      stream);
  return find_first_delimiter(readbufspan, '\n', stream);
}

/**
 * @brief Get the byte range between record starts and ends starting from the given range.
 *
 * if get_byte_range_offset == 0, then we can skip the first delimiter search
 * if get_byte_range_offset != 0, then we need to search for the first delimiter in given range.
 * if not found, skip this chunk, if found, then search for first delimiter in next range until we
 * find a delimiter. Use this as actual range for parsing.
 *
 * @param sources Data sources to read from
 * @param reader_opts JSON reader options with range offset and range size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns Data source owning buffer enclosing the bytes read
 */
datasource::owning_buffer<rmm::device_uvector<char>> get_record_range_raw_input(
  host_span<std::unique_ptr<datasource>> sources,
  json_reader_options const& reader_opts,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto geometric_mean = [](double a, double b) { return std::sqrt(a * b); };

  size_t const total_source_size            = sources_size(sources, 0, 0);
  auto constexpr num_delimiter_chars        = 1;
  auto const num_extra_delimiters           = num_delimiter_chars * (sources.size() - 1);
  compression_type const reader_compression = reader_opts.get_compression();
  size_t const chunk_offset                 = reader_opts.get_byte_range_offset();
  size_t chunk_size                         = reader_opts.get_byte_range_size();

  CUDF_EXPECTS(total_source_size ? chunk_offset < total_source_size : !chunk_offset,
               "Invalid offsetting",
               std::invalid_argument);
  auto should_load_all_sources = !chunk_size || chunk_size >= total_source_size - chunk_offset;
  chunk_size = should_load_all_sources ? total_source_size - chunk_offset : chunk_size;

  // Some magic numbers
  constexpr int num_subchunks               = 10;  // per chunk_size
  constexpr size_t min_subchunk_size        = 10000;
  int const num_subchunks_prealloced        = should_load_all_sources ? 0 : 3;
  constexpr int estimated_compression_ratio = 4;

  // NOTE: heuristic for choosing subchunk size: geometric mean of minimum subchunk size (set to
  // 10kb) and the byte range size

  size_t const size_per_subchunk =
    geometric_mean(std::ceil((double)chunk_size / num_subchunks), min_subchunk_size);

  // The allocation for single source compressed input is estimated by assuming a ~4:1
  // compression ratio. For uncompressed inputs, we can getter a better estimate using the idea
  // of subchunks.
  auto constexpr header_size = 4096;
  size_t const buffer_size =
    reader_compression != compression_type::NONE
      ? total_source_size * estimated_compression_ratio + header_size
      : std::min(total_source_size, chunk_size + num_subchunks_prealloced * size_per_subchunk) +
          num_extra_delimiters;
  rmm::device_uvector<char> buffer(buffer_size, stream);
  device_span<char> bufspan(buffer);

  // Offset within buffer indicating first read position
  std::int64_t buffer_offset = 0;
  auto readbufspan =
    ingest_raw_input(bufspan, sources, reader_compression, chunk_offset, chunk_size, stream);

  auto const shift_for_nonzero_offset = std::min<std::int64_t>(chunk_offset, 1);
  auto const first_delim_pos =
    chunk_offset == 0 ? 0 : find_first_delimiter(readbufspan, '\n', stream);
  if (first_delim_pos == -1) {
    // return empty owning datasource buffer
    auto empty_buf = rmm::device_uvector<char>(0, stream);
    return datasource::owning_buffer<rmm::device_uvector<char>>(std::move(empty_buf));
  } else if (!should_load_all_sources) {
    // Find next delimiter
    std::int64_t next_delim_pos = -1;
    size_t next_subchunk_start  = chunk_offset + chunk_size;
    while (next_subchunk_start < total_source_size && next_delim_pos < buffer_offset) {
      buffer_offset += readbufspan.size();
      readbufspan    = ingest_raw_input(bufspan.last(buffer_size - buffer_offset),
                                     sources,
                                     reader_compression,
                                     next_subchunk_start,
                                     size_per_subchunk,
                                     stream);
      next_delim_pos = find_first_delimiter(readbufspan, '\n', stream) + buffer_offset;
      if (next_delim_pos < buffer_offset) { next_subchunk_start += size_per_subchunk; }
    }
    if (next_delim_pos < buffer_offset) next_delim_pos = buffer_offset + readbufspan.size();

    return datasource::owning_buffer<rmm::device_uvector<char>>(
      std::move(buffer),
      reinterpret_cast<uint8_t*>(buffer.data()) + first_delim_pos + shift_for_nonzero_offset,
      next_delim_pos - first_delim_pos - shift_for_nonzero_offset);
  }
  return datasource::owning_buffer<rmm::device_uvector<char>>(
    std::move(buffer),
    reinterpret_cast<uint8_t*>(buffer.data()) + first_delim_pos + shift_for_nonzero_offset,
    readbufspan.size() - first_delim_pos - shift_for_nonzero_offset);
}

table_with_metadata read_batch(host_span<std::unique_ptr<datasource>> sources,
                               json_reader_options const& reader_opts,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  datasource::owning_buffer<rmm::device_uvector<char>> bufview =
    get_record_range_raw_input(sources, reader_opts, stream);

  // If input JSON buffer has single quotes and option to normalize single quotes is enabled,
  // invoke pre-processing FST
  if (reader_opts.is_enabled_normalize_single_quotes()) {
    normalize_single_quotes(bufview, stream, rmm::mr::get_current_device_resource());
  }

  // If input JSON buffer has unquoted spaces and tabs and option to normalize whitespaces is
  // enabled, invoke pre-processing FST
  if (reader_opts.is_enabled_normalize_whitespace()) {
    normalize_whitespace(bufview, stream, rmm::mr::get_current_device_resource());
  }

  auto buffer =
    cudf::device_span<char const>(reinterpret_cast<char const*>(bufview.data()), bufview.size());
  stream.synchronize();
  return device_parse_nested_json(buffer, reader_opts, stream, mr);
}

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (reader_opts.get_byte_range_offset() != 0 or reader_opts.get_byte_range_size() != 0) {
    CUDF_EXPECTS(reader_opts.is_enabled_lines(),
                 "Specifying a byte range is supported only for JSON Lines");
  }

  if (sources.size() > 1) {
    CUDF_EXPECTS(reader_opts.get_compression() == compression_type::NONE,
                 "Multiple compressed inputs are not supported");
    CUDF_EXPECTS(reader_opts.is_enabled_lines(),
                 "Multiple inputs are supported only for JSON Lines format");
  }

  std::for_each(sources.begin(), sources.end(), [](auto const& source) {
    CUDF_EXPECTS(source->size() < std::numeric_limits<int>::max(),
                 "The size of each source file must be less than INT_MAX bytes");
  });

  constexpr size_t batch_size_ub = std::numeric_limits<int>::max();
  size_t const chunk_offset      = reader_opts.get_byte_range_offset();
  size_t chunk_size              = reader_opts.get_byte_range_size();
  chunk_size                     = !chunk_size ? sources_size(sources, 0, 0) : chunk_size;

  // Identify the position of starting source file from which to begin batching based on
  // byte range offset. If the offset is larger than the sum of all source
  // sizes, then start_source is total number of source files i.e. no file is read
  size_t const start_source = [&]() {
    size_t sum = 0;
    for (size_t src_idx = 0; src_idx < sources.size(); ++src_idx) {
      if (sum + sources[src_idx]->size() > chunk_offset) return src_idx;
      sum += sources[src_idx]->size();
    }
    return sources.size();
  }();

  // Construct batches of source files, with starting position of batches indicated by
  // batch_positions. The size of each batch i.e. the sum of sizes of the source files in the batch
  // is capped at INT_MAX bytes.
  size_t cur_size = 0;
  std::vector<size_t> batch_positions;
  std::vector<size_t> batch_sizes;
  batch_positions.push_back(0);
  for (size_t i = start_source; i < sources.size(); i++) {
    cur_size += sources[i]->size();
    if (cur_size >= batch_size_ub) {
      batch_positions.push_back(i);
      batch_sizes.push_back(cur_size - sources[i]->size());
      cur_size = sources[i]->size();
    }
  }
  batch_positions.push_back(sources.size());
  batch_sizes.push_back(cur_size);

  // If there is a single batch, then we can directly return the table without the
  // unnecessary concatenate
  if (batch_sizes.size() == 1) return read_batch(sources, reader_opts, stream, mr);

  std::vector<cudf::io::table_with_metadata> partial_tables;
  json_reader_options batched_reader_opts{reader_opts};

  // Dispatch individual batches to read_batch and push the resulting table into
  // partial_tables array. Note that the reader options need to be updated for each
  // batch to adjust byte range offset and byte range size.
  for (size_t i = 0; i < batch_sizes.size(); i++) {
    batched_reader_opts.set_byte_range_size(std::min(batch_sizes[i], chunk_size));
    partial_tables.emplace_back(read_batch(
      host_span<std::unique_ptr<datasource>>(sources.begin() + batch_positions[i],
                                             batch_positions[i + 1] - batch_positions[i]),
      batched_reader_opts,
      stream,
      rmm::mr::get_current_device_resource()));
    if (chunk_size <= batch_sizes[i]) break;
    chunk_size -= batch_sizes[i];
    batched_reader_opts.set_byte_range_offset(0);
  }

  auto expects_schema_equality =
    std::all_of(partial_tables.begin() + 1,
                partial_tables.end(),
                [&gt = partial_tables[0].metadata.schema_info](auto& ptbl) {
                  return ptbl.metadata.schema_info == gt;
                });
  CUDF_EXPECTS(expects_schema_equality,
               "Mismatch in JSON schema across batches in multi-source multi-batch reading");

  auto partial_table_views = std::vector<cudf::table_view>(partial_tables.size());
  std::transform(partial_tables.begin(),
                 partial_tables.end(),
                 partial_table_views.begin(),
                 [](auto const& table) { return table.tbl->view(); });
  return table_with_metadata{cudf::concatenate(partial_table_views, stream, mr),
                             {partial_tables[0].metadata.schema_info}};
}

}  // namespace cudf::io::json::detail
