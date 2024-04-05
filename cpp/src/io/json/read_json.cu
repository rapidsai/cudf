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
#include "io/json/legacy/read_json.hpp"
#include "io/json/nested_json.hpp"
#include "read_json.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

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
 * @brief Read from array of data sources into RMM buffer
 *
 * @param sources Array of data sources
 * @param compression Compression format of source
 * @param range_offset Number of bytes to skip from source start
 * @param range_size Number of bytes to read from source
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void ingest_raw_input(std::unique_ptr<rmm::device_uvector<char>>& bufptr,
                      host_span<std::unique_ptr<datasource>> sources,
                      compression_type compression,
                      size_t range_offset,
                      size_t range_size,
                      size_t& bufptr_offset,
                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // We append a line delimiter between two files to make sure the last line of file i and the first
  // line of file i+1 don't end up on the same JSON line, if file i does not already end with a line
  // delimiter.
  auto constexpr num_delimiter_chars = 1;
  auto const num_extra_delimiters    = num_delimiter_chars * (sources.size() - 1);

  if (compression == compression_type::NONE) {
    std::vector<size_type> delimiter_map{};
    std::vector<size_t> prefsum_source_sizes(sources.size());
    std::vector<std::unique_ptr<datasource::buffer>> h_buffers;
    delimiter_map.reserve(sources.size());
    size_t bytes_read = 0;
    std::transform(sources.begin(),
                   sources.end(),
                   prefsum_source_sizes.begin(),
                   [](const std::unique_ptr<datasource>& s) { return s->size(); });
    std::inclusive_scan(
      prefsum_source_sizes.begin(), prefsum_source_sizes.end(), prefsum_source_sizes.begin());
    auto upper =
      std::upper_bound(prefsum_source_sizes.begin(), prefsum_source_sizes.end(), range_offset);
    size_t start_source = std::distance(prefsum_source_sizes.begin(), upper);

    range_size = !range_size || range_size > prefsum_source_sizes.back()
                   ? prefsum_source_sizes.back() - range_offset
                   : range_size;
    range_offset =
      start_source ? range_offset - prefsum_source_sizes[start_source - 1] : range_offset;
    for (size_t i = start_source; i < sources.size() && range_size; i++) {
      if (sources[i]->is_empty()) continue;
      auto data_size   = std::min(sources[i]->size() - range_offset, range_size);
      auto destination = reinterpret_cast<uint8_t*>(bufptr->data()) + bufptr_offset + bytes_read;
      if (sources[i]->is_device_read_preferred(data_size)) {
        bytes_read += sources[i]->device_read(range_offset, data_size, destination, stream);
      } else {
        h_buffers.emplace_back(sources[i]->host_read(range_offset, data_size));
        auto const& h_buffer = h_buffers.back();
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          destination, h_buffer->data(), h_buffer->size(), cudaMemcpyDefault, stream.value()));
        bytes_read += h_buffer->size();
      }
      range_offset = 0;
      range_size -= bytes_read;
      delimiter_map.push_back(bytes_read);
      bytes_read += num_delimiter_chars;
    }
    if (bytes_read) bytes_read -= num_delimiter_chars;

    // If this is a multi-file source, we scatter the JSON line delimiters between files
    if (sources.size() > 1) {
      static_assert(num_delimiter_chars == 1,
                    "Currently only single-character delimiters are supported");
      auto const delimiter_source = thrust::make_constant_iterator('\n');
      auto const d_delimiter_map  = cudf::detail::make_device_uvector_async(
        host_span<size_type const>{delimiter_map.data(), delimiter_map.size() - 1},
        stream,
        rmm::mr::get_current_device_resource());
      thrust::scatter(rmm::exec_policy_nosync(stream),
                      delimiter_source,
                      delimiter_source + d_delimiter_map.size(),
                      d_delimiter_map.data(),
                      bufptr->data() + bufptr_offset);
    }
    bufptr_offset += bytes_read;
  } else {
    // Iterate through the user defined sources and read the contents into the local buffer
    auto const total_source_size =
      sources_size(sources, range_offset, range_size) + num_extra_delimiters;
    auto buffer = std::vector<uint8_t>(total_source_size);
    // Single read because only a single compressed source is supported
    // Reading to host because decompression of a single block is much faster on the CPU
    sources[0]->host_read(range_offset, total_source_size, buffer.data());
    auto uncomp_data = decompress(compression, buffer);
    CUDF_CUDA_TRY(cudaMemcpyAsync(bufptr->data() + bufptr_offset,
                                  reinterpret_cast<char*>(uncomp_data.data()),
                                  uncomp_data.size() * sizeof(char),
                                  cudaMemcpyDefault,
                                  stream.value()));
    bufptr_offset += uncomp_data.size() * sizeof(char);
  }
  stream.synchronize();
}

size_type find_first_delimiter_in_chunk(host_span<std::unique_ptr<cudf::io::datasource>> sources,
                                        json_reader_options const& reader_opts,
                                        char const delimiter,
                                        rmm::cuda_stream_view stream)
{
  auto const total_source_size =
    sources_size(sources, reader_opts.get_byte_range_offset(), reader_opts.get_byte_range_size()) +
    (sources.size() - 1);
  auto bufptr          = std::make_unique<rmm::device_uvector<char>>(total_source_size, stream);
  size_t bufptr_offset = 0;
  ingest_raw_input(bufptr,
                   sources,
                   reader_opts.get_compression(),
                   reader_opts.get_byte_range_offset(),
                   reader_opts.get_byte_range_size(),
                   bufptr_offset,
                   stream);
  return find_first_delimiter(*bufptr, delimiter, stream);
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
 * @return Byte range for parsing
 */
datasource::owning_buffer<rmm::device_uvector<char>> get_record_range_raw_input(
  std::unique_ptr<rmm::device_uvector<char>>&& bufptr,
  host_span<std::unique_ptr<datasource>> sources,
  json_reader_options const& reader_opts,
  rmm::cuda_stream_view stream)
{
  auto geometric_mean       = [](double a, double b) { return std::pow(a * b, 0.5); };
  auto find_first_delimiter = [&bufptr, &stream](
                                size_t const start, size_t const end, char const delimiter) {
    auto const first_delimiter_position = thrust::find(
      rmm::exec_policy(stream), bufptr->data() + start, bufptr->data() + end, delimiter);
    return first_delimiter_position != bufptr->begin() + end
             ? first_delimiter_position - bufptr->begin()
             : -1;
  };

  size_t const total_source_size            = sources_size(sources, 0, 0);
  auto constexpr num_delimiter_chars        = 1;
  auto const num_extra_delimiters           = num_delimiter_chars * (sources.size() - 1);
  size_t chunk_size                         = reader_opts.get_byte_range_size();
  size_t chunk_offset                       = reader_opts.get_byte_range_offset();
  compression_type const reader_compression = reader_opts.get_compression();
  constexpr int num_subchunks               = 10;  // per chunk_size
  /*
   * NOTE: heuristic for choosing subchunk size: geometric mean of minimum subchunk size (set to
   * 10kb) and the byte range size
   */
  size_t size_per_subchunk =
    geometric_mean(reader_opts.get_byte_range_size() / num_subchunks, 10000);

  if (!chunk_size || chunk_size > total_source_size - chunk_offset)
    chunk_size = total_source_size - chunk_offset + num_extra_delimiters;
  bufptr = std::make_unique<rmm::device_uvector<char>>(chunk_size + 3 * size_per_subchunk, stream);
  size_t bufptr_offset = 0;

  ingest_raw_input(
    bufptr, sources, reader_compression, chunk_offset, chunk_size, bufptr_offset, stream);

  auto first_delim_pos = chunk_offset == 0 ? 0 : find_first_delimiter(0, bufptr_offset, '\n');
  if (first_delim_pos == -1) {
    // return empty owning datasource buffer
    auto empty_buf = rmm::device_uvector<char>(0, stream);
    return datasource::owning_buffer<rmm::device_uvector<char>>(std::move(empty_buf));
  } else if (reader_opts.get_byte_range_size() > 0 &&
             reader_opts.get_byte_range_size() < total_source_size - chunk_offset) {
    // Find next delimiter
    std::int64_t next_delim_pos = -1;
    size_t next_subchunk_start  = chunk_offset + chunk_size;
    while (next_subchunk_start < total_source_size && next_delim_pos == -1) {
      std::int64_t bytes_read = -bufptr_offset;
      ingest_raw_input(bufptr,
                       sources,
                       reader_compression,
                       next_subchunk_start,
                       size_per_subchunk,
                       bufptr_offset,
                       stream);
      bytes_read += bufptr_offset;
      next_delim_pos = find_first_delimiter(bufptr_offset - bytes_read, bufptr_offset, '\n');
      if (next_delim_pos == -1) { next_subchunk_start += size_per_subchunk; }
    }
    if (next_delim_pos == -1)
      next_delim_pos = total_source_size - (next_subchunk_start - size_per_subchunk);

    auto* released_bufptr = bufptr.release();
    return datasource::owning_buffer<rmm::device_uvector<char>>(
      std::move(*released_bufptr),
      reinterpret_cast<uint8_t*>(released_bufptr->data()) + first_delim_pos +
        (chunk_offset ? 1 : 0),
      next_delim_pos - first_delim_pos);
  }
  auto* released_bufptr = bufptr.release();
  return datasource::owning_buffer<rmm::device_uvector<char>>(
    std::move(*released_bufptr),
    reinterpret_cast<uint8_t*>(released_bufptr->data()) + first_delim_pos + (chunk_offset ? 1 : 0),
    bufptr_offset - first_delim_pos - (chunk_offset ? 1 : 0));
}

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (reader_opts.is_enabled_legacy()) {
    return legacy::read_json(sources, reader_opts, stream, mr);
  }

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

  std::unique_ptr<rmm::device_uvector<char>> bufptr{};
  datasource::owning_buffer<rmm::device_uvector<char>> bufview =
    get_record_range_raw_input(std::move(bufptr), sources, reader_opts, stream);

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
  // For debug purposes, use host_parse_nested_json()
}

}  // namespace cudf::io::json::detail
