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
#pragma once

#include "io/json/read_json.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <numeric>

// Helper function to test correctness of JSON byte range reading.
// We split the input source files into a set of byte range chunks each of size
// `chunk_size` and return an array of partial tables constructed from each chunk
template <typename IndexType = std::int32_t>
std::vector<cudf::io::table_with_metadata> split_byte_range_reading(
  cudf::host_span<std::unique_ptr<cudf::io::datasource>> sources,
  cudf::io::json_reader_options const& reader_opts,
  IndexType chunk_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto total_source_size = [&sources]() {
    return std::accumulate(sources.begin(), sources.end(), 0ul, [=](size_t sum, auto& source) {
      auto const size = source->size();
      return sum + size;
    });
  }();
  auto find_first_delimiter_in_chunk =
    [total_source_size, &sources, &stream](
      cudf::io::json_reader_options const& reader_opts) -> IndexType {
    rmm::device_uvector<char> buffer(total_source_size, stream);
    auto readbufspan = cudf::io::json::detail::ingest_raw_input(buffer,
                                                                sources,
                                                                reader_opts.get_compression(),
                                                                reader_opts.get_byte_range_offset(),
                                                                reader_opts.get_byte_range_size(),
                                                                reader_opts.get_delimiter(),
                                                                stream);
    // Note: we cannot reuse cudf::io::json::detail::find_first_delimiter since the
    // return type of that function is size_type. However, when the chunk_size is
    // larger than INT_MAX, the position of the delimiter can also be larger than
    // INT_MAX. We do not encounter this overflow error in the detail function
    // since the batched JSON reader splits the byte_range_size into chunk_sizes
    // smaller than INT_MAX bytes
    auto const first_delimiter_position_it =
      thrust::find(rmm::exec_policy(stream), readbufspan.begin(), readbufspan.end(), '\n');
    return first_delimiter_position_it != readbufspan.end()
             ? thrust::distance(readbufspan.begin(), first_delimiter_position_it)
             : -1;
  };
  size_t num_chunks                = (total_source_size + chunk_size - 1) / chunk_size;
  constexpr IndexType no_min_value = -1;

  // Get the first delimiter in each chunk.
  std::vector<IndexType> first_delimiter_index(num_chunks);
  auto reader_opts_chunk = reader_opts;
  for (size_t i = 0; i < num_chunks; i++) {
    auto const chunk_start = i * chunk_size;
    // We are updating reader_opt_chunks to store offset and size information for the current chunk
    reader_opts_chunk.set_byte_range_offset(chunk_start);
    reader_opts_chunk.set_byte_range_size(chunk_size);
    first_delimiter_index[i] = find_first_delimiter_in_chunk(reader_opts_chunk);
  }

  // Process and allocate record start, end for each worker.
  using record_range = std::pair<size_t, size_t>;
  std::vector<record_range> record_ranges;
  record_ranges.reserve(num_chunks);
  size_t prev = 0;
  for (size_t i = 1; i < num_chunks; i++) {
    // In the case where chunk_size is smaller than row size, the chunk needs to be skipped
    if (first_delimiter_index[i] == no_min_value) continue;
    size_t next = static_cast<size_t>(first_delimiter_index[i]) + (i * chunk_size);
    record_ranges.emplace_back(prev, next);
    prev = next;
  }
  record_ranges.emplace_back(prev, total_source_size);

  std::vector<cudf::io::table_with_metadata> tables;
  for (auto const& [chunk_start, chunk_end] : record_ranges) {
    reader_opts_chunk.set_byte_range_offset(chunk_start);
    reader_opts_chunk.set_byte_range_size(chunk_end - chunk_start);
    tables.push_back(cudf::io::json::detail::read_json(sources, reader_opts_chunk, stream, mr));
  }
  // assume all records have same number of columns, and inferred same type. (or schema is passed)
  // TODO a step before to merge all columns, types and infer final schema.
  return tables;
}
