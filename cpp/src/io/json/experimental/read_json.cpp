/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "read_json.hpp"
#include "cudf/io/json.hpp"
#include "rmm/device_vector.hpp"

#include <io/comp/io_uncomp.hpp>
#include <io/json/nested_json.hpp>

#include <cudf/utilities/error.hpp>

#include <numeric>

namespace cudf::io::detail::json::experimental {

std::vector<uint8_t> ingest_raw_input(host_span<std::unique_ptr<datasource>> sources,
                                      compression_type compression,
                                      size_t range_offset,
                                      size_t range_size);

// function to extract first delimiter in the string in each chunk
// share
// collate together and form byte_range for each chunk.
// parse separately.
// join together.
// rmm::device_scalar<thrust::pair<size_type, size_type>> find_first_and_last_delimiter(
//   device_span<char const> d_data,
//   char const delimiter,
//   rmm::cuda_stream_view stream,
//   rmm::mr::device_memory_resource* mr);

size_type find_first_delimiter(device_span<char const> d_data,
                               char const delimiter,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr);

// rmm::device_scalar<size_type>
size_type find_first_delimiter_in_chunk(host_span<std::unique_ptr<datasource>> sources,
                                        json_reader_options const& reader_opts,
                                        char const delimiter,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  auto const buffer = ingest_raw_input(sources,
                                       reader_opts.get_compression(),
                                       reader_opts.get_byte_range_offset(),
                                       reader_opts.get_byte_range_size());
  // auto data = host_span<char const>(reinterpret_cast<char const*>(buffer.data()), buffer.size());
  auto d_data = rmm::device_uvector<char>(buffer.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_data.data(),
                                buffer.data(),
                                buffer.size() * sizeof(decltype(buffer)::value_type),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  return find_first_delimiter(d_data, delimiter, stream, mr);
}

std::vector<table_with_metadata> skeleton_for_parellel_chunk_reader(
  host_span<std::unique_ptr<datasource>> sources,
  json_reader_options const& reader_opts,
  int chunk_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // assuming single source.
  auto reader_opts_chunk = reader_opts;
  auto const total_source_size =
    std::accumulate(sources.begin(), sources.end(), 0ul, [](size_t sum, auto& source) {
      return sum + source->size();
    });
  size_t num_chunks                = (total_source_size + chunk_size - 1) / chunk_size;
  constexpr size_type no_min_value = std::numeric_limits<size_type>::max();
  std::vector<size_type> first_delimiter_index(num_chunks);
  for (size_t i = 0; i < num_chunks; i++) {
    auto const chunk_start = i * chunk_size;
    reader_opts_chunk.set_byte_range_offset(chunk_start);
    reader_opts_chunk.set_byte_range_size(chunk_size);
    first_delimiter_index[i] =
      find_first_delimiter_in_chunk(sources, reader_opts_chunk, '\n', stream, mr);
    if (first_delimiter_index[i] != no_min_value) { first_delimiter_index[i] += chunk_start; }
  }
  for (auto i : first_delimiter_index) {
    std::cout << i << std::endl;
  }
  // process and allocate record start, end for each worker.
  using record_range = std::pair<size_type, size_type>;
  std::vector<record_range> record_ranges(num_chunks, record_range{-1, -1});
  first_delimiter_index[0] = 0;
  // record_ranges.back().second = total_source_size;
  for (size_t i = 0; i < num_chunks;) {
    record_ranges[i].first    = first_delimiter_index[i];
    auto next_delimiter_chunk = i + 1;
    while (next_delimiter_chunk < num_chunks and
           first_delimiter_index[next_delimiter_chunk] == no_min_value) {
      next_delimiter_chunk++;
    }
    if (next_delimiter_chunk < num_chunks) {
      record_ranges[i].second = first_delimiter_index[next_delimiter_chunk];
    } else {
      record_ranges[i].second = total_source_size;
    }
    i = next_delimiter_chunk;
  }

  for (auto range : record_ranges) {
    std::cout << "[" << range.first << "," << range.second << "]" << std::endl;
  }

  std::vector<table_with_metadata> tables;  //(num_chunks);
  // process each chunk in parallel.
  for (size_t i = 0; i < num_chunks; i++) {
    auto const [chunk_start, chunk_end] = record_ranges[i];
    if (chunk_start == -1 or chunk_end == -1) continue;
    reader_opts_chunk.set_byte_range_offset(chunk_start);
    reader_opts_chunk.set_byte_range_size(chunk_end - chunk_start);
    tables.push_back(read_json(sources, reader_opts_chunk, stream, mr));
  }
  // assume all records have same number of columns, and inferred same type. (or schema is passed)
  // TODO a step before to merge all columns, types and infer final schema.
  return tables;
}

std::vector<uint8_t> ingest_raw_input(host_span<std::unique_ptr<datasource>> sources,
                                      compression_type compression,
                                      size_t range_offset,
                                      size_t range_size)
{
  auto const total_source_size =
    std::accumulate(sources.begin(), sources.end(), 0ul, [=](size_t sum, auto& source) {
      auto const size = source->size();
      // TODO take care of 0, 0, or *, 0 case.
      return sum + (range_size == 0 or range_offset + range_size > size ? size - range_offset
                                                                        : range_size);
    });
  auto buffer = std::vector<uint8_t>(total_source_size);

  size_t bytes_read = 0;
  for (const auto& source : sources) {
    auto const data_size =
      (range_size == 0 or range_offset + range_size > source->size() ? source->size() - range_offset
                                                                     : range_size);
    //  FIXME: I can't see why we concatenate strings from multiple sources, but it's how it's done
    //  in csv.
    auto destination = buffer.data() + bytes_read;
    bytes_read += source->host_read(range_offset, data_size, destination);
  }

  return (compression == compression_type::NONE) ? buffer : decompress(compression, buffer);
}

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  // CUDF_EXPECTS(reader_opts.get_byte_range_offset() == 0 and reader_opts.get_byte_range_size() ==
  // 0,
  //              "specifying a byte range is not yet supported");

  auto const buffer = ingest_raw_input(sources,
                                       reader_opts.get_compression(),
                                       reader_opts.get_byte_range_offset(),
                                       reader_opts.get_byte_range_size());
  auto data = host_span<char const>(reinterpret_cast<char const*>(buffer.data()), buffer.size());

  try {
    return cudf::io::json::detail::device_parse_nested_json(data, reader_opts, stream, mr);
  } catch (cudf::logic_error const& err) {
#ifdef NJP_DEBUG_PRINT
    std::cout << "Fall back to host nested json parser" << std::endl;
#endif
    return cudf::io::json::detail::host_parse_nested_json(data, reader_opts, stream, mr);
  }
}

}  // namespace cudf::io::detail::json::experimental
