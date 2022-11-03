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

#include <io/comp/io_uncomp.hpp>
#include <io/json/nested_json.hpp>

#include <cudf/utilities/error.hpp>

#include <numeric>

namespace cudf::io::detail::json::experimental {

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

std::vector<uint8_t> ingest_raw_input(host_span<std::unique_ptr<datasource>> sources,
                                      compression_type compression,
                                      size_t range_offset,
                                      size_t range_size)
{
  auto const total_source_size = sources_size(sources, range_offset, range_size);
  auto buffer                  = std::vector<uint8_t>(total_source_size);

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

size_type find_first_delimiter_in_chunk(host_span<std::unique_ptr<cudf::io::datasource>> sources,
                                        json_reader_options const& reader_opts,
                                        char const delimiter,
                                        rmm::cuda_stream_view stream)
{
  auto const buffer = ingest_raw_input(sources,
                                       reader_opts.get_compression(),
                                       reader_opts.get_byte_range_offset(),
                                       reader_opts.get_byte_range_size());
  auto d_data       = rmm::device_uvector<char>(buffer.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_data.data(),
                                buffer.data(),
                                buffer.size() * sizeof(decltype(buffer)::value_type),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  return find_first_delimiter(d_data, delimiter, stream);
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
