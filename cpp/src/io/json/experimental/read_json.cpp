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

std::vector<uint8_t> ingest_raw_input(host_span<std::unique_ptr<datasource>> sources,
                                      compression_type compression)
{
  auto const total_source_size =
    std::accumulate(sources.begin(), sources.end(), 0ul, [](size_t sum, auto& source) {
      return sum + source->size();
    });
  auto buffer = std::vector<uint8_t>(total_source_size);

  size_t bytes_read = 0;
  for (const auto& source : sources) {
    bytes_read += source->host_read(0, source->size(), buffer.data() + bytes_read);
  }

  return (compression == compression_type::NONE) ? buffer : decompress(compression, buffer);
}

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(reader_opts.get_byte_range_offset() == 0 and reader_opts.get_byte_range_size() == 0,
               "specifying a byte range is not yet supported");

  auto const buffer = ingest_raw_input(sources, reader_opts.get_compression());
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
