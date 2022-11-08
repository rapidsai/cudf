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

#include <cudf/detail/nvtx/ranges.hpp>
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

std::vector<uint8_t> ingest_raw_input(host_span<std::unique_ptr<datasource>> const& sources,
                                      compression_type compression,
                                      size_t range_offset,
                                      size_t range_size)
{
  CUDF_FUNC_RANGE();
  // Iterate through the user defined sources and read the contents into the local buffer
  auto const total_source_size = sources_size(sources, range_offset, range_size);
  auto buffer                  = std::vector<uint8_t>(total_source_size);

  size_t bytes_read = 0;
  for (const auto& source : sources) {
    if (!source->is_empty()) {
      auto data_size   = (range_size != 0) ? range_size : source->size();
      auto destination = buffer.data() + bytes_read;
      bytes_read += source->host_read(range_offset, data_size, destination);
    }
  }

  if (compression == compression_type::NONE) {
    return buffer;
  } else {
    return decompress(compression, buffer);
  }
}

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader_opts.get_byte_range_offset() == 0 and reader_opts.get_byte_range_size() == 0,
               "specifying a byte range is not yet supported");

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
