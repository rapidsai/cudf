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

#include "reader_impl.hpp"

namespace cudf::io::orc::detail {

// Constructor and destructor are defined within this translation unit.
reader::reader()  = default;
reader::~reader() = default;

reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
               orc_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl{std::make_unique<impl>(std::move(sources), options, stream, mr)}
{
}

table_with_metadata reader::read(orc_reader_options const& options)
{
  return _impl->read(options.get_skip_rows(), options.get_num_rows(), options.get_stripes());
}

chunked_reader::chunked_reader(std::size_t chunk_read_limit,
                               std::vector<std::unique_ptr<datasource>>&& sources,
                               orc_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  _impl = std::make_unique<impl>(chunk_read_limit, std::move(sources), options, stream, mr);
}

chunked_reader::~chunked_reader() = default;

bool chunked_reader::has_next() const { return _impl->has_next(); }

table_with_metadata chunked_reader::read_chunk() const { return _impl->read_chunk(); }

}  // namespace cudf::io::orc::detail
