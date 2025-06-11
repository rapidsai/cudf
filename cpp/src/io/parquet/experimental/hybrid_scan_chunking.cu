/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_impl.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"

namespace cudf::io::parquet::experimental::detail {

rmm::device_buffer hybrid_scan_reader_impl::decompress_dictionary_page_data(
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<PageInfo> pages,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return std::get<0>(parquet::detail::decompress_page_data(chunks, pages, {}, {}, {}, stream, mr));
}

}  // namespace cudf::io::parquet::experimental::detail
