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

#include "parquet_gpu.hpp"

namespace cudf {
namespace io {
namespace parquet {
namespace detail {
void DecodePageDataFixed(cudf::detail::hostdevice_vector<PageInfo>& pages,
                         cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                         std::size_t num_rows,
                         std::size_t min_row,
                         int level_type_size,
                         rmm::cuda_stream_view stream);

void DecodePageDataFixedDict(cudf::detail::hostdevice_vector<PageInfo>& pages,
                             cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                             std::size_t num_rows,
                             std::size_t min_row,
                             int level_type_size,
                             rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace parquet
}  // namespace io
}  // namespace cudf
