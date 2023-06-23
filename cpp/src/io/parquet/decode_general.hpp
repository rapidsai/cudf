/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <io/parquet/decode.cuh>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

void DecodePageDataGeneral(cudf::detail::hostdevice_vector<PageInfo>& pages,
                           cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                           size_t num_rows,
                           size_t min_row,
                           int level_type_size,
                           rmm::cuda_stream_view stream);

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf