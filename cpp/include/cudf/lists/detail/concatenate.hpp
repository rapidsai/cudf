/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace lists::detail {

/**
 * @brief Returns a single column by concatenating the given vector of
 * lists columns.
 *
 * @code{.pseudo}
 * s1 = [{0, 1}, {2, 3, 4, 5, 6}]
 * s2 = [{7, 8, 9}, {10, 11}]
 * r = concatenate(s1, s2)
 * r is now [{0, 1}, {2, 3, 4, 5, 6}, {7, 8, 9}, {10, 11}]
 * @endcode
 *
 * @param columns Vector of lists columns to concatenate.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace CUDF_EXPORT cudf
