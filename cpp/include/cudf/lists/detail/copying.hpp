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

#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace lists::detail {

/**
 * @brief Returns a new lists column created from a subset of the
 * lists column. The subset of lists selected is between start (inclusive)
 * and end (exclusive).
 *
 * @code{.pseudo}
 * Example:
 * s1 = {{1, 2, 3}, {4, 5}, {6, 7}, {}, {8, 9}}
 * s2 = slice( s1, 1, 4)
 * s2 is {{4, 5}, {6, 7}, {}}
 * @endcode
 *
 * @param lists Lists instance for this operation.
 * @param start Index to first list to select in the column
 * @param end One past the index to last list to select in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New lists column of size (end - start)
 */
std::unique_ptr<cudf::column> copy_slice(lists_column_view const& lists,
                                         size_type start,
                                         size_type end,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace CUDF_EXPORT cudf
