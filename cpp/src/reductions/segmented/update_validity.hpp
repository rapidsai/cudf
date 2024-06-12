/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <optional>

namespace cudf {
namespace reduction {
namespace detail {

/**
 * @brief Compute the validity mask and set it on the result column
 *
 * If `null_handling == null_policy::INCLUDE`, all elements in a segment must be valid for the
 * reduced value to be valid.
 * If `null_handling == null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @param result Result of segmented reduce to update the null mask
 * @param col Input column before reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling How null entries are processed within each segment
 * @param init Optional initial value
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
void segmented_update_validity(column& result,
                               column_view const& col,
                               device_span<size_type const> offsets,
                               null_policy null_handling,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
