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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
class column_device_view;

namespace reduction {
namespace detail {

/**
 * @brief Compute the number of elements per segment
 *
 * If `null_handling == null_policy::EXCLUDE`, the count for each
 * segment omits any null entries. Otherwise, this returns the number
 * of elements in each segment.
 *
 * @param null_mask Null values over which the segment offsets apply
 * @param has_nulls True if d_col contains any nulls
 * @param offsets Indices to segment boundaries
 * @param null_handling How null entries are processed within each segment
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The number of elements in each segment
 */
rmm::device_uvector<size_type> segmented_counts(bitmask_type const* null_mask,
                                                bool has_nulls,
                                                device_span<size_type const> offsets,
                                                null_policy null_handling,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
