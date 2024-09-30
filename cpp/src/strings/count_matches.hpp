/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

class column_device_view;

namespace strings {
namespace detail {

class reprog_device;

/**
 * @brief Returns a column of regex match counts for each string in the given column.
 *
 * A null entry will result in a zero count for that output row.
 *
 * @param d_strings Device view of the input strings column.
 * @param d_prog Regex instance to evaluate on each string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Integer column of match counts
 */
std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      reprog_device& d_prog,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
