/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Returns a new strings column created from a subset of
 * of the strings column.
 *
 * The subset of strings selected is between
 * start (inclusive) and end (exclusive).
 *
 * @code{.pseudo}
 * Example:
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * s2 = copy_slice( s1, 2 )
 * s2 is ["c", "d", "e", "f"]
 * s2 = copy_slice( s1, 1, 3 )
 * s2 is ["b", "c"]
 * @endcode
 *
 * @param strings Strings instance for this operation.
 * @param start Index to first string to select in the column (inclusive).
 * @param end Index to last string to select in the column (exclusive).
 *            Default -1 indicates the last element.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column of size (end-start)/step.
 */
std::unique_ptr<cudf::column> copy_slice(
  strings_column_view const& strings,
  size_type start,
  size_type end                       = -1,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a new strings column created by shifting the rows by a specified offset.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b", "c", "d", "e", "f"]
 * r1 = shift(s, 2, "_")
 * r1 is now ["_", "_", "a", "b", "c", "d"]
 * r2 = shift(s, -2, "_")
 * r2 is now ["c", "d", "e", "f", "_", "_"]
 * @endcode
 *
 * The caller should set the validity mask in the output column.
 *
 * @param input Strings instance for this operation.
 * @param offset The offset by which to shift the input.
 * @param fill_value Fill value for indeterminable outputs.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> shift(strings_column_view const& input,
                              size_type offset,
                              scalar const& fill_value,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
