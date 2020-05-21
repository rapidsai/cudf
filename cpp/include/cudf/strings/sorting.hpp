/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Sort types for the sort method.
 **/
enum sort_type {
  none   = 0,  ///< no sorting
  length = 1,  ///< sort by string length
  name   = 2   ///< sort by characters code-points
};

/**
 * @brief Returns a new strings column that is a sorted version of the
 * strings in this instance.
 *
 * @param strings Strings instance for this operation.
 * @param stype Specify what attribute of the string to sort on.
 * @param order Sort strings in ascending or descending order.
 * @param null_order Sort nulls to the beginning or the end of the new column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<cudf::column> sort(
  strings_column_view strings,
  sort_type stype,
  cudf::order order                   = cudf::order::ASCENDING,
  cudf::null_order null_order         = cudf::null_order::BEFORE,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace detail
}  // namespace strings
}  // namespace cudf
