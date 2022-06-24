/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf::lists::detail {

/**
 * @copydoc cudf::lists::apply_boolean_mask(lists_column_view const&, lists_column_view const&,
 * rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<column> apply_boolean_mask(
  lists_column_view const& input,
  lists_column_view const& boolean_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::list::distinct
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> distinct(
  lists_column_view const& input,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Remove duplicate list elements from a lists column.
 *
 * The input lists column is not given to this function directly. Instead, its child column and a
 * label array containing the corresponding list labels for each element are used to access the
 * input lists. The output null mask and null count are also provided as the input into this
 * function.
 *
 * This function performs exactly the same as the API
 * `cudf::lists::distinct(lists_column_view const& input)` but requires a different set of
 * parameters. This is because it is called internally in various APIs where the label array and the
 * output null_mask and null_count already exist.
 *
 * @param n_lists Number of lists in the input and output lists columns
 * @param child_labels Array containing labels of the list elements
 * @param child The child column of the input lists column
 * @param null_mask The null_mask used for constructing the output column
 * @param null_count The null_count used for constructing the output column
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether floating-point NaNs should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return A pair of output columns `{out_offsets, out_child}`
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> distinct(
  size_type n_lists,
  column_view const& child_labels,
  column_view const& child,
  rmm::device_buffer&& null_mask,
  size_type null_count,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cudf::lists::detail
