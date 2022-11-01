/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::detail {
/**
 * @copydoc cudf::lower_bound
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> lower_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::upper_bound
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> upper_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::contains(column_view const&, scalar const&, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
bool contains(column_view const& haystack, scalar const& needle, rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::contains(column_view const&, column_view const&, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Check if rows in the given `needles` table exist in the `haystack` table.
 *
 * Given two tables, each row in the `needles` table is checked to see if there is any matching row
 * (i.e., compared equal to it) in the `haystack` table. The boolean search results are written into
 * the corresponding rows of the output array.
 *
 * @code{.pseudo}
 * Example:
 *
 * haystack = { { 5, 4, 1, 2, 3 } }
 * needles  = { { 0, 1, 2 } }
 * output   = { false, true, true }
 * @endcode
 *
 * @param haystack The table containing the search space
 * @param needles A table of rows whose existence to check in the search space
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A vector of bools indicating if each row in `needles` has matching rows in `haystack`
 */
rmm::device_uvector<bool> contains(
  table_view const& haystack,
  table_view const& needles,
  null_equality compare_nulls,
  nan_equality compare_nans,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cudf::detail
