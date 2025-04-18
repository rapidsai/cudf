/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

/**
 * @brief Returns a pair of row index vectors corresponding to an
 * inner join between the specified tables.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{1, 2}, {0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing an inner join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(cudf::table_view const& left_keys,
           cudf::table_view const& right_keys,
           null_equality compare_nulls       = null_equality::EQUAL,
           rmm::cuda_stream_view stream      = cudf::get_default_stream(),
           rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a
 * left join between the specified tables.
 *
 * The first returned vector contains all the row indices from the left
 * table (in unspecified order). The corresponding value in the
 * second returned vector is either (1) the row index of the matched row
 * from the right table, if there is a match  or  (2) an unspecified
 * out-of-bounds value.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{0, 1, 2}, {None, 0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{0, 1, 2}, {None, 0, None}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a left join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(cudf::table_view const& left_keys,
          cudf::table_view const& right_keys,
          null_equality compare_nulls       = null_equality::EQUAL,
          rmm::cuda_stream_view stream      = cudf::get_default_stream(),
          rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a
 * full join between the specified tables.
 *
 * Taken pairwise, the values from the returned vectors are one of:
 * (1) row indices corresponding to matching rows from the left and
 * right tables, (2) a row index and an unspecified out-of-bounds value,
 * representing a row from one table without a match in the other.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{0, 1, 2, None}, {None, 0, 1, 2}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{0, 1, 2, None, None}, {None, 0, None, 1, 2}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a full join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(cudf::table_view const& left_keys,
          cudf::table_view const& right_keys,
          null_equality compare_nulls       = null_equality::EQUAL,
          rmm::cuda_stream_view stream      = cudf::get_default_stream(),
          rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a vector of row indices corresponding to a left semi-join
 * between the specified tables.
 *
 * The returned vector contains the row indices from the left table
 * for which there is a matching row in the right table.
 *
 * @code{.pseudo}
 * TableA: {{0, 1, 2}}
 * TableB: {{1, 2, 3}}
 * Result: {1, 2}
 * @endcode
 *
 * @param left_keys The left table
 * @param right_keys The right table
 * @param compare_nulls Controls whether null join-key values should match or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector `left_indices` that can be used to construct
 * the result of performing a left semi join between two tables with
 * `left_keys` and `right_keys` as the join keys .
 */
std::unique_ptr<rmm::device_uvector<size_type>> left_semi_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a vector of row indices corresponding to a left anti join
 * between the specified tables.
 *
 * The returned vector contains the row indices from the left table
 * for which there is no matching row in the right table.
 *
 * @code{.pseudo}
 * TableA: {{0, 1, 2}}
 * TableB: {{1, 2, 3}}
 * Result: {0}
 * @endcode
 *
 * @throw cudf::logic_error if the number of columns in either `left_keys` or `right_keys` is 0
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A column `left_indices` that can be used to construct
 * the result of performing a left anti join between two tables with
 * `left_keys` and `right_keys` as the join keys .
 */
std::unique_ptr<rmm::device_uvector<size_type>> left_anti_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a cross join on two tables (`left`, `right`)
 *
 * The cross join returns the cartesian product of rows from each table.
 *
 * @note Warning: This function can easily cause out-of-memory errors. The size of the output is
 * equal to `left.num_rows() * right.num_rows()`. Use with caution.
 *
 * @code{.pseudo}
 * Left a: {0, 1, 2}
 * Right b: {3, 4, 5}
 * Result: { a: {0, 0, 0, 1, 1, 1, 2, 2, 2}, b: {3, 4, 5, 3, 4, 5, 3, 4, 5} }
 * @endcode

 * @throw cudf::logic_error if the number of columns in either `left` or `right` table is 0
 *
 * @param left  The left table
 * @param right The right table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr    Device memory resource used to allocate the returned table's device memory
 *
 * @return     Result of cross joining `left` and `right` tables
 */
std::unique_ptr<cudf::table> cross_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
