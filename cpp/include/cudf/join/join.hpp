/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/limits>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

/**
 * @brief Sentinel value used to indicate an unmatched row index in join operations.
 *
 * This value is used in join result indices to represent rows that do not have a match
 * in the other table (e.g., in left joins, full joins, or when using filter_gather_map
 * with null indices from outer joins).
 *
 * The value is set to the minimum possible value for `size_type` to ensure it's easily
 * distinguishable from valid row indices, which are always non-negative.
 */
CUDF_HOST_DEVICE constexpr size_type JoinNoMatch = cuda::std::numeric_limits<size_type>::min();

/**
 * @brief Holds context information about matches between tables during a join operation.
 *
 * This structure stores the left table view and a device vector containing the count of
 * matching rows in the right table for each row in the left table. Used primarily by
 * inner_join_match_context() to track join match information.
 */
struct join_match_context {
  table_view _left_table;  ///< View of the left table involved in the join operation
  std::unique_ptr<rmm::device_uvector<size_type>>
    _match_counts;  ///< A device vector containing the count of matching rows in the right table
                    ///< for each row in left table
};

/**
 * @brief Stores context information for partitioned join operations.
 *
 * This structure maintains context for partitioned join operations, containing the match
 * context from a previous join operation along with the start and end indices that define
 * the current partition of the left table being processed.
 *
 * Used with partitioned_inner_join() to perform large joins in smaller chunks while
 * preserving the context from the initial match operation.
 */
struct join_partition_context {
  join_match_context
    left_table_context;      ///< The match context from a previous inner_join_match_context call
  size_type left_start_idx;  ///< The starting row index of the current left table partition
  size_type left_end_idx;  ///< The ending row index (exclusive) of the current left table partition
};

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
 * from the right table, if there is a match  or  (2) `JoinNoMatch`.
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
 * right tables, (2) a row index and `JoinNoMatch`,
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
 * @deprecated Use the object-oriented filtered_join `cudf::filtered_join::anti_join` instead
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
[[deprecated]] std::unique_ptr<rmm::device_uvector<size_type>> left_semi_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a vector of row indices corresponding to a left anti join
 * between the specified tables.
 *
 * @deprecated Use the object-oriented filtered_join `cudf::filtered_join::semi_join` instead
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
[[deprecated]] std::unique_ptr<rmm::device_uvector<size_type>> left_anti_join(
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
