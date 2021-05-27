/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/ast/nodes.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>
#include <utility>
#include <vector>

namespace cudf {
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
           null_equality compare_nulls         = null_equality::EQUAL,
           rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs an inner join on the specified columns of two
 * tables (`left`, `right`)
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{4, 9, 3}, {1, 2, 5}}
 * left_on: {0}
 * right_on: {1}
 * Result: {{1, 2}, {4, 9}, {1, 2}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_on` or `right_on`
 * mismatch.
 * @throw cudf::logic_error if number of columns in either `left` or `right`
 * table is 0 or exceeds MAX_JOIN_SIZE
 * @throw std::out_of_range if element of `left_on` or `right_on` exceed the
 * number of columns in the left or right table.
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column
 * from `left` indicated by `left_on[i]`.
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`.
 */
std::unique_ptr<cudf::table> inner_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
          null_equality compare_nulls         = null_equality::EQUAL,
          rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs a left join (also known as left outer join) on the
 * specified columns of two tables (`left`, `right`)
 *
 * Left join returns all the rows from the left table and those rows from the
 * right table that match on the joined columns.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}, {1, 2 ,5}}
 * left_on: {0}
 * right_on: {1}
 * Result: { {0, 1, 2}, {NULL, 1, 2}, {NULL, 1, 2} }
 *
 * Left: {{0, 1, 2}}
 * Right {{1, 2, 3}, {1, 2, 5}}
 * left_on: {0}
 * right_on: {0}
 * Result: { {0, 1, 2}, {NULL, 1, 2}, {NULL, 1, 2} }
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_on` or `right_on`
 * mismatch.
 * @throw cudf::logic_error if number of columns in either `left` or `right`
 * table is 0 or exceeds MAX_JOIN_SIZE
 * @throw std::out_of_range if element of `left_on` or `right_on` exceed the
 * number of columns in the left or right table.
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column
 * from `left` indicated by `left_on[i]`.
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`.
 */
std::unique_ptr<cudf::table> left_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
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
          null_equality compare_nulls         = null_equality::EQUAL,
          rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs a full join (also known as full outer join) on the
 * specified columns of two tables (`left`, `right`)
 *
 * Full Join returns the rows that would be returned by a left join and those
 * rows from the right table that do not have a match.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}, {1, 2, 5}}
 * left_on: {0}
 * right_on: {1}
 * Result: { {0, 1, 2, NULL}, {NULL, 1, 2, 3}, {NULL, 1, 2, 5} }
 *
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}, {1, 2, 5}}
 * left_on: {0}
 * right_on: {0}
 * Result: { {0, 1, 2, NULL}, {NULL, 1, 2, 3}, {NULL, 1, 2, 5} }
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_on` or `right_on`
 * mismatch.
 * @throw cudf::logic_error if number of columns in either `left` or `right`
 * table is 0 or exceeds MAX_JOIN_SIZE
 * @throw std::out_of_range if element of `left_on` or `right_on` exceed the
 * number of columns in the left or right table.
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column
 * from `left` indicated by `left_on[i]`.
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`.
 */
std::unique_ptr<cudf::table> full_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a vector of row indices corresponding to a left semi join
 * between the specified tables.
 *
 * The returned vector contains the row indices from the left table
 * for which there is a matching row in the right table.
 *
 * @code{.pseudo}
 * TableA: {{0, 1, 2}}
 * TableB: {{1, 2, 3}}
 * right_on: {1}
 * Result: {1, 2}
 * @endcode
 *
 * @throw cudf::logic_error if number of columns in either
 * `left_keys` or `right_keys` table is 0 or exceeds MAX_JOIN_SIZE
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector `left_indices` that can be used to construct
 * the result of performing a left semi join between two tables with
 * `left_keys` and `right_keys` as the join keys .
 */
std::unique_ptr<rmm::device_uvector<size_type>> left_semi_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs a left semi join on the specified columns of two
 * tables (`left`, `right`)
 *
 * A left semi join only returns data from the left table, and only
 * returns rows that exist in the right table.
 *
 * @code{.pseudo}
 * TableA: {{0, 1, 2}}
 * TableB: {{1, 2, 3}, {1, 2, 5}}
 * left_on: {0}
 * right_on: {1}
 * Result: { {1, 2} }
 *
 * TableA {{0, 1, 2}, {1, 2, 5}}
 * TableB {{1, 2, 3}}
 * left_on: {0}
 * right_on: {0}
 * Result: { {1, 2}, {2, 5} }
 * @endcode
 *
 * @throw cudf::logic_error if the number of columns in either `left_keys` or `right_keys` is 0
 *
 * @param[in] left             The left table
 * @param[in] right            The right table
 * @param[in] left_on          The column indices from `left` to join on.
 *                             The column from `left` indicated by `left_on[i]`
 *                             will be compared against the column from `right`
 *                             indicated by `right_on[i]`.
 * @param[in] right_on         The column indices from `right` to join on.
 *                             The column from `right` indicated by `right_on[i]`
 *                             will be compared against the column from `left`
 *                             indicated by `left_on[i]`.
 * @param[in] compare_nulls    Controls whether null join-key values should match or not.
 * @param[in] mr               Device memory resource used to allocate the returned table's
 *                             device memory
 *
 * @return                     Result of joining `left` and `right` tables on the columns
 *                             specified by `left_on` and `right_on`.
 */
std::unique_ptr<cudf::table> left_semi_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A column `left_indices` that can be used to construct
 * the result of performing a left anti join between two tables with
 * `left_keys` and `right_keys` as the join keys .
 */
std::unique_ptr<rmm::device_uvector<size_type>> left_anti_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs a left anti join on the specified columns of two
 * tables (`left`, `right`)
 *
 * A left anti join only returns data from the left table, and only
 * returns rows that do not exist in the right table.
 *
 * @code{.pseudo}
 * TableA: {{0, 1, 2}}
 * TableB: {{1, 2, 3},  {1, 2, 5}}
 * left_on: {0}
 * right_on: {1}
 * Result: {{0}}
 *
 * TableA: {{0, 1, 2}, {1, 2, 5}}
 * TableB: {{1, 2, 3}}
 * left_on: {0}
 * right_on: {0}
 * Result: { {0}, {1} }
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_on` or `right_on`
 * mismatch.
 * @throw cudf::logic_error if number of columns in either `left` or `right`
 * table is 0 or exceeds MAX_JOIN_SIZE
 *
 * @param[in] left             The left table
 * @param[in] right            The right table
 * @param[in] left_on          The column indices from `left` to join on.
 *                             The column from `left` indicated by `left_on[i]`
 *                             will be compared against the column from `right`
 *                             indicated by `right_on[i]`.
 * @param[in] right_on         The column indices from `right` to join on.
 *                             The column from `right` indicated by `right_on[i]`
 *                             will be compared against the column from `left`
 *                             indicated by `left_on[i]`.
 * @param[in] compare_nulls    Controls whether null join-key values should match or not.
 * @param[in] mr               Device memory resource used to allocate the returned table's
 *                             device memory
 *
 * @return                     Result of joining `left` and `right` tables on the columns
 *                             specified by `left_on` and `right_on`.
 */
std::unique_ptr<cudf::table> left_anti_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @param mr    Device memory resource used to allocate the returned table's device memory
 *
 * @return     Result of cross joining `left` and `right` tables
 */
std::unique_ptr<cudf::table> cross_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Hash join that builds hash table in creation and probes results in subsequent `*_join`
 * member functions.
 *
 * This class enables the hash join scheme that builds hash table once, and probes as many times as
 * needed (possibly in parallel).
 */
class hash_join {
 public:
  hash_join() = delete;
  ~hash_join();
  hash_join(hash_join const&) = delete;
  hash_join(hash_join&&)      = delete;
  hash_join& operator=(hash_join const&) = delete;
  hash_join& operator=(hash_join&&) = delete;

  /**
   * @brief Construct a hash join object for subsequent probe calls.
   *
   * @note The `hash_join` object must not outlive the table viewed by `build`, else behavior is
   * undefined.
   *
   * @param build The build table, from which the hash table is built.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  hash_join(cudf::table_view const& build,
            null_equality compare_nulls,
            rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @param probe The probe table, from which the tuples are probed.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param output_size Optional value which allows users to specify the exact output size.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing an inner join between two tables with `build` and `probe`
   * as the the join keys .
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& probe,
             null_equality compare_nulls            = null_equality::EQUAL,
             std::optional<std::size_t> output_size = {},
             rmm::cuda_stream_view stream           = rmm::cuda_stream_default,
             rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a left join between two tables. @see cudf::left_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @param probe The probe table, from which the tuples are probed.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param output_size Optional value which allows users to specify the exact output size.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a left join between two tables with `build` and `probe`
   * as the the join keys .
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& probe,
            null_equality compare_nulls            = null_equality::EQUAL,
            std::optional<std::size_t> output_size = {},
            rmm::cuda_stream_view stream           = rmm::cuda_stream_default,
            rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a full join between two tables. @see cudf::full_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @param probe The probe table, from which the tuples are probed.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param output_size Optional value which allows users to specify the exact output size.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a full join between two tables with `build` and `probe`
   * as the the join keys .
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& probe,
            null_equality compare_nulls            = null_equality::EQUAL,
            std::optional<std::size_t> output_size = {},
            rmm::cuda_stream_view stream           = rmm::cuda_stream_default,
            rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource()) const;

  /**
   * Returns the exact number of matches (rows) when performing an inner join with the specified
   * probe table.
   *
   * @param probe The probe table, from which the tuples are probed.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing an inner join between two tables with
   * `build` and `probe` as the the join keys .
   */
  std::size_t inner_join_size(cudf::table_view const& probe,
                              null_equality compare_nulls  = null_equality::EQUAL,
                              rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * Returns the exact number of matches (rows) when performing a left join with the specified probe
   * table.
   *
   * @param probe The probe table, from which the tuples are probed.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing a left join between two tables with `build`
   * and `probe` as the the join keys .
   */
  std::size_t left_join_size(cudf::table_view const& probe,
                             null_equality compare_nulls  = null_equality::EQUAL,
                             rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * Returns the exact number of matches (rows) when performing a full join with the specified probe
   * table.
   *
   * @param probe The probe table, from which the tuples are probed.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the intermediate table and columns' device
   * memory.
   *
   * @return The exact number of output when performing a full join between two tables with `build`
   * and `probe` as the the join keys .
   */
  std::size_t full_join_size(
    cudf::table_view const& probe,
    null_equality compare_nulls         = null_equality::EQUAL,
    rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

 private:
  struct hash_join_impl;
  const std::unique_ptr<const hash_join_impl> impl;
};

/**
 * @brief Performs a join predicated on an expression.
 *
 * The predicate join returns rows from each table where the expression evaluates to true.
 *
 * @code{.pseudo}
 * Left a: {0, 1, 2}
 * Right b: {3, 4, 5}
 * Result: { a: {0, 0, 0, 1, 1, 1, 2, 2, 2}, b: {3, 4, 5, 3, 4, 5, 3, 4, 5} }
 * @endcode

 * @throw cudf::logic_error if the number of columns in either `left` or `right` table is 0
 *
 * @param left           The left table
 * @param right          The right table
 * @param binary_pred    The expression on which the join is conditioned.
 * @param mr             Device memory resource used to allocate the returned table's device memory
 *
 * @return     Result of the predicated join of the `left` and `right` tables
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_inner_join(
  table_view left,
  table_view right,
  ast::expression binary_pred,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_left_join(table_view left,
                      table_view right,
                      ast::expression binary_pred,
                      rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
/** @} */  // end of group
}  // namespace cudf
