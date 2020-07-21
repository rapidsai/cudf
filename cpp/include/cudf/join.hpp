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

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
/**
 * @addtogroup column_join
 * @{
 */

/**
 * @brief Performs an inner join on the specified columns of two
 * tables (`left`, `right`)
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * @code{.pseudo}
 *          Left a: {0, 1, 2}
 *          Right b: {1, 2, 3}, a: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {1}
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 *          Left a: {0, 1, 2}
 *          Right b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {0}
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 * @endcode
 *
 * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
 * (L, R) if L does not exist in `left_on` or R does not exist in `right_on`.
 * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
 * (L, R) such that the location of `L` within `left_on` is not equal to
 * location of R within `right_on`
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
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left` and `right`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.  For each of these pairs (L, R), L
 * should exist in `left_on` and R should exist in `right_on`.
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<cudf::table> inner_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Performs a left join (also known as left outer join) on the
 * specified columns of two tables (`left`, `right`)
 *
 * Left Join returns all the rows from the left table and those rows from the
 * right table that match on the joined columns.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @code{.pseudo}
 *          Left a: {0, 1, 2}
 *          Right b: {1, 2, 3}, a: {1 ,2 ,5}
 *          left_on: {0}
 *          right_on: {1}
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
 *
 *          Left a: {0, 1, 2}
 *          Right b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {0}
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2}, c: {NULL, 1, 2} }
 * @endcode
 *
 * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
 * (L, R) if L does not exist in `left_on` or R does not exist in `right_on`.
 * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
 * (L, R) such that the location of `L` within `left_on` is not equal to
 * location of R within `right_on`
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
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left` and `right`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.  For each of these pairs (L, R), L
 * should exist in `left_on` and R should exist in `right_on`.
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<cudf::table> left_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

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
 *          Left a: {0, 1, 2}
 *          Right b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {1}
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 *          Left a: {0, 1, 2}
 *          Right b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {0}
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 * @endcode
 *
 * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
 * (L, R) if L does not exist in `left_on` or R does not exist in `right_on`.
 * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
 * (L, R) such that the location of `L` within `left_on` is not equal to
 * location of R within `right_on`
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
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left` and `right`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.  For each of these pairs (L, R), L
 * should exist in `left_on` and R should exist in `right_on`.
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<cudf::table> full_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
/**
 * @brief Performs a left semi join on the specified columns of two
 * tables (`left`, `right`)
 *
 * A left semi join only returns data from the left table, and only
 * returns rows that exist in the right table.
 *
 * @code{.pseudo}
 *          TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, a: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {1}
 *          return_columns: { 0 }
 * Result: { a: {1, 2} }
 *
 *          TableA a: {0, 1, 2}, c: {1, 2, 5}
 *          TableB b: {1, 2, 3}
 *          left_on: {0}
 *          right_on: {0}
 *          return_columns: { 1 }
 * Result: { c: {1, 2} }
 * @endcode
 *
 * @throw cudf::logic_error if the number of columns in either `left` or `right` table is 0
 * @throw cudf::logic_error if the number of returned columns is 0
 * @throw cudf::logic_error if the number of elements in `left_on` and `right_on` are not equal
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
 * @param[in] return_columns   A vector of column indices from `left` to
 *                             include in the returned table.
 * @param[in] compare_nulls    Controls whether null join-key values should match or not.
 * @param[in] mr               Device memory resource used to allocate the returned table's
 *                             device memory
 *
 * @return                     Result of joining `left` and `right` tables on the columns
 *                             specified by `left_on` and `right_on`. The resulting table
 *                             will contain `return_columns` from `left` that match in right.
 */
std::unique_ptr<cudf::table> left_semi_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<cudf::size_type> const& return_columns,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Performs a left anti join on the specified columns of two
 * tables (`left`, `right`)
 *
 * A left anti join only returns data from the left table, and only
 * returns rows that do not exist in the right table.
 *
 * @code{.pseudo}
 *          TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, a: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {1}
 *          return_columns: { 0 }
 * Result: { a: {0} }
 *
 *          TableA a: {0, 1, 2}, c: {1, 2, 5}
 *          TableB b: {1, 2, 3}
 *          left_on: {0}
 *          right_on: {0}
 *          return_columns: { 1 }
 * Result: { c: {1} }
 * @endcode
 *
 * @throw cudf::logic_error if the number of columns in either `left` or `right` table is 0
 * @throw cudf::logic_error if the number of returned columns is 0
 * @throw cudf::logic_error if the number of elements in `left_on` and `right_on` are not equal
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
 * @param[in] return_columns   A vector of column indices from `left` to
 *                             include in the returned table.
 * @param[in] compare_nulls    Controls whether null join-key values should match or not.
 * @param[in] mr               Device memory resource used to allocate the returned table's
 *                             device memory
 *
 * @return                     Result of joining `left` and `right` tables on the columns
 *                             specified by `left_on` and `right_on`. The resulting table
 *                             will contain `return_columns` from `left` that match in right.
 */
std::unique_ptr<cudf::table> left_anti_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<cudf::size_type> const& return_columns,
  null_equality compare_nulls         = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Performs a cross join on two tables (`left`, `right`)
 *
 * The cross join returns the cartesian product of rows from each table.
 *
 * @note Warning: This function can easily cause out-of-memory errors. The size of the output is
 * equal to `left.num_rows() * right.num_rows()`. Use with caution.
 *
 * @code{.pseudo}
 *          Left a: {0, 1, 2}
 *          Right b: {3, 4, 5}
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
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of group
}  // namespace cudf
