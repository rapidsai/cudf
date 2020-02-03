/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace join {

/**
 * @brief Type defining join comparison operators
 *
 * Indicates how two elements `a` and `b` compare with one and another.
 */
enum class join_comparison_operator {
  LESS_THAN,             ///< Compare a < b
  LESS_THAN_OR_EQUAL,    ///< Compare a <= b
  EQUAL,                 ///< Compare a == b
  NOT_EQUAL,             ///< Compare a != b
  GREATER_THAN,          ///< Compare a >  b
  GREATER_THAN_OR_EQUAL, ///< Compare a >= b
};

/**
 * @brief Type defining join operations
 */
struct join_operation {
  join_comparison_operator op;                   ///< type of comparison
  cudf::size_type          left_column_idx;      ///< index of left column to compare
  cudf::size_type          right_column_idx;     ///< index of right column to compare
};
  
// NOTES:
//    1) This interface would support inequality joins, although we could first
//       implement only equijoins.
//       *** eventually if we need to support complex and/or/not expressions
//       the primary and secondary join ops could be changed to expression trees
//       (although maybe only the secondary join)
//

//
//  Option 1: namespace by type of join
//
namespace inner_join {

/**
 * @brief  Performs an inner join on the specified columns of two
 * tables (left, right) using a nested loop join (iterate over all
 * possible results and compare them).
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] join_ops           The join operations.  Each join operation identifies a comparison
 *                               operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs an inner join on the specified columns of two
 * tables (left, right) using a sort merge join (sort the primary
 * join columns and use merge logic to identify the intermediate result).
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * Note that join will be done in two steps.  The first step uses primary_join_ops to create an
 * intermediate result identifying pairs of rows in the left and right tables that could be part
 * of the solution.  The second step uses secondary_join_ops to filter that result to determine
 * the final contents of the join output.  This two pass process allows the caller to identify
 * the column pairs that should be used to create that intermediate product - which allows us to
 * avoid sorting all of the columns that are part of the join criteria.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {2}, b: {2}, c: {2}, d: {2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of primary_join_ops or secondary_join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                cudf::table_view const& right,
                                                std::vector<join_operation> const& primary_join_ops,
                                                std::vector<join_operation> const& secondary_join_ops,
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs an inner join on the specified columns of two
 * tables (left, right) using a hash join (build a hash table from the primary
 * join columns and probe the hash table to identify the intermediate result).
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * Note that join will be done in two steps.  The first step uses primary_join_ops to create an
 * intermediate result identifying pairs of rows in the left and right tables that could be part
 * of the solution.  The second step uses secondary_join_ops to filter that result to determine
 * the final contents of the join output.  This two pass process allows the caller to identify
 * the column pairs that should be used to create that intermediate product - which allows us to
 * avoid including all columns that are part of the join criteria in the hash computation.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {2}, b: {2}, c: {2}, d: {2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of primary_join_ops or secondary_join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                          cudf::table_view const& right,
                                          std::vector<join_operation> const& primary_join_ops,
                                          std::vector<join_operation> const& secondary_join_ops,
                                          std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace inner_join

namespace left_join {

/**
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two tables (left, right) using a nested loop
 * join (iterate over all possible results and compare them).
 *
 * Left Join returns all the rows from the left table and those rows from the
 * right table that match on the joined columns.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2}, c: {NULL, 1, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] join_ops           The join operations.  Each join operation identifies a comparison
 *                               operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two tables (left, right) using a sort merge
 * join (sort the primary join columns and use merge logic to identify
 * the intermediate result).
 *
 * Left Join returns all the rows from the left table and those rows from the
 * right table that match on the joined columns.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2}, c: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                cudf::table_view const& right,
                                                std::vector<join_operation> const& primary_join_ops,
                                                std::vector<join_operation> const& secondary_join_ops,
                                                std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two tables (left, right) using a hash join
 * (build a hash table from the primary join columns and probe the hash
 * table to identify the intermediate result).
 *
 * Left Join returns all the rows from the left table and those rows from the
 * right table that match on the joined columns.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2}, c: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                          cudf::table_view const& right,
                                          std::vector<join_operation> const& primary_join_ops,
                                          std::vector<join_operation> const& secondary_join_ops,
                                          std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace left_join

namespace full_join {

/**
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two tables (left, right) using a nested loop join
 * (iterate over all possible results and compare them).
 *
 * Full Join returns the rows that would be returned by a left join and those
 * rows from the right table that do not have a match.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] join_ops           The join operations.  Each join operation identifies a comparison
 *                               operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two tables (left, right) using a sort merge join
 * (sort the primary join columns and use merge logic to identify the
 * intermediate result).
 *
 * Full Join returns the rows that would be returned by a left join and those
 * rows from the right table that do not have a match.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                cudf::table_view const& right,
                                                std::vector<join_operation> const& primary_join_ops,
                                                std::vector<join_operation> const& secondary_join_ops,
                                                std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two tables (left, right) using a hash join
 * (build a hash table from the primary join columns and probe the 
 * hash table to identify the intermeidate result).
 *
 * Full Join returns the rows that would be returned by a left join and those
 * rows from the right table that do not have a match.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                          cudf::table_view const& right,
                                          std::vector<join_operation> const& primary_join_ops,
                                          std::vector<join_operation> const& secondary_join_ops,
                                          std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace full_join

//
// Option 2:  namespace by implementation
//
namespace nested_loop {

/**
 * @brief  Performs an inner join on the specified columns of two
 * tables (left, right) using a nested loop join (iterate over all
 * possible results and compare them).
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] join_ops           The join operations.  Each join operation identifies a comparison
 *                               operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                cudf::table_view const& right,
                                                std::vector<join_operation> const& join_ops,
                                                std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                               cudf::table_view const& right,
                                               std::vector<join_operation> const& join_ops,
                                               std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                               cudf::table_view const& right,
                                               std::vector<join_operation> const& join_ops,
                                               std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace nested_loop

namespace sort_merge {

/**
 * @brief  Performs an inner join on the specified columns of two
 * tables (left, right) using a sort merge join (sort the primary
 * join columns and use merge logic to identify the intermediate result).
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * Note that join will be done in two steps.  The first step uses primary_join_ops to create an
 * intermediate result identifying pairs of rows in the left and right tables that could be part
 * of the solution.  The second step uses secondary_join_ops to filter that result to determine
 * the final contents of the join output.  This two pass process allows the caller to identify
 * the column pairs that should be used to create that intermediate product - which allows us to
 * avoid sorting all of the columns that are part of the join criteria.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {2}, b: {2}, c: {2}, d: {2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of primary_join_ops or secondary_join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                cudf::table_view const& right,
                                                std::vector<join_operation> const& primary_join_ops,
                                                std::vector<join_operation> const& secondary_join_ops,
                                                std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two tables (left, right) using a sort merge
 * join (sort the primary join columns and use merge logic to identify
 * the intermediate result).
 *
 * Left Join returns all the rows from the left table and those rows from the
 * right table that match on the joined columns.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2}, c: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                               cudf::table_view const& right,
                                               std::vector<join_operation> const& primary_join_ops,
                                               std::vector<join_operation> const& secondary_join_ops,
                                               std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two tables (left, right) using a sort merge join
 * (sort the primary join columns and use merge logic to identify the
 * intermediate result).
 *
 * Full Join returns the rows that would be returned by a left join and those
 * rows from the right table that do not have a match.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                               cudf::table_view const& right,
                                               std::vector<join_operation> const& primary_join_ops,
                                               std::vector<join_operation> const& secondary_join_ops,
                                               std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace sort_merge
  
namespace hash {

/**
 * @brief  Performs an inner join on the specified columns of two
 * tables (left, right) using a hash join (build a hash table from the primary
 * join columns and probe the hash table to identify the intermediate result).
 *
 * Inner Join returns rows from both tables as long as the values
 * in the columns being joined on match.
 *
 * Note that join will be done in two steps.  The first step uses primary_join_ops to create an
 * intermediate result identifying pairs of rows in the left and right tables that could be part
 * of the solution.  The second step uses secondary_join_ops to filter that result to determine
 * the final contents of the join output.  This two pass process allows the caller to identify
 * the column pairs that should be used to create that intermediate product - which allows us to
 * avoid including all columns that are part of the join criteria in the hash computation.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {2}, b: {2}, c: {2}, d: {2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of primary_join_ops or secondary_join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                cudf::table_view const& right,
                                                std::vector<join_operation> const& primary_join_ops,
                                                std::vector<join_operation> const& secondary_join_ops,
                                                std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two tables (left, right) using a hash join
 * (build a hash table from the primary join columns and probe the hash
 * table to identify the intermediate result).
 *
 * Left Join returns all the rows from the left table and those rows from the
 * right table that match on the joined columns.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, a: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2}, c: {NULL, 1, 2} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                               cudf::table_view const& right,
                                               std::vector<join_operation> const& primary_join_ops,
                                               std::vector<join_operation> const& secondary_join_ops,
                                               std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two tables (left, right) using a hash join
 * (build a hash table from the primary join columns and probe the 
 * hash table to identify the intermeidate result).
 *
 * Full Join returns the rows that would be returned by a left join and those
 * rows from the right table that do not have a match.
 * For rows from the right table that do not have a match, the corresponding
 * values in the left columns will be null.
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 1} }
 *          secondary_join_ops: { }
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example left a: {0, 1, 2}, d: {9, 2, 5}
 *          right b: {1, 2, 3}, c: {1, 2, 5}
 *          primary_join_ops: { {join_comparison_operator::EQUAL, 0, 0} }
 *          secondary_join_ops: { {join_comparison_operator::EQUAL, 1, 1} }
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, NULL, 2}, c: {NULL, NULL, 2}, d: {NULL, NULL, 2} }
 *
 * @throws cudf::logic_error if `columns_in_common` contains a pair of indices
 *                           (L, R) such that the type of `L` within `left_on` is not equal to
 *                           type of R within `right_on`
 * @throws cudf::logic_error if number of columns in either `left` or `right`
 *                           table is 0
 * @throws std::out_of_range if element of join_ops references a left or right
 *                           index greater than the number of columns in the corresponding left or right table.
 *
 * @param[in] left               The left table
 * @param[in] right              The right table
 * @param[in] primary_join_ops   The primary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The primary_join_ops
 *                               is used as the sort/merge key for creating an intermediate result.
 * @param[in] secondary_join_ops The secondary join operations.  Each join operation identifies a
 *                               comparison operator and a pair of columns.  The join operations in the
 *                               vector are logically combined with an AND.  The secondary_join_ops
 *                               is used as a filter on the intermediate result to create the final result.
 * @param[in] columns_in_common  A vector of pairs of column indices into `left` and `right`,
 *                               respectively, that are "in common". For "common" columns, only a
 *                               single output column will be produced, which is gathered from `left_on`
 *                               columns.  Columns from left and right which are not identified within
 *                               columns_in_common will also be output.
 * @param mr                     Memory resource used to allocate the returned table and columns
 *
 * @returns                      Result of joining `left` and `right` tables on the columns
 *                               specified by join_ops.  The resulting table will be joined columns of
 *                               `left(common columns)+left(excluding common columns)+right(excluding common columns)`.
 */
std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                               cudf::table_view const& right,
                                               std::vector<join_operation> const& primary_join_ops,
                                               std::vector<join_operation> const& secondary_join_ops,
                                               std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace hash
  
} //namespace join

} //namespace cudf
