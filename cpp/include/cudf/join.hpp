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
// joins

/** 
 * @brief  Performs an inner join on the specified columns of two
 * tables (left, right)
 *
 * @example TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, a: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {1}
 *          columns_in_common: { {0, 1} }
 * Result: { a: {1, 2}, b: {1, 2} }
 *
 * @example TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {0}
 *          columns_in_common: { }
 * Result: { a: {1, 2}, b: {1, 2}, c: {1, 2} }
 *
 * @throws cudf::logic_error 
 * if either table is empty or if number of rows in either table exceeds INT_MAX
 * if number of elements in `right_on` and `left_on` are not equal
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column's indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column 
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column's indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column 
 * from `left` indicated by `left_on[i]`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<cudf::experimental::table> inner_join(
                         cudf::table_view const& left,
                         cudf::table_view const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudaStream_t stream=0,
                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
/** 
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two tables (left, right)
 *
 * @example TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, a: {1 ,2 ,5}
 *          left_on: {0}
 *          right_on: {1}
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
 *
 * @example TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {0}
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2}, b: {NULL, 1, 2}, c: {NULL, 1, 2} }
 *
 * @throws cudf::logic_error
 * if either table is empty or if number of rows in either table exceeds INT_MAX
 * if number of elements in `right_on` and `left_on` are not equal
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column's indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column 
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column's indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column 
 * from `left` indicated by `left_on[i]`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<cudf::experimental::table> left_join(
                         cudf::table_view const& left,
                         cudf::table_view const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudaStream_t stream=0,
                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** 
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two tables (left, right)
 * 
 * @example TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {1}
 *          columns_in_common: { {0, 1} }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @example TableA a: {0, 1, 2}
 *          TableB b: {1, 2, 3}, c: {1, 2, 5}
 *          left_on: {0}
 *          right_on: {0}
 *          columns_in_common: { }
 * Result: { a: {0, 1, 2, NULL}, b: {NULL, 1, 2, 3}, c: {NULL, 1, 2, 5} }
 *
 * @throws cudf::logic_error
 * if either table is empty or if number of rows in either table exceeds INT_MAX
 * if number of elements in `right_on` and `left_on` are not equal
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column's indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column 
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column's indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column 
 * from `left` indicated by `left_on[i]`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from respective columns in `left_on` and `right_on`. Else, for every column in
 * `left_on` and `right_on`, an output column will be produced.
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
std::unique_ptr<cudf::experimental::table> full_join(
                         cudf::table_view const& left,
                         cudf::table_view const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudaStream_t stream=0,
                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /** 
   * @brief  Performs a left semi join on the specified columns of two 
   * tables (left, right)
   *
   * A left semi join only returns data from the left table, and only 
   * returns rows that exist in the right table.
   *
   * @example TableA a: {0, 1, 2}
   *          TableB b: {1, 2, 3}, a: {1, 2, 5}
   *          left_on: {0}
   *          right_on: {1}
   *          return_columns: { 0 }
   * Result: { a: {1, 2} }
   *
   * @example TableA a: {0, 1, 2}, c: {1, 2, 5}
   *          TableB b: {1, 2, 3}
   *          left_on: {0}
   *          right_on: {0}
   *          return_columns: { 1 }
   * Result: { c: {1, 2} }
   *
   * @throws cudf::logic_error if either table is empty
   * @throws cudf::logic_error if number of returned columns is 0
   * @throws cudf::logic_error if number of elements in `right_on` and `left_on` are not equal
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
   * @param[in] mr               Device memory resource to use for device memory allocation
   *
   * @returns                    Result of joining `left` and `right` tables on the columns
   *                             specified by `left_on` and `right_on`. The resulting table
   *                             will contain `return_columns` from `left` that match in right.
   */
  std::unique_ptr<cudf::experimental::table> left_semi_join(cudf::table_view const& left,
                                                            cudf::table_view const& right,
                                                            std::vector<cudf::size_type> const& left_on,
                                                            std::vector<cudf::size_type> const& right_on,
                                                            std::vector<cudf::size_type> const& return_columns,
                                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /** 
   * @brief  Performs a left anti join on the specified columns of two 
   * tables (left, right)
   *
   * A left anti join only returns data from the left table, and only 
   * returns rows that do not exist in the right table.
   *
   * @example TableA a: {0, 1, 2}
   *          TableB b: {1, 2, 3}, a: {1, 2, 5}
   *          left_on: {0}
   *          right_on: {1}
   *          return_columns: { 0 }
   * Result: { a: {0} }
   *
   * @example TableA a: {0, 1, 2}, c: {1, 2, 5}
   *          TableB b: {1, 2, 3}
   *          left_on: {0}
   *          right_on: {0}
   *          return_columns: { 1 }
   * Result: { c: {1} }
   *
   * @throws cudf::logic_error if either table is empty
   * @throws cudf::logic_error if number of returned columns is 0
   * @throws cudf::logic_error if number of elements in `right_on` and `left_on` are not equal
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
   * @param[in] mr               Device memory resource to use for device memory allocation
   *
   * @returns                    Result of joining `left` and `right` tables on the columns
   *                             specified by `left_on` and `right_on`. The resulting table
   *                             will contain `return_columns` from `left` that match in right.
   */
  std::unique_ptr<cudf::experimental::table> left_anti_join(cudf::table_view const& left,
                                                            cudf::table_view const& right,
                                                            std::vector<cudf::size_type> const& left_on,
                                                            std::vector<cudf::size_type> const& right_on,
                                                            std::vector<cudf::size_type> const& return_columns,
                                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**
   * @brief Type defining join comparison operators
   *
   * Indicates how two elements `a` and `b` compare with one and another.
   */
  enum class join_comparison_operator {
    LESS_THAN,             ///< Compare a < b
    LESS_THAN_OR_EQUAL,    ///< Compare a <= b
    EQUAL,                 ///< Compare a == b
    GREATER_THAN,          ///< Compare a >  b
    GREATER_THAN_OR_EQUAL, ///< Compare a >= b
  };

  /**
   * @brief Type defining non-equi join operations
   */
  struct join_operation {
    join_comparison_operator op;                   ///< type of comparison
    cudf::size_type          left_column_idx;      ///< index of left column to compare
    cudf::size_type          right_column_idx;     ///< index of right column to compare
    bool                     only_left_in_output;  ///< if true, only output the left column
  }
  
  /** 
   * @brief  Performs a left non-equi inner join on two tables (left, right)
   *
   * A left non-equi inner join acts like an inner join, however it allows
   * for comparison conditions beyond equality.
   *
   * Ultimately, this interface could be quite complex.  For the moment it is limited
   * to defining a list of conditions that will be anded together to provide the join
   * criteria.
   *
   * @example TableA a: {0, 1, 2}
   *          TableB b: {1, 2, 3}, a: {1, 2, 5}
   *          join_ops: { {LESS_THAN, 0, 1, true} }
   * Result: { a: {1, 2, 5}, b: {1, 2, 3} }
   *
   * @example TableA a: {0, 1, 2}, c: {1, 2, 5}
   *          TableB b: {1, 2, 3}
   *          join_ops: { {GREATER_THAN, 0, 0, false} }
   * Result: { a: {2}, b: {1}, c: {5} }
   *
   * @throws cudf::logic_error if either table is empty
   *
   * @param[in] left               The left table
   * @param[in] right              The right table
   * @param[in] join_ops           A vector of join operations.  Each element in the vector
   *                               defines a pair of columns and the comparison operator
   *                               that should be applied and a flag indicating whether
   *                               to include just the column from left, or both columns
   *                               in the generated output.
   * @param[in] mr                 Device memory resource to use for device memory allocation
   *
   * @returns                      Result of joining `left` and `right` tables on the columns
   *                               specified by join_ops.  The resulting table will contain
   *                               all columns of `left` and columns of `right` except for
   *                               columns from join_ops that are marked to not be included.
   */
  std::unique_ptr<cudf::experimental::table> inner_non_equi_join(cudf::table_view const& left,
                                                                 cudf::table_view const& right,
                                                                 std::vector<join_operation> const& join_ops,
                                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  // TODO:  Think about this.  Should the public API for all joins be this, and have the inner workings call
  //        the existing API for equi join and the new internal mechanism for non-equi join?  Perhaps
  //        I implement this public API toward that purpose and we can decide whether to deprecate the
  //        old interface?
  /** 
   * @brief  Performs a left non-equi join (also known as left non-equi outer join) on two tables (left, right)
   *
   * A left non-equi outer join acts like an outer join, however it allows for comparison
   * conditions beyond equality.
   *
   * @example TableA a: {0, 1, 2}
   *          TableB b: {1, 2, 3}, a: {1 ,2 ,5}
   *          join_ops: { {LESS_THAN, 0, 1, true} }
   * Result: { a: {0, 1, 2}, b: {NULL, 1, 2} }
   *
   * @example TableA a: {0, 1, 2}
   *          TableB b: {1, 2, 3}, c: {1, 2, 5}
   *          join_ops: { {GREATER_THAN, 0, 1, false} }
   * Result: { a: {0, 1, 2}, b: {NULL, NULL, 1}, c: {NULL, NULL, 1} }
   *
   * @throws cudf::logic_error if either table is empty
   *
   * @param[in] left               The left table
   * @param[in] right              The right table
   * @param[in] join_ops           A vector of join operations.  Each element in the vector
   *                               defines a pair of columns and the comparison operator
   *                               that should be applied and a flag indicating whether
   *                               to include just the column from left, or both columns
   *                               in the generated output.
   * @param[in] mr                 Device memory resource to use for device memory allocation
   *
   * @returns                      Result of joining `left` and `right` tables on the columns
   *                               specified by join_ops.  The resulting table will contain
   *                               all columns of `left` and columns of `right` except for
   *                               columns from join_ops that are marked to not be included.
   */
  std::unique_ptr<cudf::experimental::table> left_non_equi_join(cudf::table_view const& left,
                                                                cudf::table_view const& right,
                                                                std::vector<join_operation> const& join_ops,
                                                                std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} //namespace cudf

