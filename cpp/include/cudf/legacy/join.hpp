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
 * if a sort-based join is requested and either `right_on` or `left_on` contains null values.
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column's indices from `left` to join on.
 * Column `i` from `left_on` will be compared against column `i` of `right_on`.
 * @param[in] right_on The column's indices from `right` to join on.
 * Column `i` from `right_on` will be compared with column `i` of `left_on`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.
 *
 * @param[out] * @returns joined_indices Optional, if not `nullptr`, on return, will
 * contain two non-nullable, `GDF_INT32` columns containing the indices of
 * matching rows between `left_on` and `right_on`. The first column corresponds to
 * rows in `left_on`, and the second to `right_on`. It is the caller's
 * responsibility to free these columns.
 * @param[in] join_context The context to use to control how
 * the join is performed,e.g., sort vs hash based implementation
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
cudf::table inner_join(
  cudf::table const& left,
  cudf::table const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  cudf::table* joined_indices,
  gdf_context* join_context);
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
 * if a sort-based join is requested and either `right_on` or `left_on` contains null values.
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column's indices from `left` to join on.
 * Column `i` from `left_on` will be compared against column `i` of `right_on`.
 * @param[in] right_on The column's indices from `right` to join on.
 * Column `i` from `right_on` will be compared with column `i` of `left_on`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from intersection of respective columns in `left_on` and `right_on`. Else, for every column
 * in `left_on` and `right_on`, an output column will be produced.
 *
 * @param[out] * @returns joined_indices Optional, if not `nullptr`, on return, will
 * contain two non-nullable, `GDF_INT32` columns containing the indices of
 * matching rows between `left_on` and `right_on`. The first column corresponds to
 * rows in `left_on`, and the second to `right_on`. A value of `-1` in the second column
 * indicates that the corresponding row in `left_on` has no match. It is the caller's
 * responsibility to free these columns.
 * @param[in] join_context The context to use to control how
 * the join is performed,e.g., sort vs hash based implementation
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
cudf::table left_join(
  cudf::table const& left,
  cudf::table const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  cudf::table* joined_indices,
  gdf_context* join_context);

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
 * if a sort-based join is requested and either `right_on` or `left_on` contains null values.
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column's indices from `left` to join on.
 * Column `i` from `left_on` will be compared against column `i` of `right_on`.
 * @param[in] right_on The column's indices from `right` to join on.
 * Column `i` from `right_on` will be compared with column `i` of `left_on`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from respective columns in `left_on` and `right_on`. Else, for every column in
 * `left_on` and `right_on`, an output column will be produced.
 *
 * @param[out] * @returns joined_indices Optional, if not `nullptr`, on return, will
 * contain two non-nullable, `GDF_INT32` columns containing the indices of
 * matching rows between `left_on` and `right_on`. The first column corresponds to
 * rows in `left_on`, and the second to `right_on`. A value of `-1` in the second column
 * indicates that the corresponding row in `left_on` has no match. And similarly `-1` in
 * first column indicates that the corresponding row in `right_on` has no match.
 * It is the caller's responsibility to free these columns.
 * @param[in] join_context The context to use to control how
 * the join is performed,e.g., sort vs hash based implementation
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
cudf::table full_join(
  cudf::table const& left,
  cudf::table const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  cudf::table* joined_indices,
  gdf_context* join_context);
}  // namespace cudf
