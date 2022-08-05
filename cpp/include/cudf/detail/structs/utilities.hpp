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

#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace structs {
namespace detail {

enum class column_nullability {
  MATCH_INCOMING,  ///< generate a null column if the incoming column has nulls
  FORCE            ///< always generate a null column
};

/**
 * @brief Flatten the children of the input columns into a vector where the i'th element
 * is a vector of column_views representing the i'th child from each input column_view.
 *
 * @code{.pseudo}
 * s1 = [ col0 : {0, 1}
 *        col1 : {2, 3, 4, 5, 6}
 *        col2 : {"abc", "def", "ghi"} ]
 *
 * s2 = [ col0 : {7, 8}
 *        col1 : {-4, -5, -6}
 *        col2 : {"uvw", "xyz"} ]
 *
 * e = extract_ordered_struct_children({s1, s2})
 *
 * e is now [ {{0, 1}, {7, 8}}
 *            {{2, 3, 4, 5, 6}, {-4, -5, -6}}
 *            {{"abc", "def", "ghi"}, {"uvw", "xyz"} ]
 * @endcode
 *
 * @param columns Vector of structs columns to extract from.
 * @return New column with concatenated results.
 */
std::vector<std::vector<column_view>> extract_ordered_struct_children(
  host_span<column_view const> struct_cols);

/**
 * @brief Check whether the specified column is of type LIST, or any LISTs in its descendent
 * columns.
 * @param col column to check for lists.
 * @return true if the column or any of it's children is a list, false otherwise.
 */
bool is_or_has_nested_lists(cudf::column_view const& col);

/**
 * @brief Result of `flatten_nested_columns()`, where all `STRUCT` columns are replaced with
 * their non-nested member columns, and `BOOL8` columns for their null masks.
 *
 * `flatten_nested_columns()` produces a "flattened" table_view with all `STRUCT` columns
 * replaced with their child column_views, preceded by their null masks.
 * All newly allocated columns and device_buffers that back the returned table_view
 * are also encapsulated in `flatten_result`.
 *
 * Objects of `flatten_result` need to kept alive while its table_view is accessed.
 */
class flattened_table {
 public:
  /**
   * @brief Constructor, to be used from `flatten_nested_columns()`.
   *
   * @param flattened_columns_ table_view resulting from `flatten_nested_columns()`
   * @param orders_ Per-column ordering of the table_view
   * @param null_orders_ Per-column null_order of the table_view
   * @param columns_ Newly allocated columns to back the table_view
   * @param null_masks_ Newly allocated null masks to back the table_view
   */
  flattened_table(table_view const& flattened_columns_,
                  std::vector<order> const& orders_,
                  std::vector<null_order> const& null_orders_,
                  std::vector<std::unique_ptr<column>>&& columns_,
                  std::vector<rmm::device_buffer>&& null_masks_)
    : _flattened_columns{flattened_columns_},
      _orders{orders_},
      _null_orders{null_orders_},
      _columns{std::move(columns_)},
      _superimposed_nullmasks{std::move(null_masks_)}
  {
  }

  flattened_table() = default;

  /**
   * @brief Getter for the flattened columns, as a `table_view`.
   */
  [[nodiscard]] table_view flattened_columns() const { return _flattened_columns; }

  /**
   * @brief Getter for the cudf::order of the table_view's columns.
   */
  [[nodiscard]] std::vector<order> orders() const { return _orders; }

  /**
   * @brief Getter for the cudf::null_order of the table_view's columns.
   */
  [[nodiscard]] std::vector<null_order> null_orders() const { return _null_orders; }

  /**
   * @brief Conversion to `table_view`, to fetch flattened columns.
   */
  operator table_view() const { return flattened_columns(); }

 private:
  table_view _flattened_columns;
  std::vector<order> _orders;
  std::vector<null_order> _null_orders;
  std::vector<std::unique_ptr<column>> _columns;
  std::vector<rmm::device_buffer> _superimposed_nullmasks;
};

/**
 * @brief Flatten table with struct columns to table with constituent columns of struct columns.
 *
 * If a table does not have struct columns, same input arguments are returned.
 *
 * @param input input table to be flattened
 * @param column_order column order for input table
 * @param null_precedence null order for input table
 * @param nullability force output to have nullability columns even if input columns
 * are all valid
 * @return `flatten_result` with flattened table, flattened column order, flattened null precedence,
 * alongside the supporting columns and device_buffers for the flattened table.
 */
flattened_table flatten_nested_columns(
  table_view const& input,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  column_nullability nullability = column_nullability::MATCH_INCOMING);

/**
 * @brief Unflatten columns flattened as by `flatten_nested_columns()`,
 *        based on the provided `blueprint`.
 *
 * cudf::flatten_nested_columns() executes depth first, and serializes the struct null vector
 * before the child/member columns.
 * E.g. STRUCT_1< STRUCT_2< A, B >, C > is flattened to:
 *      1. Null Vector for STRUCT_1
 *      2. Null Vector for STRUCT_2
 *      3. Member STRUCT_2::A
 *      4. Member STRUCT_2::B
 *      5. Member STRUCT_1::C
 *
 * `unflatten_nested_columns()` reconstructs nested columns from flattened input that follows
 * the convention above.
 *
 * Note: This function requires a null-mask vector for each STRUCT column, including for nested
 * STRUCT members.
 *
 * @param flattened "Flattened" `table` of input columns, following the conventions in
 * `flatten_nested_columns()`.
 * @param blueprint The exemplar `table_view` with nested columns intact, whose structure defines
 * the nesting of the reconstructed output table.
 * @return std::unique_ptr<cudf::table> Unflattened table (with nested STRUCT columns) reconstructed
 * based on `blueprint`.
 */
std::unique_ptr<cudf::table> unflatten_nested_columns(std::unique_ptr<cudf::table>&& flattened,
                                                      table_view const& blueprint);

/**
 * @brief Push down nulls from a parent mask into a child column, using bitwise AND.
 *
 * This function will recurse through all struct descendants. It is expected that
 * the size of `parent_null_mask` in bits is the same as `child.size()`
 *
 * @param parent_null_mask The mask to be applied to descendants
 * @param parent_null_count Null count in the null mask
 * @param column Column to apply the null mask to.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr     Device memory resource used to allocate new device memory.
 */
void superimpose_parent_nulls(bitmask_type const* parent_null_mask,
                              size_type parent_null_count,
                              column& child,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr);

/**
 * @brief Push down nulls from a parent mask into a child column, using bitwise AND.
 *
 * This function constructs a new column_view instance equivalent to the argument column_view,
 * with possibly new child column_views, all with possibly new null mask values reflecting
 * null rows from the parent column:
 * 1. If the specified column is not STRUCT, the column is returned unmodified, with no new
 *    supporting device_buffer instances.
 * 2. If the column is STRUCT, the null masks of the parent and child are bitwise-ANDed, and a
 *    modified column_view is returned. This applies recursively.
 *
 * @param parent The parent (possibly STRUCT) column whose nulls need to be pushed to its members.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr     Device memory resource used to allocate new device memory.
 * @return A pair of:
 *         1. column_view with nulls pushed down to child columns, as appropriate.
 *         2. Supporting device_buffer instances, for any newly constructed null masks.
 */
std::tuple<cudf::column_view, std::vector<rmm::device_buffer>> superimpose_parent_nulls(
  column_view const& parent,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Push down nulls from a parent mask into child columns, using bitwise AND,
 * for all columns in the specified table.
 *
 * This function constructs a table_view containing a new column_view instance equivalent to
 * every column_view in the specified table. Each column_view might contain possibly new
 * child column_views, all with possibly new null mask values reflecting null rows from the
 * parent column:
 * 1. If the column is not STRUCT, the column is returned unmodified, with no new
 *    supporting device_buffer instances.
 * 2. If the column is STRUCT, the null masks of the parent and child are bitwise-ANDed, and a
 *    modified column_view is returned. This applies recursively.
 *
 * @param table The table_view of (possibly STRUCT) columns whose nulls need to be pushed to its
 * members.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr     Device memory resource used to allocate new device memory.
 * @return A pair of:
 *         1. table_view of columns with nulls pushed down to child columns, as appropriate.
 *         2. Supporting device_buffer instances, for any newly constructed null masks.
 */
std::tuple<cudf::table_view, std::vector<rmm::device_buffer>> superimpose_parent_nulls(
  table_view const& table,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Checks if a column or any of its children is a struct column with structs that are null.
 *
 * This function searches for structs that are null -- differentiating between structs that are null
 * and structs containing null values. Null structs add a column to the result of the flatten column
 * utility and necessitates column_nullability::FORCE when flattening the column for comparison
 * operations.
 *
 * @param col Column to check for null structs
 * @return A boolean indicating if the column is or contains a struct column that contains a null
 * struct.
 */
bool contains_null_structs(column_view const& col);
}  // namespace detail
}  // namespace structs
}  // namespace cudf
