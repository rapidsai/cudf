/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace structs {
namespace detail {

enum class column_nullability {
  MATCH_INCOMING,  // generate a null column if the incoming column has nulls
  FORCE            // always generate a null column
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
 * @brief Flatten table with struct columns to table with constituent columns of struct columns.
 *
 * If a table does not have struct columns, same input arguments are returned.
 *
 * @param input input table to be flattened
 * @param column_order column order for input table
 * @param null_precedence null order for input table
 * @param nullability force output to have nullability columns even if input columns
 * are all valid
 * @return tuple with flattened table, flattened column order, flattened null precedence,
 * vector of boolean columns (struct validity).
 */
std::tuple<table_view,
           std::vector<order>,
           std::vector<null_order>,
           std::vector<std::unique_ptr<column>>>
flatten_nested_columns(table_view const& input,
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
 * @brief Pushdown nulls from a parent mask into a child column, using bitwise AND.
 *
 * Rather than modify the argument column, this function constructs new equivalent column_view
 * instances, with new null mask values. This function returns both a (possibly new) column,
 * and the device_buffer instances to support any modified null masks.
 * 1. If the specified column is not STRUCT, the column is returned unmodified, with no new
 *    supporting device_buffer instances.
 * 2. If the column is STRUCT, the null masks of the parent and child are bitwise-ANDed, and a
 *    modified column_view is returned. This applies recursively to support
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
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace structs
}  // namespace cudf
