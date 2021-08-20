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

#include <thrust/iterator/counting_iterator.h>

#include <cudf/detail/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <structs/utilities.hpp>

namespace cudf {
namespace structs {
namespace detail {

/**
 * @copydoc cudf::structs::detail::extract_ordered_struct_children
 */
std::vector<std::vector<column_view>> extract_ordered_struct_children(
  host_span<column_view const> struct_cols)
{
  auto const num_children = struct_cols[0].num_children();
  auto const num_cols     = static_cast<size_type>(struct_cols.size());

  std::vector<std::vector<column_view>> result;
  result.reserve(num_children);

  for (size_type child_index = 0; child_index < num_children; child_index++) {
    std::vector<column_view> children;
    children.reserve(num_cols);
    for (size_type col_index = 0; col_index < num_cols; col_index++) {
      structs_column_view scv(struct_cols[col_index]);

      // all inputs must have the same # of children and they must all be of the
      // same type.
      CUDF_EXPECTS(struct_cols[0].num_children() == scv.num_children(),
                   "Mismatch in number of children during struct concatenate");
      CUDF_EXPECTS(struct_cols[0].child(child_index).type() == scv.child(child_index).type(),
                   "Mismatch in child types during struct concatenate");
      children.push_back(scv.get_sliced_child(child_index));
    }

    result.push_back(std::move(children));
  }

  return result;
}

/**
 * @brief Flattens struct columns to constituent non-struct columns in the input table.
 *
 */
struct flattened_table {
  // reference variables
  table_view const& input;
  std::vector<order> const& column_order;
  std::vector<null_order> const& null_precedence;
  // output
  std::vector<std::unique_ptr<column>> validity_as_column;
  std::vector<column_view> flat_columns;
  std::vector<order> flat_column_order;
  std::vector<null_order> flat_null_precedence;
  column_nullability nullability;

  flattened_table(table_view const& input,
                  std::vector<order> const& column_order,
                  std::vector<null_order> const& null_precedence,
                  column_nullability nullability)
    : input(input),
      column_order(column_order),
      null_precedence(null_precedence),
      nullability(nullability)
  {
  }

  // Convert null_mask to BOOL8 columns and flatten the struct children in order.
  void flatten_struct_column(structs_column_view const& col,
                             order col_order,
                             null_order col_null_order)
  {
    // Even if it is not required to extract the bitmask to a separate column,
    // we should always do that if the structs column has any null element.
    //
    // In addition, we should check for null by calling to `has_nulls()`, not `nullable()`.
    // This is because when comparing structs columns, if one column has bitmask while the other
    // does not (and both columns do not have any null element) then flattening them using
    // `nullable()` will result in tables with different number of columns.
    //
    // Notice that, for comparing structs columns when one column has null while the other
    // doesn't, `nullability` must be passed in with value `column_nullability::FORCE` to make
    // sure the flattening results are tables having the same number of columns.

    if (nullability == column_nullability::FORCE || col.has_nulls()) {
      validity_as_column.push_back(cudf::is_valid(col));
      if (col.has_nulls()) {
        // copy bitmask is needed only if the column has null
        validity_as_column.back()->set_null_mask(copy_bitmask(col));
      }
      flat_columns.push_back(validity_as_column.back()->view());
      if (not column_order.empty()) { flat_column_order.push_back(col_order); }  // doesn't matter.
      if (not null_precedence.empty()) { flat_null_precedence.push_back(col_null_order); }
    }
    for (decltype(col.num_children()) i = 0; i < col.num_children(); ++i) {
      auto const& child = col.get_sliced_child(i);
      if (child.type().id() == type_id::STRUCT) {
        flatten_struct_column(structs_column_view{child}, col_order, col_null_order);
      } else {
        flat_columns.push_back(child);
        if (not column_order.empty()) flat_column_order.push_back(col_order);
        if (not null_precedence.empty()) flat_null_precedence.push_back(col_null_order);
      }
    }
  }
  // Note: possibly expand for flattening list columns too.

  /**
   * @copydoc flattened_table
   *
   * @return tuple with flattened table, flattened column order, flattened null precedence,
   * vector of boolean columns (struct validity).
   */
  auto operator()()
  {
    for (auto i = 0; i < input.num_columns(); ++i) {
      auto const& col = input.column(i);
      if (col.type().id() == type_id::STRUCT) {
        flatten_struct_column(structs_column_view{col},
                              (column_order.empty() ? order() : column_order[i]),
                              (null_precedence.empty() ? null_order() : null_precedence[i]));
      } else {
        flat_columns.push_back(col);
        if (not column_order.empty()) flat_column_order.push_back(column_order[i]);
        if (not null_precedence.empty()) flat_null_precedence.push_back(null_precedence[i]);
      }
    }

    return std::make_tuple(table_view{flat_columns},
                           std::move(flat_column_order),
                           std::move(flat_null_precedence),
                           std::move(validity_as_column));
  }
};

/**
 * @copydoc cudf::detail::flatten_nested_columns
 */
std::tuple<table_view,
           std::vector<order>,
           std::vector<null_order>,
           std::vector<std::unique_ptr<column>>>
flatten_nested_columns(table_view const& input,
                       std::vector<order> const& column_order,
                       std::vector<null_order> const& null_precedence,
                       column_nullability nullability)
{
  std::vector<std::unique_ptr<column>> validity_as_column;
  auto const has_struct = std::any_of(
    input.begin(), input.end(), [](auto const& col) { return col.type().id() == type_id::STRUCT; });
  if (not has_struct)
    return std::make_tuple(input, column_order, null_precedence, std::move(validity_as_column));

  return flattened_table{input, column_order, null_precedence, nullability}();
}

// Helper function to superimpose validity of parent struct
// over the specified member (child) column.
void superimpose_parent_nulls(bitmask_type const* parent_null_mask,
                              size_type parent_null_count,
                              column& child,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  if (!child.nullable()) {
    // Child currently has no null mask. Copy parent's null mask.
    child.set_null_mask(cudf::detail::copy_bitmask(parent_null_mask, 0, child.size(), stream, mr));
    child.set_null_count(parent_null_count);
  } else {
    // Child should have a null mask.
    // `AND` the child's null mask with the parent's.

    auto current_child_mask = child.mutable_view().null_mask();

    std::vector<bitmask_type const*> masks{
      reinterpret_cast<bitmask_type const*>(parent_null_mask),
      reinterpret_cast<bitmask_type const*>(current_child_mask)};
    std::vector<size_type> begin_bits{0, 0};
    cudf::detail::inplace_bitmask_and(
      device_span<bitmask_type>(current_child_mask, num_bitmask_words(child.size())),
      masks,
      begin_bits,
      child.size(),
      stream,
      mr);
    child.set_null_count(UNKNOWN_NULL_COUNT);
  }

  // If the child is also a struct, repeat for all grandchildren.
  if (child.type().id() == cudf::type_id::STRUCT) {
    const auto current_child_mask = child.mutable_view().null_mask();
    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(child.num_children()),
                  [&current_child_mask, &child, stream, mr](auto i) {
                    superimpose_parent_nulls(
                      current_child_mask, UNKNOWN_NULL_COUNT, child.child(i), stream, mr);
                  });
  }
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
