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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <bitset>
#include <iterator>

namespace cudf::structs::detail {

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

namespace {

/**
 * @brief Check whether the specified column is of type `STRUCT`.
 */
bool is_struct(cudf::column_view const& col) { return col.type().id() == type_id::STRUCT; }

}  // namespace

bool is_or_has_nested_lists(cudf::column_view const& col)
{
  auto is_list = [](cudf::column_view const& col) { return col.type().id() == type_id::LIST; };

  return is_list(col) || std::any_of(col.child_begin(), col.child_end(), is_or_has_nested_lists);
}

/**
 * @brief Flattens struct columns to constituent non-struct columns in the input table.
 *
 */
struct table_flattener {
  table_view input;
  // reference variables
  std::vector<order> const& column_order;
  std::vector<null_order> const& null_precedence;
  // output
  std::vector<std::unique_ptr<column>> validity_as_column;
  temporary_nullable_data nullable_data;
  std::vector<column_view> flat_columns;
  std::vector<order> flat_column_order;
  std::vector<null_order> flat_null_precedence;
  column_nullability nullability;

  table_flattener(table_view const& input,
                  std::vector<order> const& column_order,
                  std::vector<null_order> const& null_precedence,
                  column_nullability nullability)
    : column_order(column_order), null_precedence(null_precedence), nullability(nullability)
  {
    superimpose_nulls(input);
    fail_if_unsupported_types(input);
  }

  /**
   * @brief Pushes down nulls from struct columns to children, saves the resulting
   * column to `input`, and generated null masks to `superimposed_nullmasks`.
   */
  void superimpose_nulls(table_view const& input_table)
  {
    auto [table, tmp_nullable_data] = push_down_nulls(input_table, cudf::get_default_stream());
    this->input                     = table;
    this->nullable_data             = std::move(tmp_nullable_data);
  }

  void fail_if_unsupported_types(table_view const& input) const
  {
    auto const has_lists = std::any_of(input.begin(), input.end(), is_or_has_nested_lists);
    CUDF_EXPECTS(not has_lists, "Flattening LIST columns is not supported.");
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

    return flattened_table{table_view{flat_columns},
                           std::move(flat_column_order),
                           std::move(flat_null_precedence),
                           std::move(validity_as_column),
                           std::move(nullable_data)};
  }
};

flattened_table flatten_nested_columns(table_view const& input,
                                       std::vector<order> const& column_order,
                                       std::vector<null_order> const& null_precedence,
                                       column_nullability nullability)
{
  auto const has_struct = std::any_of(input.begin(), input.end(), is_struct);
  if (not has_struct) { return flattened_table{input, column_order, null_precedence, {}, {}}; }

  return table_flattener{input, column_order, null_precedence, nullability}();
}

namespace {

/**
 * @brief Check if the input column needs to be sanitized (i.e., if it may contain non-empty nulls).
 *
 * For performance reason, no non-empty null will actually be checked. Instead, this function
 * performs conservative checking to see if the input column contains any lists or strings column
 * that is nulllable. As such, it may return false positive answer.
 *
 * @param col The input column to check
 */
auto may_need_sanitize(column_view const& col)
{
  auto const string_or_list_type = [](type_id id) {
    return id == type_id::STRING || id == type_id::LIST;
  };
  return (col.nullable() && string_or_list_type(col.type().id())) ||
         std::any_of(col.child_begin(), col.child_end(), [](auto const& child) {
           return may_need_sanitize(child);
         });
}

/**
 * @brief Superimpose the given null_mask into the input column, without any sanitization for
 * non-empty nulls.
 *
 * @copydoc cudf::structs::detail::superimpose_nulls
 */
column superimpose_nulls_no_sanitize(bitmask_type const* null_mask,
                                     size_type null_count,
                                     column&& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  if (input.type().id() == cudf::type_id::EMPTY) {
    // EMPTY columns should not have a null mask,
    // so don't superimpose null mask on empty columns.
    return std::move(input);
  }

  if (!input.nullable()) {
    // Child currently has no null mask. Copy parent's null mask.
    input.set_null_mask(cudf::detail::copy_bitmask(null_mask, 0, input.size(), stream, mr));
    input.set_null_count(null_count);
  } else {
    auto current_mask = input.mutable_view().null_mask();
    std::vector<bitmask_type const*> masks{reinterpret_cast<bitmask_type const*>(null_mask),
                                           reinterpret_cast<bitmask_type const*>(current_mask)};
    std::vector<size_type> begin_bits{0, 0};
    auto const valid_count = cudf::detail::inplace_bitmask_and(
      device_span<bitmask_type>(current_mask, num_bitmask_words(input.size())),
      masks,
      begin_bits,
      input.size(),
      stream);
    auto const null_count = input.size() - valid_count;
    input.set_null_count(null_count);
  }

  // If the input is not a structs column, just let it pass through.
  if (input.type().id() != cudf::type_id::STRUCT) { return std::move(input); }

  // If the input is a structs column, recursively superimpose null mask for all grandchildren.
  auto const num_rows           = input.size();
  auto const current_null_count = input.null_count();
  auto content                  = input.release();

  // Build new children columns.
  // If any child column is strings/lists column, its non-empty nulls will also be purged.
  const auto current_mask = reinterpret_cast<bitmask_type const*>(content.null_mask->data());
  std::for_each(
    content.children.begin(), content.children.end(), [current_mask, stream, mr](auto& child) {
      child = std::make_unique<column>(superimpose_nulls_no_sanitize(
        current_mask, UNKNOWN_NULL_COUNT, std::move(*child), stream, mr));
    });

  // Return a new structs column with newly rebuilt children.
  return std::move(*cudf::make_structs_column(num_rows,
                                              std::move(content.children),
                                              current_null_count,
                                              std::move(*content.null_mask),
                                              stream,
                                              mr));
}

/**
 * @brief Push down nulls from the given input column into its children columns, without any
 * sanitization for non-empty nulls.
 *
 * @copydoc cudf::structs::detail::push_down_nulls
 */
std::pair<column_view, temporary_nullable_data> push_down_nulls_no_sanitize(
  column_view const& input, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  auto ret_temp_data = temporary_nullable_data{};
  if (input.type().id() != type_id::STRUCT) {
    // NOOP for non-STRUCT columns.
    return {input, std::move(ret_temp_data)};
  }

  auto const structs_view = structs_column_view{input};

  // Function to rewrite child null mask.
  auto const child_with_new_mask = [&](auto const& child_idx) {
    auto child = structs_view.get_sliced_child(child_idx);

    // If struct is not nullable, child null mask is retained. NOOP.
    if (not structs_view.nullable()) { return child; }

    auto parent_child_null_masks =
      std::vector<cudf::bitmask_type const*>{structs_view.null_mask(), child.null_mask()};

    auto [new_child_mask, null_count] = [&] {
      if (not child.nullable()) {
        // Adopt parent STRUCT's null mask.
        return std::pair(structs_view.null_mask(), 0);
      }

      // Both STRUCT and child are nullable. AND() for the child's new null mask.
      //
      // Note: ANDing only [offset(), offset()+size()) would not work. The null-mask produced thus
      // would start at offset=0. The column-view attempts to apply its offset() to both the _data
      // and the _null_mask(). It would be better to AND the bits from the beginning, and apply
      // offset() uniformly.
      // Alternatively, one could construct a big enough buffer, and use inplace_bitwise_and.
      auto [new_mask, null_count] = cudf::detail::bitmask_and(parent_child_null_masks,
                                                              std::vector<size_type>{0, 0},
                                                              child.offset() + child.size(),
                                                              stream,
                                                              mr);
      ret_temp_data.new_null_masks.push_back(std::move(new_mask));
      return std::pair(
        reinterpret_cast<bitmask_type const*>(ret_temp_data.new_null_masks.back().data()),
        null_count);
    }();

    return column_view(child.type(),
                       child.size(),
                       child.head(),
                       new_child_mask,
                       null_count,
                       child.offset(),
                       std::vector<column_view>{child.child_begin(), child.child_end()});
  };

  auto const child_begin =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), child_with_new_mask);
  auto const child_end = child_begin + structs_view.num_children();
  auto ret_children    = std::vector<column_view>{};

  std::for_each(child_begin, child_end, [&](auto const& child) {
    auto [processed_child, child_nullable_data] = push_down_nulls_no_sanitize(child, stream, mr);
    ret_children.emplace_back(std::move(processed_child));
    ret_temp_data.emplace_back(std::move(child_nullable_data));
  });

  // Make column view out of newly constructed column_views, and all the validity buffers.

  return std::pair(column_view(input.type(),
                               input.size(),
                               nullptr,
                               input.null_mask(),
                               input.null_count(),  // Alternatively, postpone.
                               input.offset(),
                               ret_children),
                   std::move(ret_temp_data));
}

}  // namespace

void temporary_nullable_data::emplace_back(temporary_nullable_data&& other)
{
  auto const append_fn = [](auto& dst, auto& src) {
    dst.insert(dst.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()));
  };
  append_fn(new_null_masks, other.new_null_masks);
  append_fn(new_columns, other.new_columns);
}

column superimpose_nulls(bitmask_type const* null_mask,
                         size_type null_count,
                         column&& input,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
{
  auto output =
    superimpose_nulls_no_sanitize(null_mask, null_count, std::forward<column>(input), stream, mr);

  if (auto const output_view = output.view(); may_need_sanitize(output_view)) {
    // We can't call `purge_nonempty_nulls` for individual child column(s) that needs to be
    // sanitized. Instead, we have to call it on the top level column.
    // This is to make sure all the columns (top level + all children) have consistent offsets.
    // Otherwise, the sanitized children may have offsets that are different from the others and
    // also different from the parent column, causing data corruption.
    return std::move(*cudf::detail::purge_nonempty_nulls(output_view, stream, mr));
  }

  return output;
}

std::pair<column_view, temporary_nullable_data> push_down_nulls(column_view const& input,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::mr::device_memory_resource* mr)
{
  auto output = push_down_nulls_no_sanitize(input, stream, mr);

  if (auto const output_view = output.first; may_need_sanitize(output_view)) {
    output.second.new_columns.emplace_back(
      cudf::detail::purge_nonempty_nulls(output_view, stream, mr));
    output.first = output.second.new_columns.back()->view();

    // Don't need the temp null mask anymore, as we will create a new column.
    // Note: The new null masks are still needed for `purge_nonempty_nulls` thus this must be called
    // after it.
    output.second.new_null_masks.clear();
  }

  return output;
}

std::pair<table_view, temporary_nullable_data> push_down_nulls(table_view const& table,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::mr::device_memory_resource* mr)
{
  auto superimposed_columns = std::vector<column_view>{};
  auto nullable_data        = temporary_nullable_data{};
  for (auto const& col : table) {
    auto [superimposed_col, col_nullable_data] = push_down_nulls_no_sanitize(col, stream, mr);
    superimposed_columns.emplace_back(std::move(superimposed_col));
    nullable_data.emplace_back(std::move(col_nullable_data));
  }
  return {table_view{superimposed_columns}, std::move(nullable_data)};
}

bool contains_null_structs(column_view const& col)
{
  return (is_struct(col) && col.has_nulls()) ||
         std::any_of(col.child_begin(), col.child_end(), contains_null_structs);
}

}  // namespace cudf::structs::detail
