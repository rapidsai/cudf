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
  std::vector<rmm::device_buffer> superimposed_nullmasks;
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
    auto [table, null_masks]     = superimpose_parent_nulls(input_table);
    this->input                  = table;
    this->superimposed_nullmasks = std::move(null_masks);
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
                           std::move(superimposed_nullmasks)};
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
using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using column_index_t    = typename vector_of_columns::size_type;

// Forward declaration, to enable recursion via `unflattener`.
std::unique_ptr<cudf::column> unflatten_struct(vector_of_columns& flattened,
                                               column_index_t& current_index,
                                               cudf::column_view const& blueprint);

/**
 * @brief Helper functor to reconstruct STRUCT columns from its flattened member columns.
 *
 */
class unflattener {
 public:
  unflattener(vector_of_columns& flattened_, column_index_t& current_index_)
    : flattened{flattened_}, current_index{current_index_}
  {
  }

  auto operator()(column_view const& blueprint)
  {
    return is_struct(blueprint) ? unflatten_struct(flattened, current_index, blueprint)
                                : std::move(flattened[current_index++]);
  }

 private:
  vector_of_columns& flattened;
  column_index_t& current_index;

};  // class unflattener;

std::unique_ptr<cudf::column> unflatten_struct(vector_of_columns& flattened,
                                               column_index_t& current_index,
                                               cudf::column_view const& blueprint)
{
  // "Consume" columns from `flattened`, starting at `current_index`,
  // based on the provided `blueprint` struct col. Recurse for struct children.
  CUDF_EXPECTS(blueprint.type().id() == type_id::STRUCT,
               "Expected blueprint column to be a STRUCT column.");

  CUDF_EXPECTS(current_index < flattened.size(), "STRUCT column can't have 0 children.");

  auto const num_rows = flattened[current_index]->size();

  // cudf::flatten_nested_columns() executes depth first, and serializes the struct null vector
  // before the child/member columns.
  // E.g. STRUCT_1< STRUCT_2< A, B >, C > is flattened to:
  //      1. Null Vector for STRUCT_1
  //      2. Null Vector for STRUCT_2
  //      3. Member STRUCT_2::A
  //      4. Member STRUCT_2::B
  //      5. Member STRUCT_1::C
  //
  // Extract null-vector *before* child columns are constructed.
  auto struct_null_column_contents = flattened[current_index++]->release();
  auto unflattening_iter =
    thrust::make_transform_iterator(blueprint.child_begin(), unflattener{flattened, current_index});

  return cudf::make_structs_column(
    num_rows,
    vector_of_columns{unflattening_iter, unflattening_iter + blueprint.num_children()},
    UNKNOWN_NULL_COUNT,  // Do count?
    std::move(*struct_null_column_contents.null_mask));
}
}  // namespace

std::unique_ptr<cudf::table> unflatten_nested_columns(std::unique_ptr<cudf::table>&& flattened,
                                                      table_view const& blueprint)
{
  // Bail, if LISTs are present.
  auto const has_lists = std::any_of(blueprint.begin(), blueprint.end(), is_or_has_nested_lists);
  CUDF_EXPECTS(not has_lists, "Unflattening LIST columns is not supported.");

  // If there are no STRUCTs, unflattening is a NOOP.
  auto const has_structs = std::any_of(blueprint.begin(), blueprint.end(), is_struct);
  if (not has_structs) {
    return std::move(flattened);  // Unchanged.
  }

  // There be struct columns.
  // Note: Requires null vectors for all struct input columns.
  auto flattened_columns = flattened->release();
  auto current_idx       = column_index_t{0};

  auto unflattening_iter =
    thrust::make_transform_iterator(blueprint.begin(), unflattener{flattened_columns, current_idx});

  return std::make_unique<cudf::table>(
    vector_of_columns{unflattening_iter, unflattening_iter + blueprint.num_columns()});
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
    auto const valid_count = cudf::detail::inplace_bitmask_and(
      device_span<bitmask_type>(current_child_mask, num_bitmask_words(child.size())),
      masks,
      begin_bits,
      child.size(),
      stream,
      mr);
    auto const null_count = child.size() - valid_count;
    child.set_null_count(null_count);
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

std::tuple<cudf::column_view, std::vector<rmm::device_buffer>> superimpose_parent_nulls(
  column_view const& parent, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  if (parent.type().id() != type_id::STRUCT) {
    // NOOP for non-STRUCT columns.
    return std::make_tuple(parent, std::vector<rmm::device_buffer>{});
  }

  auto structs_column = structs_column_view{parent};

  auto ret_validity_buffers = std::vector<rmm::device_buffer>{};

  // Function to rewrite child null mask.
  auto rewrite_child_mask = [&](auto const& child_idx) {
    auto child = structs_column.get_sliced_child(child_idx);

    // If struct is not nullable, child null mask is retained. NOOP.
    if (not structs_column.nullable()) { return child; }

    auto parent_child_null_masks =
      std::vector<cudf::bitmask_type const*>{structs_column.null_mask(), child.null_mask()};

    auto [new_child_mask, null_count] = [&] {
      if (not child.nullable()) {
        // Adopt parent STRUCT's null mask.
        return std::pair(structs_column.null_mask(), 0);
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
      ret_validity_buffers.push_back(std::move(new_mask));
      return std::pair(reinterpret_cast<bitmask_type const*>(ret_validity_buffers.back().data()),
                       null_count);
    }();

    return cudf::column_view(
      child.type(),
      child.size(),
      child.head(),
      new_child_mask,
      null_count,
      child.offset(),
      std::vector<cudf::column_view>{child.child_begin(), child.child_end()});
  };

  auto child_begin =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), rewrite_child_mask);
  auto child_end = child_begin + structs_column.num_children();

  auto ret_children = std::vector<cudf::column_view>{};
  std::for_each(child_begin, child_end, [&](auto const& child) {
    auto [processed_child, backing_buffers] = superimpose_parent_nulls(child, stream, mr);
    ret_children.push_back(processed_child);
    ret_validity_buffers.insert(ret_validity_buffers.end(),
                                std::make_move_iterator(backing_buffers.begin()),
                                std::make_move_iterator(backing_buffers.end()));
  });

  // Make column view out of newly constructed column_views, and all the validity buffers.

  return std::make_tuple(column_view(parent.type(),
                                     parent.size(),
                                     nullptr,
                                     parent.null_mask(),
                                     parent.null_count(),  // Alternatively, postpone.
                                     parent.offset(),
                                     ret_children),
                         std::move(ret_validity_buffers));
}

std::tuple<cudf::table_view, std::vector<rmm::device_buffer>> superimpose_parent_nulls(
  table_view const& table, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  auto superimposed_columns   = std::vector<column_view>{};
  auto superimposed_nullmasks = std::vector<rmm::device_buffer>{};
  for (auto col : table) {
    auto [superimposed_col, null_masks] = superimpose_parent_nulls(col, stream, mr);
    superimposed_columns.push_back(superimposed_col);
    superimposed_nullmasks.insert(superimposed_nullmasks.begin(),
                                  std::make_move_iterator(null_masks.begin()),
                                  std::make_move_iterator(null_masks.end()));
  }
  return {table_view{superimposed_columns}, std::move(superimposed_nullmasks)};
}

bool contains_null_structs(column_view const& col)
{
  return (is_struct(col) && col.has_nulls()) ||
         std::any_of(col.child_begin(), col.child_end(), contains_null_structs);
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
