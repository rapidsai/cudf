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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace structs::detail::experimental {

struct linked_column_view;

using LinkedColPtr    = std::shared_ptr<linked_column_view>;
using LinkedColVector = std::vector<LinkedColPtr>;

/**
 * @brief column_view with the added member pointer to the parent of this column.
 *
 */
struct linked_column_view : public cudf::column_view {
  // TODO: since it brings its own children, find a way to inherit from column_view_base which has
  //       everything except the children.
  linked_column_view(column_view const& col) : cudf::column_view(col), parent(nullptr)
  {
    for (auto child_it = col.child_begin(); child_it < col.child_end(); ++child_it) {
      children.push_back(std::make_shared<linked_column_view>(this, *child_it));
    }
  }

  linked_column_view(linked_column_view* parent, column_view const& col)
    : cudf::column_view(col), parent(parent)
  {
    for (auto child_it = col.child_begin(); child_it < col.child_end(); ++child_it) {
      children.push_back(std::make_shared<linked_column_view>(this, *child_it));
    }
  }

  linked_column_view* parent;  //!< Pointer to parent of this column. Nullptr if root
  LinkedColVector children;
};

/**
 * @brief Converts all column_views of a table into linked_column_views
 *
 * @param table table of columns to convert
 * @return Vector of converted linked_column_views
 */
LinkedColVector input_table_to_linked_columns(table_view const& table)
{
  LinkedColVector result;
  for (column_view const& col : table) {
    result.emplace_back(std::make_shared<linked_column_view>(col));
  }

  return result;
}

std::tuple<structs::detail::flattened_table, std::vector<int>> verticalize_nested_columns(
  table_view input,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence)
{
  // auto [table, null_masks] = superimpose_parent_nulls(input);
  auto linked_columns = input_table_to_linked_columns(input);

  // auto table = input;
  std::vector<column_view> verticalized_columns;
  std::vector<order> new_column_order;
  std::vector<null_order> new_null_precedence;
  std::vector<int> verticalized_col_depths;
  for (size_t col_idx = 0; col_idx < linked_columns.size(); ++col_idx) {
    auto const& col = linked_columns[col_idx];
    if (is_nested(col->type())) {
      // convert and insert
      std::vector<column_view> r_verticalized_columns;
      std::vector<int> r_verticalized_col_depths;
      std::vector<LinkedColPtr> flattened;
      std::vector<int> depths;
      // TODO: Here I added a bogus leaf column at the beginning to help in the while loop below.
      //       Refactor the while loop so that it can handle the last case.
      flattened.push_back(
        std::make_shared<linked_column_view>(make_empty_column(type_id::INT32)->view()));
      std::function<void(LinkedColPtr, int)> recursive_child = [&](LinkedColPtr c, int depth) {
        flattened.push_back(c);
        depths.push_back(depth);
        if (c->type().id() == type_id::LIST) {
          recursive_child(c->children[lists_column_view::child_column_index], depth + 1);
        } else if (c->type().id() == type_id::STRUCT) {
          for (auto& child : c->children) {
            recursive_child(child, depth + 1);
          }
        }
      };
      recursive_child(col, 0);
      int curr_col_idx     = flattened.size() - 1;
      column_view curr_col = *flattened[curr_col_idx];
      while (curr_col_idx > 0) {
        auto const& prev_col = flattened[curr_col_idx - 1];
        if (not is_nested(prev_col->type())) {
          // We hit a column that's a leaf so seal this hierarchy
          // But first, traverse upward and include any list columns in the ancestors
          linked_column_view* parent = flattened[curr_col_idx]->parent;
          while (parent) {
            if (parent->type().id() == type_id::LIST) {
              // Include this parent
              curr_col = column_view(
                parent->type(),
                parent->size(),
                nullptr,  // list has no data of its own
                nullptr,  // If we're going through this then nullmaks already in another branch
                UNKNOWN_NULL_COUNT,
                parent->offset(),
                {parent->child(lists_column_view::offsets_column_index), curr_col});
            }
            parent = parent->parent;
          }
          r_verticalized_columns.push_back(curr_col);
          r_verticalized_col_depths.push_back(depths[curr_col_idx - 1]);
          curr_col = *prev_col;
        } else {
          auto children =
            (prev_col->type().id() == type_id::LIST)
              ? std::vector<column_view>{prev_col->child(lists_column_view::offsets_column_index),
                                         curr_col}
              : std::vector<column_view>{curr_col};
          curr_col = column_view(prev_col->type(),
                                 prev_col->size(),
                                 nullptr,
                                 prev_col->null_mask(),
                                 UNKNOWN_NULL_COUNT,
                                 prev_col->offset(),
                                 children);
        }
        --curr_col_idx;
      }
      verticalized_columns.insert(
        verticalized_columns.end(), r_verticalized_columns.rbegin(), r_verticalized_columns.rend());
      verticalized_col_depths.insert(verticalized_col_depths.end(),
                                     r_verticalized_col_depths.rbegin(),
                                     r_verticalized_col_depths.rend());
      if (not column_order.empty()) {
        new_column_order.insert(
          new_column_order.end(), r_verticalized_columns.size(), column_order[col_idx]);
      }
      if (not null_precedence.empty()) {
        new_null_precedence.insert(
          new_null_precedence.end(), r_verticalized_columns.size(), null_precedence[col_idx]);
      }
    } else {
      verticalized_columns.push_back(*col);
    }
  }
  return std::make_tuple(
    structs::detail::flattened_table(
      table_view(verticalized_columns), new_column_order, new_null_precedence, {}, {}),
    std::move(verticalized_col_depths));
}

}  // namespace structs::detail::experimental
}  // namespace cudf
