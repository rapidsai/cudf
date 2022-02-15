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
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace structs::detail::experimental {

std::tuple<structs::detail::flattened_table, std::vector<int>> verticalize_nested_columns(
  table_view input,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence)
{
  // auto [table, null_masks] = superimpose_parent_nulls(input);

  auto table = input;
  std::vector<column_view> verticalized_columns;
  std::vector<order> new_column_order;
  std::vector<null_order> new_null_precedence;
  std::vector<int> verticalized_col_depths;
  for (size_type col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
    auto const& col = table.column(col_idx);
    if (is_nested(col.type())) {
      // convert and insert
      std::vector<column_view> r_verticalized_columns;
      std::vector<int> r_verticalized_col_depths;
      std::vector<column_view> flattened;
      std::vector<int> depths;
      // TODO: Here I added a bogus leaf column at the beginning to help in the while loop below.
      //       Refactor the while loop so that it can handle the last case.
      flattened.push_back(make_empty_column(type_id::INT32)->view());
      std::function<void(column_view const&, int)> recursive_child = [&](column_view const& c,
                                                                         int depth) {
        flattened.push_back(c);
        depths.push_back(depth);
        for (int child_idx = 0; child_idx < c.num_children(); ++child_idx) {
          recursive_child(c.child(child_idx), depth + 1);
        }
      };
      recursive_child(col, 0);
      int curr_col_idx     = flattened.size() - 1;
      column_view curr_col = flattened[curr_col_idx];
      while (curr_col_idx > 0) {
        auto const& prev_col = flattened[curr_col_idx - 1];
        if (not is_nested(prev_col.type())) {
          // We hit a column that's a leaf so seal this hierarchy
          r_verticalized_columns.push_back(curr_col);
          r_verticalized_col_depths.push_back(depths[curr_col_idx - 1]);
          curr_col = prev_col;
        } else {
          curr_col = column_view(prev_col.type(),
                                 prev_col.size(),
                                 nullptr,
                                 prev_col.null_mask(),
                                 UNKNOWN_NULL_COUNT,
                                 prev_col.offset(),
                                 {curr_col});
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
      verticalized_columns.push_back(col);
    }
  }
  return std::make_tuple(
    structs::detail::flattened_table(
      table_view(verticalized_columns), new_column_order, new_null_precedence, {}, {}),
    std::move(verticalized_col_depths));
}

}  // namespace structs::detail::experimental
}  // namespace cudf
