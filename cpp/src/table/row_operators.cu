/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace experimental {

namespace {

auto struct_lex_verticalize(table_view table,
                            host_span<order const> column_order         = {},
                            host_span<null_order const> null_precedence = {})
{
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
        if (c.type().id() == type_id::STRUCT) {
          for (int child_idx = 0; child_idx < c.num_children(); ++child_idx) {
            auto scol = structs_column_view(c);
            recursive_child(scol.get_sliced_child(child_idx), depth + 1);
          }
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
      verticalized_col_depths.push_back(0);
      if (not column_order.empty()) { new_column_order.push_back(column_order[col_idx]); }
      if (not null_precedence.empty()) { new_null_precedence.push_back(null_precedence[col_idx]); }
    }
  }
  return std::make_tuple(table_view(verticalized_columns),
                         std::move(new_column_order),
                         std::move(new_null_precedence),
                         std::move(verticalized_col_depths));
}

void check_lex_compatibility(table_view const& input)
{
  // Basically check if there's any LIST hiding anywhere in the table
  std::function<void(column_view const&)> check_column = [&](column_view const& c) {
    CUDF_EXPECTS(c.type().id() != type_id::LIST,
                 "Cannot lexicographic compare a table with a LIST column");
    for (int i = 0; i < c.num_children(); ++i) {
      check_column(c.child(i));
    }
  };
  for (column_view const& c : input) {
    check_column(c);
  }
}

}  // namespace

namespace lexicographic_comparison {

preprocessed_table::preprocessed_table(table_view const& t,
                                       host_span<order const> column_order,
                                       host_span<null_order const> null_precedence,
                                       rmm::cuda_stream_view stream)
  : d_column_order(0, stream),
    d_null_precedence(0, stream),
    d_depths(0, stream),
    _has_nulls(has_nested_nulls(t))
{
  check_lex_compatibility(t);

  auto [verticalized_lhs, new_column_order, new_null_precedence, verticalized_col_depths] =
    struct_lex_verticalize(t, column_order, null_precedence);

  d_t =
    std::make_unique<table_device_view_owner>(table_device_view::create(verticalized_lhs, stream));

  d_column_order    = detail::make_device_uvector_async(new_column_order, stream);
  d_null_precedence = detail::make_device_uvector_async(new_null_precedence, stream);
  d_depths          = detail::make_device_uvector_async(verticalized_col_depths, stream);
}

}  // namespace lexicographic_comparison
}  // namespace experimental
}  // namespace cudf
