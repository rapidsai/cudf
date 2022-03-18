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

#include <jit/type.hpp>

namespace cudf {
namespace experimental {

namespace {

/**
 * @brief Linearizes all struct columns in a table.
 *
 * If a struct column is a tree with N leaves, then this function "linearizes" the tree into
 * N "linear trees" (branch factor == 1) and prunes common parents. Also returns a vector of
 * per-column `depth`s.
 *
 * A `depth` value is the number of nested levels as parent of the column in the original,
 * non-linearized table, which are pruned during linearizing.
 *
 * For example, if the original table has a column `Struct<Struct<int, float>, decimal>`,
 *      S
 *     / \
 *    S   d
 *   / \
 *  i   f
 * then after linearizing, we get three columns:
 * `Struct<Struct<int>>`, `float`, and `decimal`.
 * 0   2   1  <- depths
 * S
 * |
 * S       d
 * |
 * i   f
 * The depth of the first column is 0 because it contains all its parent levels, while the depth
 * of the second column is 2 because two of its parent struct levels were pruned.
 *
 * @param table The table to linearize.
 * @param column_order The per-column order if using linearized output with lexicographic comparison
 * @param null_precedence The per-column null precedence
 * @return A tuple containing a table with all struct columns linearized, new corresponding column
 *         orders and null precedences and depths of the linearized branches
 */
auto struct_linearize(table_view table,
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

/**
 * @brief Check a table for compatibility with lexicographic comparison
 *
 * Checks whether a given table contains columns of non-relationally comparable types.
 */
void check_lex_compatibility(table_view const& input)
{
  // Basically check if there's any LIST hiding anywhere in the table
  std::function<void(column_view const&)> check_column = [&](column_view const& c) {
    CUDF_EXPECTS(c.type().id() != type_id::LIST,
                 "Cannot lexicographic compare a table with a LIST column");
    if (not is_nested(c.type())) {
      CUDF_EXPECTS(is_relationally_comparable(c.type()),
                   "Cannot lexicographic compare a table with a column of type " +
                     jit::get_type_name(c.type()));
    }
    for (auto child = c.child_begin(); child < c.child_end(); ++child) {
      check_column(*child);
    }
  };
  for (column_view const& c : input) {
    check_column(c);
  }
}

}  // namespace

namespace lex {

std::shared_ptr<preprocessed_table> preprocessed_table::create(
  table_view const& t,
  host_span<order const> column_order,
  host_span<null_order const> null_precedence,
  rmm::cuda_stream_view stream)
{
  check_lex_compatibility(t);

  auto [verticalized_lhs, new_column_order, new_null_precedence, verticalized_col_depths] =
    decompose_structs(t, column_order, null_precedence);

  auto d_t               = table_device_view::create(verticalized_lhs, stream);
  auto d_column_order    = detail::make_device_uvector_async(new_column_order, stream);
  auto d_null_precedence = detail::make_device_uvector_async(new_null_precedence, stream);
  auto d_depths          = detail::make_device_uvector_async(verticalized_col_depths, stream);

  return std::shared_ptr<preprocessed_table>(new preprocessed_table(
    std::move(d_t), std::move(d_column_order), std::move(d_null_precedence), std::move(d_depths)));
}

}  // namespace lex
}  // namespace experimental
}  // namespace cudf
