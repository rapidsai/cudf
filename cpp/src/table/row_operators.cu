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
#include <cudf/detail/utilities/column.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <jit/type.hpp>

#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace experimental {

namespace {

/**
 * @brief Removes the offsets of struct column's children
 *
 * @param c The column whose children are to be un-sliced
 * @return Children of `c` with offsets removed
 */
std::vector<column_view> unslice_children(column_view const& c)
{
  if (c.type().id() == type_id::STRUCT) {
    auto child_it = thrust::make_transform_iterator(c.child_begin(), [](auto const& child) {
      return column_view(
        child.type(),
        child.offset() + child.size(),  // This is hacky, we don't know the actual unsliced size but
                                        // it is at least offset + size
        child.head(),
        child.null_mask(),
        child.null_count(),
        0,
        unslice_children(child));
    });
    return {child_it, child_it + c.num_children()};
  }
  return {c.child_begin(), c.child_end()};
};

/**
 * @brief Removes the child column offsets of struct columns in a table.
 *
 * Given a table, this replaces any struct columns with similar struct columns that have their
 * offsets removed from their children. Structs that are children of list columns are not affected.
 *
 */
table_view remove_struct_child_offsets(table_view table)
{
  std::vector<column_view> cols;
  cols.reserve(table.num_columns());
  std::transform(table.begin(), table.end(), std::back_inserter(cols), [&](column_view const& c) {
    return column_view(c.type(),
                       c.size(),
                       c.head<uint8_t>(),
                       c.null_mask(),
                       c.null_count(),
                       c.offset(),
                       unslice_children(c));
  });
  return table_view(cols);
}

/**
 * @brief Decompose all struct columns in a table
 *
 * If a struct column is a tree with N leaves, then this function decomposes the tree into
 * N "linear trees" (branch factor == 1) and prunes common parents. Also returns a vector of
 * per-column `depth`s.
 *
 * A `depth` value is the number of nested levels as parent of the column in the original,
 * non-decomposed table, which are pruned during decomposition.
 *
 * For example, if the original table has a column `Struct<Struct<int, float>, decimal>`,
 *
 *      S1
 *     / \
 *    S2  d
 *   / \
 *  i   f
 *
 * then after decomposition, we get three columns:
 * `Struct<Struct<int>>`, `float`, and `decimal`.
 *
 *  0   2   1  <- depths
 *  S1
 *  |
 *  S2      d
 *  |
 *  i   f
 *
 * The depth of the first column is 0 because it contains all its parent levels, while the depth
 * of the second column is 2 because two of its parent struct levels were pruned.
 *
 * Similarly, a struct column of type Struct<int, Struct<float, decimal>> is decomposed as follows
 *
 *     S1
 *    / \
 *   i   S2
 *      / \
 *     f   d
 *
 *  0   1   2  <- depths
 *  S1  S2  d
 *  |   |
 *  i   f
 *
 * When list columns are present, the decomposition is performed similarly to pure structs but list
 * parent columns are NOT pruned
 *
 * For example, if the original table has a column `List<Struct<int, float>>`,
 *
 *    L
 *    |
 *    S
 *   / \
 *  i   f
 *
 * after decomposition, we get two columns
 *
 *  L   L
 *  |   |
 *  S   f
 *  |
 *  i
 *
 * The list parents are still needed to define the range of elements in the leaf that belong to the
 * same row.
 *
 * @param table The table whose struct columns to decompose.
 * @param column_order The per-column order if using output with lexicographic comparison
 * @param null_precedence The per-column null precedence
 * @return A tuple containing a table with all struct columns decomposed, new corresponding column
 *         orders and null precedences and depths of the linearized branches
 */
auto decompose_structs(table_view table,
                       host_span<order const> column_order         = {},
                       host_span<null_order const> null_precedence = {})
{
  auto linked_columns = detail::table_to_linked_columns(table);

  std::vector<column_view> verticalized_columns;
  std::vector<order> new_column_order;
  std::vector<null_order> new_null_precedence;
  std::vector<int> verticalized_col_depths;
  for (size_t col_idx = 0; col_idx < linked_columns.size(); ++col_idx) {
    detail::linked_column_view const* col = linked_columns[col_idx].get();
    if (is_nested(col->type())) {
      // convert and insert
      std::vector<std::vector<detail::linked_column_view const*>> flattened;
      std::function<void(
        detail::linked_column_view const*, std::vector<detail::linked_column_view const*>*, int)>
        recursive_child = [&](detail::linked_column_view const* c,
                              std::vector<detail::linked_column_view const*>* branch,
                              int depth) {
          branch->push_back(c);
          if (c->type().id() == type_id::LIST) {
            recursive_child(
              c->children[lists_column_view::child_column_index].get(), branch, depth + 1);
          } else if (c->type().id() == type_id::STRUCT) {
            for (size_t child_idx = 0; child_idx < c->children.size(); ++child_idx) {
              if (child_idx > 0) {
                verticalized_col_depths.push_back(depth + 1);
                branch = &flattened.emplace_back();
              }
              recursive_child(c->children[child_idx].get(), branch, depth + 1);
            }
          }
        };
      auto& branch = flattened.emplace_back();
      verticalized_col_depths.push_back(0);
      recursive_child(col, &branch, 0);

      for (auto const& branch : flattened) {
        column_view temp_col = *branch.back();
        for (auto it = branch.crbegin() + 1; it < branch.crend(); ++it) {
          auto const& prev_col = *(*it);
          auto children =
            (prev_col.type().id() == type_id::LIST)
              ? std::vector<column_view>{*prev_col
                                            .children[lists_column_view::offsets_column_index],
                                         temp_col}
              : std::vector<column_view>{temp_col};
          temp_col = column_view(prev_col.type(),
                                 prev_col.size(),
                                 nullptr,
                                 prev_col.null_mask(),
                                 UNKNOWN_NULL_COUNT,
                                 prev_col.offset(),
                                 std::move(children));
        }
        // Traverse upward and include any list columns in the ancestors
        for (detail::linked_column_view* parent = branch.front()->parent; parent;
             parent                             = parent->parent) {
          if (parent->type().id() == type_id::LIST) {
            // Include this parent
            temp_col = column_view(
              parent->type(),
              parent->size(),
              nullptr,  // list has no data of its own
              nullptr,  // If we're going through this then nullmask is already in another branch
              UNKNOWN_NULL_COUNT,
              parent->offset(),
              {*parent->children[lists_column_view::offsets_column_index], temp_col});
          } else if (parent->type().id() == type_id::STRUCT) {
            // Replace offset with parent's offset
            temp_col = column_view(temp_col.type(),
                                   parent->size(),
                                   temp_col.head(),
                                   temp_col.null_mask(),
                                   UNKNOWN_NULL_COUNT,
                                   parent->offset(),
                                   {temp_col.child_begin(), temp_col.child_end()});
          }
        }
        verticalized_columns.push_back(temp_col);
      }
      if (not column_order.empty()) {
        new_column_order.insert(new_column_order.end(), flattened.size(), column_order[col_idx]);
      }
      if (not null_precedence.empty()) {
        new_null_precedence.insert(
          new_null_precedence.end(), flattened.size(), null_precedence[col_idx]);
      }
    } else {
      verticalized_columns.push_back(*col);
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

using column_checker_fn_t = std::function<void(column_view const&)>;

/**
 * @brief Check a table for compatibility with lexicographic comparison
 *
 * Checks whether a given table contains columns of non-relationally comparable types.
 */
void check_lex_compatibility(table_view const& input)
{
  // Basically check if there's any LIST hiding anywhere in the table
  column_checker_fn_t check_column = [&](column_view const& c) {
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

/**
 * @brief Check a table for compatibility with equality comparison
 *
 * Checks whether a given table contains columns of non-equality comparable types.
 */
void check_eq_compatibility(table_view const& input)
{
  column_checker_fn_t check_column = [&](column_view const& c) {
    if (not is_nested(c.type())) {
      CUDF_EXPECTS(is_equality_comparable(c.type()),
                   "Cannot compare equality for a table with a column of type " +
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

void check_shape_compatibility(table_view const& lhs, table_view const& rhs)
{
  CUDF_EXPECTS(lhs.num_columns() == rhs.num_columns(),
               "Cannot compare tables with different number of columns");
  for (size_type i = 0; i < lhs.num_columns(); ++i) {
    CUDF_EXPECTS(column_types_equal(lhs.column(i), rhs.column(i)),
                 "Cannot compare tables with different column types");
  }
}

}  // namespace

namespace row {

namespace lexicographic {

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

two_table_comparator::two_table_comparator(table_view const& left,
                                           table_view const& right,
                                           host_span<order const> column_order,
                                           host_span<null_order const> null_precedence,
                                           rmm::cuda_stream_view stream)
  : d_left_table{preprocessed_table::create(left, column_order, null_precedence, stream)},
    d_right_table{preprocessed_table::create(right, column_order, null_precedence, stream)}
{
  check_shape_compatibility(left, right);
}

}  // namespace lexicographic

namespace equality {

std::shared_ptr<preprocessed_table> preprocessed_table::create(table_view const& t,
                                                               rmm::cuda_stream_view stream)
{
  check_eq_compatibility(t);

  auto [null_pushed_table, null_masks] = structs::detail::superimpose_parent_nulls(t, stream);
  auto struct_offset_removed_table     = remove_struct_child_offsets(null_pushed_table);
  auto [verticalized_lhs, _, __, ___]  = decompose_structs(struct_offset_removed_table);

  auto d_t = table_device_view_owner(table_device_view::create(verticalized_lhs, stream));
  return std::shared_ptr<preprocessed_table>(
    new preprocessed_table(std::move(d_t), std::move(null_masks)));
}

two_table_comparator::two_table_comparator(table_view const& left,
                                           table_view const& right,
                                           rmm::cuda_stream_view stream)
  : d_left_table{preprocessed_table::create(left, stream)},
    d_right_table{preprocessed_table::create(right, stream)}
{
  check_shape_compatibility(left, right);
}

}  // namespace equality

}  // namespace row
}  // namespace experimental
}  // namespace cudf
