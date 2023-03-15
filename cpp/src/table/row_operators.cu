/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

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

/*
 * This helper function generates dremel data for any list-type columns in a
 * table. This data is necessary for lexicographic comparisons.
 */
auto list_lex_preprocess(table_view table, rmm::cuda_stream_view stream)
{
  std::vector<detail::dremel_data> dremel_data;
  std::vector<detail::dremel_device_view> dremel_device_views;
  for (auto const& col : table) {
    if (col.type().id() == type_id::LIST) {
      dremel_data.push_back(detail::get_comparator_data(col, {}, false, stream));
      dremel_device_views.push_back(dremel_data.back());
    }
  }
  auto d_dremel_device_views = detail::make_device_uvector_sync(dremel_device_views, stream);
  return std::make_tuple(std::move(dremel_data), std::move(d_dremel_device_views));
}

using column_checker_fn_t = std::function<void(column_view const&)>;

/**
 * @brief Check a table for compatibility with lexicographic comparison
 *
 * Checks whether a given table contains columns of non-relationally comparable types.
 */
void check_lex_compatibility(table_view const& input)
{
  // Basically check if there's any LIST of STRUCT or STRUCT of LIST hiding anywhere in the table
  column_checker_fn_t check_column = [&](column_view const& c) {
    if (c.type().id() == type_id::LIST) {
      auto const& list_col = lists_column_view(c);
      CUDF_EXPECTS(list_col.child().type().id() != type_id::STRUCT,
                   "Cannot lexicographic compare a table with a LIST of STRUCT column");
      check_column(list_col.child());
    } else if (c.type().id() == type_id::STRUCT) {
      for (auto child = c.child_begin(); child < c.child_end(); ++child) {
        CUDF_EXPECTS(child->type().id() != type_id::LIST,
                     "Cannot lexicographic compare a table with a STRUCT of LIST column");
        check_column(*child);
      }
    }
    if (not is_nested(c.type())) {
      CUDF_EXPECTS(is_relationally_comparable(c.type()),
                   "Cannot lexicographic compare a table with a column of type " +
                     cudf::type_to_name(c.type()));
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
                     cudf::type_to_name(c.type()));
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

namespace {

/**
 * @brief Transform a lists-of-structs column into lists-of-integers column.
 *
 * For lists-of-structs column at any nested level, the child structs column will be replaced by an
 * integer column of its ranks generated using `cudf::rank()`.
 *
 * If the input column is not lists-of-structs, or does not contain lists-of-structs at any nested
 * level, the input will be passed through.
 *
 * @param input The input column to transform
 * @param
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of new column_view representing the transformed input and the generated rank
 *         (integer) column which needs to be kept alive
 */
std::
  tuple<column_view, std::optional<column_view>, std::unique_ptr<column>, std::unique_ptr<column>>
  transform_lists_of_structs(column_view const& lhs,
                             std::optional<column_view> const& rhs,
                             rmm::cuda_stream_view stream)
{
  auto const make_transformed_input = [&](auto const& input, auto const& new_child) {
    return column_view{data_type{type_id::LIST},
                       input.size(),
                       nullptr,
                       input.null_mask(),
                       input.null_count(),
                       input.offset(),
                       {input.child(lists_column_view::offsets_column_index), new_child}};
  };

  auto const default_mr = rmm::mr::get_current_device_resource();

  if (lhs.type().id() == type_id::LIST) {
    // Should not use sliced child because we reuse the input's offset value and offsets child.
    auto const child_lhs = lhs.child(lists_column_view::child_column_index);
    auto const child_rhs =
      rhs ? std::optional<column_view>{rhs.value().child(lists_column_view::child_column_index)}
          : std::nullopt;

    if (child_lhs.type().id() == type_id::STRUCT) {
      if (child_rhs) {
        auto child_lhs_rhs = cudf::detail::concatenate(
          /*std::vector<column_view>*/ std::vector<column_view>{child_lhs, child_rhs.value()},
          stream,
          default_mr);

        // Dense ranks should be used because we are ranking two separate columns concatenating
        // together.
        auto const ranks        = cudf::detail::rank(child_lhs_rhs->view(),
                                              rank_method::DENSE,
                                              order::ASCENDING,
                                              null_policy::EXCLUDE,
                                              null_order::BEFORE,
                                              false /*percentage*/,
                                              stream,
                                              default_mr);
        auto const ranks_slices = cudf::detail::slice(
          ranks->view(),
          {0, child_lhs.size(), child_lhs.size(), child_lhs.size() + child_rhs.value().size()},
          stream);
        auto child_lhs_ranks = std::make_unique<column>(ranks_slices.front());
        auto child_rhs_ranks = std::make_unique<column>(ranks_slices.back());
        auto transformed_lhs = make_transformed_input(lhs, child_lhs_ranks->view());
        auto transformed_rhs = make_transformed_input(rhs.value(), child_rhs_ranks->view());
        return {
          transformed_lhs, transformed_rhs, std::move(child_lhs_ranks), std::move(child_rhs_ranks)};
      } else {
        // Dense ranks can accurately reflect the order of structs: structs compared equal will have
        // the same rank values.
        // However, first ranks are computed faster and are good enough for ordering them. Structs
        // compared equal always have consecutive rank values (in stable order) thus they are still
        // sorted correctly by their ranks.
        auto child_lhs_ranks = cudf::detail::rank(child_lhs,
                                                  rank_method::FIRST,
                                                  order::ASCENDING,
                                                  null_policy::EXCLUDE,
                                                  null_order::BEFORE,
                                                  false /*percentage*/,
                                                  stream,
                                                  default_mr);
        auto transformed_lhs = make_transformed_input(lhs, child_lhs_ranks->view());
        return {std::move(transformed_lhs), std::nullopt, std::move(child_lhs_ranks), nullptr};
      }
    } else if (child_lhs.type().id() == type_id::LIST) {
      auto [new_child_lhs, new_child_rhs_opt, child_lhs_ranks, child_rhs_ranks] =
        transform_lists_of_structs(child_lhs, child_rhs, stream);
      if (child_lhs_ranks) {
        auto transformed_lhs = make_transformed_input(lhs, new_child_lhs);
        auto transformed_rhs = make_transformed_input(rhs.value(), new_child_rhs_opt.value());
        return {
          transformed_lhs, transformed_rhs, std::move(child_lhs_ranks), std::move(child_rhs_ranks)};
      }
    }
  } else if (lhs.type().id() == type_id::STRUCT) {
    CUDF_UNREACHABLE("Structs columns should be flattened before calling this function.");
  }

  return {lhs, rhs, nullptr, nullptr};
}

/**
 * @brief Transform any lists-of-structs column in a given table into lists-of-integers column.
 *
 * If the rhs table is specified, its shape should be pre-checked to match with lhs through
 * `check_shape_compatibility`.
 *
 * @param input The input table to transform
 * @param tba
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of new table_view representing the transformed input and the generated rank
 *         (integer) column which needs to be kept alive
 */
std::tuple<table_view,
           std::optional<table_view>,
           std::vector<std::unique_ptr<column>>,
           std::vector<std::unique_ptr<column>>>
transform_lists_of_structs(table_view const& lhs,
                           std::optional<table_view> const& rhs,
                           rmm::cuda_stream_view stream)
{
  std::vector<column_view> transformed_lhs_cols;
  std::vector<column_view> transformed_rhs_cols;
  std::vector<std::unique_ptr<column>> lhs_aux_cols;
  std::vector<std::unique_ptr<column>> rhs_aux_cols;
  for (size_type child_idx = 0; child_idx < lhs.num_columns(); ++child_idx) {
    auto const& col = lhs.column(child_idx);
    auto [transformed_lhs, transformed_rhs_opt, lhs_aux_data, rhs_aux_data] =
      transform_lists_of_structs(
        col,
        rhs ? std::optional<column_view>{rhs.value().column(child_idx)} : std::nullopt,
        stream);
    transformed_lhs_cols.push_back(transformed_lhs);
    if (rhs) { transformed_rhs_cols.push_back(transformed_rhs_opt.value()); }
    if (lhs_aux_data) { lhs_aux_cols.emplace_back(std::move(lhs_aux_data)); }
    if (rhs_aux_data) { rhs_aux_cols.emplace_back(std::move(rhs_aux_data)); }
  }

  return {table_view{transformed_lhs_cols},
          rhs ? std::optional<table_view>{table_view{transformed_rhs_cols}} : std::nullopt,
          std::move(lhs_aux_cols),
          std::move(rhs_aux_cols)};
}

/**
 * @brief Check if the input column has structs-of-lists column at any nested level.
 *
 * @param input The input column to check
 * @return Boolean value indicating if there is structs-of-lists column
 */
bool has_nested_structs_of_lists(column_view const& input)
{
  if (input.type().id() == type_id::STRUCT) {
    return std::any_of(input.child_begin(), input.child_end(), [](auto const& child) {
      return child.type().id() == type_id::LIST || has_nested_structs_of_lists(child);
    });
  } else if (input.type().id() == type_id::LIST) {
    return has_nested_structs_of_lists(input.child(lists_column_view::child_column_index));
  }

  return false;
}

/**
 * @brief Flatten the given table if it contains any structs-of-lists column.
 *
 * If the input table contains any structs-of-lists column, the entire table will be flattened to
 * a table of non-struct columns. Otherwise, the input table is passed through.
 *
 * @param input The input table
 * @param stream The stream to launch kernels and h->d copies on while preprocessing
 * @return A pair of table_view representing the flattened input and an auxiliary data structure
 *         that needs to be kept alive
 */
std::pair<table_view, std::unique_ptr<cudf::structs::detail::flattened_table>>
flatten_nested_structs_of_lists(table_view const& input,
                                host_span<order const> column_order,
                                host_span<null_order const> null_precedence,
                                rmm::cuda_stream_view stream)
{
  if (std::any_of(input.begin(), input.end(), has_nested_structs_of_lists)) {
    auto structs_flattened = cudf::structs::detail::flatten_nested_columns(
      input,
      std::vector<order>{column_order.begin(), column_order.end()},
      std::vector<null_order>{null_precedence.begin(), null_precedence.end()},
      cudf::structs::detail::column_nullability::FORCE,
      stream);
    auto output_table = structs_flattened->flattened_columns();
    return {std::move(output_table), std::move(structs_flattened)};
  }

  return {input, nullptr};
}

}  // namespace

std::shared_ptr<preprocessed_table> preprocessed_table::create_preprocessed_table(
  table_view const& input,
  std::unique_ptr<structs::detail::flattened_table>&& flattened_input_aux_data,
  std::vector<std::unique_ptr<column>>&& transformed_aux_data,
  host_span<order const> column_order,
  host_span<null_order const> null_precedence,
  bool safe_for_two_table_comparator,
  rmm::cuda_stream_view stream)
{
  check_lex_compatibility(input);

  auto [verticalized_lhs, new_column_order, new_null_precedence, verticalized_col_depths] =
    flattened_input_aux_data
      ? decompose_structs(
          input, flattened_input_aux_data->orders(), flattened_input_aux_data->null_orders())
      : decompose_structs(input, column_order, null_precedence);

  auto d_t               = table_device_view::create(verticalized_lhs, stream);
  auto d_column_order    = detail::make_device_uvector_async(new_column_order, stream);
  auto d_null_precedence = detail::make_device_uvector_async(new_null_precedence, stream);
  auto d_depths          = detail::make_device_uvector_async(verticalized_col_depths, stream);

  if (detail::has_nested_columns(input)) {
    auto [dremel_data, d_dremel_device_view] = list_lex_preprocess(verticalized_lhs, stream);
    return std::shared_ptr<preprocessed_table>(
      new preprocessed_table(std::move(d_t),
                             std::move(d_column_order),
                             std::move(d_null_precedence),
                             std::move(d_depths),
                             std::move(dremel_data),
                             std::move(d_dremel_device_view),
                             std::move(flattened_input_aux_data),
                             std::move(transformed_aux_data),
                             safe_for_two_table_comparator));
  } else {
    return std::shared_ptr<preprocessed_table>(
      new preprocessed_table(std::move(d_t),
                             std::move(d_column_order),
                             std::move(d_null_precedence),
                             std::move(d_depths),
                             std::move(flattened_input_aux_data),
                             std::move(transformed_aux_data),
                             safe_for_two_table_comparator));
  }
}

std::shared_ptr<preprocessed_table> preprocessed_table::create(
  table_view const& t,
  host_span<order const> column_order,
  host_span<null_order const> null_precedence,
  rmm::cuda_stream_view stream)
{
  // Firstly, flatten the input table if it contains any structs-of-lists column.
  auto [flattened_t, flattened_t_aux_data] =
    flatten_nested_structs_of_lists(t, column_order, null_precedence, stream);

  // Next, transform any (nested) lists-of-structs column into lists-of-integers column.
  [[maybe_unused]] auto [transformed_t, unused_0, transformed_aux_data, unused_1] =
    transform_lists_of_structs(flattened_t, std::nullopt, stream);

  // Since the preprocessed_table is created alone, it is safe for two-table comparator
  // only if not any transformation for lists-of-structs was performed.
  bool const safe_for_two_table_comparator = transformed_aux_data.size() == 0;

  return create_preprocessed_table(transformed_t,
                                   std::move(flattened_t_aux_data),
                                   std::move(transformed_aux_data),
                                   column_order,
                                   null_precedence,
                                   safe_for_two_table_comparator,
                                   stream);
}

std::pair<std::shared_ptr<preprocessed_table>, std::shared_ptr<preprocessed_table>>
preprocessed_table::create(table_view const& lhs,
                           table_view const& rhs,
                           host_span<order const> column_order,
                           host_span<null_order const> null_precedence,
                           rmm::cuda_stream_view stream)
{
  // Firstly, flatten the input table if it contains any structs-of-lists column.
  auto [flattened_lhs, flattened_lhs_aux_data] =
    flatten_nested_structs_of_lists(lhs, column_order, null_precedence, stream);
  auto [flattened_rhs, flattened_rhs_aux_data] =
    flatten_nested_structs_of_lists(rhs, column_order, null_precedence, stream);

  // Next, transform any (nested) lists-of-structs column into lists-of-integers column.
  auto [transformed_lhs, transformed_rhs_opt, transformed_aux_lhs, transformed_aux_rhs] =
    transform_lists_of_structs(flattened_lhs, flattened_rhs, stream);

  return {create_preprocessed_table(transformed_lhs,
                                    std::move(flattened_lhs_aux_data),
                                    std::move(transformed_aux_lhs),
                                    column_order,
                                    null_precedence,
                                    true /*safe_for_two_table_comparator*/
                                    ,
                                    stream),
          create_preprocessed_table(transformed_rhs_opt.value(),
                                    std::move(flattened_rhs_aux_data),
                                    std::move(transformed_aux_rhs),
                                    column_order,
                                    null_precedence,
                                    true /*safe_for_two_table_comparator*/,
                                    stream)};
}

preprocessed_table::preprocessed_table(
  table_device_view_owner&& table,
  rmm::device_uvector<order>&& column_order,
  rmm::device_uvector<null_order>&& null_precedence,
  rmm::device_uvector<size_type>&& depths,
  std::vector<detail::dremel_data>&& dremel_data,
  rmm::device_uvector<detail::dremel_device_view>&& dremel_device_views,
  std::unique_ptr<structs::detail::flattened_table>&& flattened_input_aux_data,
  std::vector<std::unique_ptr<column>>&& transformed_structs_columns,
  bool safe_for_two_table_comparator)
  : _t(std::move(table)),
    _column_order(std::move(column_order)),
    _null_precedence(std::move(null_precedence)),
    _depths(std::move(depths)),
    _dremel_data(std::move(dremel_data)),
    _dremel_device_views(std::move(dremel_device_views)),
    _flattened_input_aux_data(std::move(flattened_input_aux_data)),
    _transformed_structs_aux_data(std::move(transformed_structs_columns)),
    _safe_for_two_table_comparator(safe_for_two_table_comparator)
{
}

preprocessed_table::preprocessed_table(
  table_device_view_owner&& table,
  rmm::device_uvector<order>&& column_order,
  rmm::device_uvector<null_order>&& null_precedence,
  rmm::device_uvector<size_type>&& depths,
  std::unique_ptr<structs::detail::flattened_table>&& flattened_input_aux_data,
  std::vector<std::unique_ptr<column>>&& transformed_structs_columns,
  bool safe_for_two_table_comparator)
  : _t(std::move(table)),
    _column_order(std::move(column_order)),
    _null_precedence(std::move(null_precedence)),
    _depths(std::move(depths)),
    _dremel_data{},
    _dremel_device_views{},
    _flattened_input_aux_data(std::move(flattened_input_aux_data)),
    _transformed_structs_aux_data(std::move(transformed_structs_columns)),
    _safe_for_two_table_comparator(safe_for_two_table_comparator)
{
}

two_table_comparator::two_table_comparator(table_view const& left,
                                           table_view const& right,
                                           host_span<order const> column_order,
                                           host_span<null_order const> null_precedence,
                                           rmm::cuda_stream_view stream)
{
  check_shape_compatibility(left, right);
  std::tie(d_left_table, d_right_table) =
    preprocessed_table::create(left, right, column_order, null_precedence, stream);
}

}  // namespace lexicographic

namespace equality {

std::shared_ptr<preprocessed_table> preprocessed_table::create(table_view const& t,
                                                               rmm::cuda_stream_view stream)
{
  check_eq_compatibility(t);

  auto [null_pushed_table, nullable_data] = structs::detail::push_down_nulls(t, stream);
  auto struct_offset_removed_table        = remove_struct_child_offsets(null_pushed_table);
  auto verticalized_t = std::get<0>(decompose_structs(struct_offset_removed_table));

  auto d_t = table_device_view_owner(table_device_view::create(verticalized_t, stream));
  return std::shared_ptr<preprocessed_table>(new preprocessed_table(
    std::move(d_t), std::move(nullable_data.new_null_masks), std::move(nullable_data.new_columns)));
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
