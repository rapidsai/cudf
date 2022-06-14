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

#include "cudf/detail/utilities/cuda.cuh"
#include "rmm/exec_policy.hpp"
#include "thrust/gather.h"
#include "thrust/iterator/discard_iterator.h"
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

struct def_level_fn {
  column_device_view const* parent_col;
  uint8_t const* d_nullability;
  uint8_t sub_level_start;
  uint8_t curr_def_level;

  __device__ uint32_t operator()(size_type i)
  {
    uint32_t def       = curr_def_level;
    uint8_t l          = sub_level_start;
    bool is_col_struct = false;
    auto col           = *parent_col;
    do {
      // If col not nullable then it does not contribute to def levels
      if (d_nullability[l]) {
        if (not col.nullable() or bit_is_set(col.null_mask(), i)) {
          ++def;
        } else {  // We have found the shallowest level at which this row is null
          break;
        }
      }
      is_col_struct = (col.type().id() == type_id::STRUCT);
      if (is_col_struct) {
        col = col.child(0);
        ++l;
      }
    } while (is_col_struct);
    return def;
  }
};

row::lexicographic::dremel_data get_dremel_data(
  column_view h_col,
  // TODO(cp): use device_span once it is converted to a single hd_vec
  rmm::device_uvector<uint8_t> const& d_nullability,
  std::vector<uint8_t> const& nullability,
  rmm::cuda_stream_view stream)
{
  auto get_list_level = [](column_view col) {
    while (col.type().id() == type_id::STRUCT) {
      col = col.child(0);
    }
    return col;
  };

  auto get_empties = [&](column_view col, size_type start, size_type end) {
    auto lcv = lists_column_view(get_list_level(col));
    rmm::device_uvector<size_type> empties_idx(lcv.size(), stream);
    rmm::device_uvector<size_type> empties(lcv.size(), stream);
    auto d_off = lcv.offsets().data<size_type>();

    auto empties_idx_end =
      thrust::copy_if(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(start),
                      thrust::make_counting_iterator(end),
                      empties_idx.begin(),
                      [d_off] __device__(auto i) { return d_off[i] == d_off[i + 1]; });
    auto empties_end = thrust::gather(rmm::exec_policy(stream),
                                      empties_idx.begin(),
                                      empties_idx_end,
                                      lcv.offsets().begin<size_type>(),
                                      empties.begin());

    auto empties_size = empties_end - empties.begin();
    return std::make_tuple(std::move(empties), std::move(empties_idx), empties_size);
  };

  auto curr_col = h_col;
  std::vector<column_view> nesting_levels;
  std::vector<uint8_t> def_at_level;
  std::vector<uint8_t> start_at_sub_level;
  uint8_t curr_nesting_level_idx = 0;

  auto add_def_at_level = [&](column_view col) {
    // Add up all def level contributions in this column all the way till the first list column
    // appears in the hierarchy or until we get to leaf
    uint32_t def = 0;
    start_at_sub_level.push_back(curr_nesting_level_idx);
    while (col.type().id() == type_id::STRUCT) {
      def += (nullability[curr_nesting_level_idx]) ? 1 : 0;
      col = col.child(0);
      ++curr_nesting_level_idx;
    }
    // At the end of all those structs is either a list column or the leaf. Leaf column contributes
    // at least one def level. It doesn't matter what the leaf contributes because it'll be at the
    // end of the exclusive scan.
    def += (nullability[curr_nesting_level_idx]) ? 2 : 1;
    def_at_level.push_back(def);
    ++curr_nesting_level_idx;
  };
  while (cudf::is_nested(curr_col.type())) {
    nesting_levels.push_back(curr_col);
    add_def_at_level(curr_col);
    while (curr_col.type().id() == type_id::STRUCT) {
      // Go down the hierarchy until we get to the LIST or the leaf level
      curr_col = curr_col.child(0);
    }
    if (curr_col.type().id() == type_id::LIST) {
      curr_col = curr_col.child(lists_column_view::child_column_index);
      if (not is_nested(curr_col.type())) {
        // Special case: when the leaf data column is the immediate child of the list col then we
        // want it to be included right away. Otherwise the struct containing it will be included in
        // the next iteration of this loop.
        nesting_levels.push_back(curr_col);
        add_def_at_level(curr_col);
        break;
      }
    }
  }

  std::unique_ptr<rmm::device_buffer> device_view_owners;
  column_device_view* d_nesting_levels;
  std::tie(device_view_owners, d_nesting_levels) =
    contiguous_copy_column_device_views<column_device_view>(nesting_levels, stream);

  thrust::exclusive_scan(
    thrust::host, def_at_level.begin(), def_at_level.end(), def_at_level.begin());

  // Sliced list column views only have offsets applied to top level. Get offsets for each level.
  rmm::device_uvector<size_type> d_column_offsets(nesting_levels.size(), stream);
  rmm::device_uvector<size_type> d_column_ends(nesting_levels.size(), stream);

  auto d_col = column_device_view::create(h_col, stream);
  cudf::detail::device_single_thread(
    [offset_at_level  = d_column_offsets.data(),
     end_idx_at_level = d_column_ends.data(),
     col              = *d_col] __device__() {
      auto curr_col           = col;
      size_type off           = curr_col.offset();
      size_type end           = off + curr_col.size();
      size_type level         = 0;
      offset_at_level[level]  = off;
      end_idx_at_level[level] = end;
      ++level;
      // Apply offset recursively until we get to leaf data
      // Skip doing the following for any structs we encounter in between.
      while (curr_col.type().id() == type_id::LIST or curr_col.type().id() == type_id::STRUCT) {
        if (curr_col.type().id() == type_id::LIST) {
          off = curr_col.child(lists_column_view::offsets_column_index).element<size_type>(off);
          end = curr_col.child(lists_column_view::offsets_column_index).element<size_type>(end);
          offset_at_level[level]  = off;
          end_idx_at_level[level] = end;
          ++level;
          curr_col = curr_col.child(lists_column_view::child_column_index);
        } else {
          curr_col = curr_col.child(0);
        }
      }
    },
    stream);

  thrust::host_vector<size_type> column_offsets =
    cudf::detail::make_host_vector_async(d_column_offsets, stream);
  thrust::host_vector<size_type> column_ends =
    cudf::detail::make_host_vector_async(d_column_ends, stream);
  stream.synchronize();

  size_t max_vals_size = 0;
  for (size_t l = 0; l < column_offsets.size(); ++l) {
    max_vals_size += column_ends[l] - column_offsets[l];
  }

  rmm::device_uvector<uint8_t> rep_level(max_vals_size, stream);
  rmm::device_uvector<uint8_t> def_level(max_vals_size, stream);

  rmm::device_uvector<uint8_t> temp_rep_vals(max_vals_size, stream);
  rmm::device_uvector<uint8_t> temp_def_vals(max_vals_size, stream);
  rmm::device_uvector<size_type> new_offsets(0, stream);
  size_type curr_rep_values_size = 0;
  {
    // At this point, curr_col contains the leaf column. Max nesting level is
    // nesting_levels.size().

    // We are going to start by merging the last column in nesting_levels (the leaf, which is at the
    // index `nesting_levels.size() - 1`) with the second-to-last (which is at
    // `nesting_levels.size() - 2`).
    size_t level              = nesting_levels.size() - 2;
    curr_col                  = nesting_levels[level];
    auto lcv                  = lists_column_view(get_list_level(curr_col));
    auto offset_size_at_level = column_ends[level] - column_offsets[level] + 1;

    // Get empties at this level
    rmm::device_uvector<size_type> empties(0, stream);
    rmm::device_uvector<size_type> empties_idx(0, stream);
    size_t empties_size;
    std::tie(empties, empties_idx, empties_size) =
      get_empties(nesting_levels[level], column_offsets[level], column_ends[level]);

    // Merge empty at deepest parent level with the rep, def level vals at leaf level

    auto input_parent_rep_it = thrust::make_constant_iterator(level);
    auto input_parent_def_it =
      thrust::make_transform_iterator(empties_idx.begin(),
                                      def_level_fn{d_nesting_levels + level,
                                                   d_nullability.data(),
                                                   start_at_sub_level[level],
                                                   def_at_level[level]});

    // `nesting_levels.size()` == no of list levels + leaf. Max repetition level = no of list levels
    auto input_child_rep_it = thrust::make_constant_iterator(nesting_levels.size() - 1);
    auto input_child_def_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(column_offsets[level + 1]),
                                      def_level_fn{d_nesting_levels + level + 1,
                                                   d_nullability.data(),
                                                   start_at_sub_level[level + 1],
                                                   def_at_level[level + 1]});

    // Zip the input and output value iterators so that merge operation is done only once
    auto input_parent_zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(input_parent_rep_it, input_parent_def_it));

    auto input_child_zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(input_child_rep_it, input_child_def_it));

    auto output_zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(rep_level.begin(), def_level.begin()));

    auto ends = thrust::merge_by_key(rmm::exec_policy(stream),
                                     empties.begin(),
                                     empties.begin() + empties_size,
                                     thrust::make_counting_iterator(column_offsets[level + 1]),
                                     thrust::make_counting_iterator(column_ends[level + 1]),
                                     input_parent_zip_it,
                                     input_child_zip_it,
                                     thrust::make_discard_iterator(),
                                     output_zip_it);

    curr_rep_values_size = ends.second - output_zip_it;

    // Scan to get distance by which each offset value is shifted due to the insertion of empties
    auto scan_it = cudf::detail::make_counting_transform_iterator(
      column_offsets[level],
      [off = lcv.offsets().data<size_type>(), size = lcv.offsets().size()] __device__(
        auto i) -> int { return (i + 1 < size) && (off[i] == off[i + 1]); });
    rmm::device_uvector<size_type> scan_out(offset_size_at_level, stream);
    thrust::exclusive_scan(
      rmm::exec_policy(stream), scan_it, scan_it + offset_size_at_level, scan_out.begin());

    // Add scan output to existing offsets to get new offsets into merged rep level values
    new_offsets = rmm::device_uvector<size_type>(offset_size_at_level, stream);
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       offset_size_at_level,
                       [off      = lcv.offsets().data<size_type>() + column_offsets[level],
                        scan_out = scan_out.data(),
                        new_off  = new_offsets.data()] __device__(auto i) {
                         new_off[i] = off[i] - off[0] + scan_out[i];
                       });

    // Set rep level values at level starts to appropriate rep level
    auto scatter_it = thrust::make_constant_iterator(level);
    thrust::scatter(rmm::exec_policy(stream),
                    scatter_it,
                    scatter_it + new_offsets.size() - 1,
                    new_offsets.begin(),
                    rep_level.begin());
  }

  // Having already merged the last two levels, we are now going to merge the result with the
  // third-last level which is at index `nesting_levels.size() - 3`.
  for (int level = nesting_levels.size() - 3; level >= 0; level--) {
    curr_col                  = nesting_levels[level];
    auto lcv                  = lists_column_view(get_list_level(curr_col));
    auto offset_size_at_level = column_ends[level] - column_offsets[level] + 1;

    // Get empties at this level
    rmm::device_uvector<size_type> empties(0, stream);
    rmm::device_uvector<size_type> empties_idx(0, stream);
    size_t empties_size;
    std::tie(empties, empties_idx, empties_size) =
      get_empties(nesting_levels[level], column_offsets[level], column_ends[level]);

    auto offset_transformer = [new_child_offsets = new_offsets.data(),
                               child_start       = column_offsets[level + 1]] __device__(auto x) {
      return new_child_offsets[x - child_start];  // (x - child's offset)
    };

    // We will be reading from old rep_levels and writing again to rep_levels. Swap the current
    // rep values into temp_rep_vals so it can become the input and rep_levels can again be output.
    std::swap(temp_rep_vals, rep_level);
    std::swap(temp_def_vals, def_level);

    // Merge empty at parent level with the rep, def level vals at current level
    auto transformed_empties = thrust::make_transform_iterator(empties.begin(), offset_transformer);

    auto input_parent_rep_it = thrust::make_constant_iterator(level);
    auto input_parent_def_it =
      thrust::make_transform_iterator(empties_idx.begin(),
                                      def_level_fn{d_nesting_levels + level,
                                                   d_nullability.data(),
                                                   start_at_sub_level[level],
                                                   def_at_level[level]});

    // Zip the input and output value iterators so that merge operation is done only once
    auto input_parent_zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(input_parent_rep_it, input_parent_def_it));

    auto input_child_zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(temp_rep_vals.begin(), temp_def_vals.begin()));

    auto output_zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(rep_level.begin(), def_level.begin()));

    auto ends = thrust::merge_by_key(rmm::exec_policy(stream),
                                     transformed_empties,
                                     transformed_empties + empties_size,
                                     thrust::make_counting_iterator(0),
                                     thrust::make_counting_iterator(curr_rep_values_size),
                                     input_parent_zip_it,
                                     input_child_zip_it,
                                     thrust::make_discard_iterator(),
                                     output_zip_it);

    curr_rep_values_size = ends.second - output_zip_it;

    // Scan to get distance by which each offset value is shifted due to the insertion of dremel
    // level value fof an empty list
    auto scan_it = cudf::detail::make_counting_transform_iterator(
      column_offsets[level],
      [off = lcv.offsets().data<size_type>(), size = lcv.offsets().size()] __device__(
        auto i) -> int { return (i + 1 < size) && (off[i] == off[i + 1]); });
    rmm::device_uvector<size_type> scan_out(offset_size_at_level, stream);
    thrust::exclusive_scan(
      rmm::exec_policy(stream), scan_it, scan_it + offset_size_at_level, scan_out.begin());

    // Add scan output to existing offsets to get new offsets into merged rep level values
    rmm::device_uvector<size_type> temp_new_offsets(offset_size_at_level, stream);
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       offset_size_at_level,
                       [off      = lcv.offsets().data<size_type>() + column_offsets[level],
                        scan_out = scan_out.data(),
                        new_off  = temp_new_offsets.data(),
                        offset_transformer] __device__(auto i) {
                         new_off[i] = offset_transformer(off[i]) + scan_out[i];
                       });
    new_offsets = std::move(temp_new_offsets);

    // Set rep level values at level starts to appropriate rep level
    auto scatter_it = thrust::make_constant_iterator(level);
    thrust::scatter(rmm::exec_policy(stream),
                    scatter_it,
                    scatter_it + new_offsets.size() - 1,
                    new_offsets.begin(),
                    rep_level.begin());
  }

  size_t level_vals_size = new_offsets.back_element(stream);
  rep_level.resize(level_vals_size, stream);
  def_level.resize(level_vals_size, stream);

  stream.synchronize();

  size_type leaf_data_size = column_ends.back() - column_offsets.back();

  return row::lexicographic::dremel_data{
    std::move(new_offsets), std::move(rep_level), std::move(def_level), leaf_data_size};
}

auto list_lex_preprocess(table_view table, rmm::cuda_stream_view stream)
{
  std::vector<row::lexicographic::dremel_data> dremel_data;
  std::vector<uint8_t> max_def_levels;
  for (auto const& col : table) {
    if (col.type().id() == type_id::LIST) {
      // Check nullability of the list
      std::vector<uint8_t> nullability;
      auto cur_col          = col;
      uint8_t max_def_level = 0;
      while (cur_col.type().id() == type_id::LIST) {
        max_def_level += (cur_col.nullable() ? 2 : 1);
        nullability.push_back(static_cast<uint8_t>(cur_col.nullable()));
        cur_col = cur_col.child(lists_column_view::child_column_index);
      }
      max_def_level += (cur_col.nullable() ? 1 : 0);
      nullability.push_back(static_cast<uint8_t>(cur_col.nullable()));
      auto d_nullability = detail::make_device_uvector_async(nullability, stream);
      dremel_data.push_back(get_dremel_data(col, d_nullability, nullability, stream));
      max_def_levels.push_back(max_def_level);
    } else {
      max_def_levels.push_back(0);
    }
  }

  std::vector<size_type*> dremel_offsets;
  std::vector<uint8_t*> rep_levels;
  std::vector<uint8_t*> def_levels;
  size_type c = 0;
  for (auto const& col : table) {
    if (col.type().id() == type_id::LIST) {
      dremel_offsets.push_back(dremel_data[c].dremel_offsets.data());
      rep_levels.push_back(dremel_data[c].rep_level.data());
      def_levels.push_back(dremel_data[c].def_level.data());
      ++c;
    } else {
      dremel_offsets.push_back(nullptr);
      rep_levels.push_back(nullptr);
      def_levels.push_back(nullptr);
    }
  }
  auto d_dremel_offsets = detail::make_device_uvector_async(dremel_offsets, stream);
  auto d_rep_levels     = detail::make_device_uvector_async(rep_levels, stream);
  auto d_def_levels     = detail::make_device_uvector_async(def_levels, stream);
  auto d_max_def_levels = detail::make_device_uvector_async(max_def_levels, stream);
  return std::make_tuple(std::move(dremel_data),
                         std::move(d_dremel_offsets),
                         std::move(d_rep_levels),
                         std::move(d_def_levels),
                         std::move(d_max_def_levels));
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

  auto [dremel_data, d_dremel_offsets, d_rep_levels, d_def_levels, d_max_def_levels] =
    list_lex_preprocess(verticalized_lhs, stream);

  auto d_t               = table_device_view::create(verticalized_lhs, stream);
  auto d_column_order    = detail::make_device_uvector_async(new_column_order, stream);
  auto d_null_precedence = detail::make_device_uvector_async(new_null_precedence, stream);
  auto d_depths          = detail::make_device_uvector_async(verticalized_col_depths, stream);

  return std::shared_ptr<preprocessed_table>(new preprocessed_table(std::move(d_t),
                                                                    std::move(d_column_order),
                                                                    std::move(d_null_precedence),
                                                                    std::move(d_depths),
                                                                    std::move(dremel_data),
                                                                    std::move(d_dremel_offsets),
                                                                    std::move(d_rep_levels),
                                                                    std::move(d_def_levels),
                                                                    std::move(d_max_def_levels)));
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

  auto null_pushed_table              = structs::detail::superimpose_parent_nulls(t, stream);
  auto struct_offset_removed_table    = remove_struct_child_offsets(std::get<0>(null_pushed_table));
  auto [verticalized_lhs, _, __, ___] = decompose_structs(struct_offset_removed_table);

  auto d_t = table_device_view_owner(table_device_view::create(verticalized_lhs, stream));
  return std::shared_ptr<preprocessed_table>(
    new preprocessed_table(std::move(d_t), std::move(std::get<1>(null_pushed_table))));
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
