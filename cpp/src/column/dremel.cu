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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/dremel.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace cudf::detail {

/**
 * @brief Functor to get definition level value for a nested struct column until the leaf level or
 * the first list level.
 *
 */
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

/**
 * @brief Get the dremel offsets and repetition and definition levels for a LIST column
 *
 * The repetition and definition level values are ideally computed using a recursive call over a
 * nested structure but in order to better utilize GPU resources, this function calculates them
 * with a bottom up merge method.
 *
 * Given a LIST column of type `List<List<int>>` like so:
 * ```
 * col = {
 *    [],
 *    [[], [1, 2, 3], [4, 5]],
 *    [[]]
 * }
 * ```
 * We can represent it in cudf format with two level of offsets like this:
 * ```
 * Level 0 offsets = {0, 0, 3, 5, 6}
 * Level 1 offsets = {0, 0, 3, 5, 5}
 * Values          = {1, 2, 3, 4, 5}
 * ```
 * The desired result of this function is the repetition and definition level values that
 * correspond to the data values:
 * ```
 * col = {[], [[], [1, 2, 3], [4, 5]], [[]]}
 * def = { 0    1,  2, 2, 2,   2, 2,     1 }
 * rep = { 0,   0,  0, 2, 2,   1, 2,     0 }
 * ```
 *
 * Since repetition and definition levels arrays contain a value for each empty list, the size of
 * the rep/def level array can be given by
 * ```
 * rep_level.size() = size of leaf column + number of empty lists in level 0
 *                                        + number of empty lists in level 1 ...
 * ```
 *
 * We start with finding the empty lists in the penultimate level and merging it with the indices
 * of the leaf level. The values for the merge are the definition and repetition levels
 * ```
 * empties at level 1 = {0, 5}
 * def values at 1    = {1, 1}
 * rep values at 1    = {1, 1}
 * indices at leaf    = {0, 1, 2, 3, 4}
 * def values at leaf = {2, 2, 2, 2, 2}
 * rep values at leaf = {2, 2, 2, 2, 2}
 * ```
 *
 * merged def values  = {1, 2, 2, 2, 2, 2, 1}
 * merged rep values  = {1, 2, 2, 2, 2, 2, 1}
 *
 * The size of the rep/def values is now larger than the leaf values and the offsets need to be
 * adjusted in order to point to the correct start indices. We do this with an exclusive scan over
 * the indices of offsets of empty lists and adding to existing offsets.
 * ```
 * Level 1 new offsets = {0, 1, 4, 6, 7}
 * ```
 * Repetition values at the beginning of a list need to be decremented. We use the new offsets to
 * scatter the rep value.
 * ```
 * merged rep values  = {1, 2, 2, 2, 2, 2, 1}
 * scatter (1, new offsets)
 * new offsets        = {0, 1,       4,    6, 7}
 * new rep values     = {1, 1, 2, 2, 1, 2, 1}
 * ```
 *
 * Similarly we merge up all the way till level 0 offsets
 *
 * STRUCT COLUMNS :
 * In case of struct columns, we don't have to merge struct levels with their children because a
 * struct is the same size as its children. e.g. for a column `struct<int, float>`, if the row `i`
 * is null, then the children columns `int` and `float` are also null at `i`. They also have the
 * null entry represented in their respective null masks. So for any case of strictly struct based
 * nesting, we can get the definition levels merely by iterating over the nesting for the same row.
 *
 * In case struct and lists are intermixed, the definition levels of all the contiguous struct
 * levels can be constructed using the aforementioned iterative method. Only when we reach a list
 * level, we need to do a merge with the subsequent level.
 *
 * So, for a column like `struct<list<int>>`, we are going to merge between the levels `struct<list`
 * and `int`.
 * For a column like `list<struct<int>>`, we are going to merge between `list` and `struct<int>`.
 *
 * In general, one nesting level is the list level and any struct level that precedes it.
 *
 * A few more examples to visualize the partitioning of column hierarchy into nesting levels:
 * (L is list, S is struct, i is integer(leaf data level), angle brackets omitted)
 * ```
 * 1. LSi     = L   Si
 *              - | --
 *
 * 2. LLSi    = L   L   Si
 *              - | - | --
 *
 * 3. SSLi    = SSL   i
 *              --- | -
 *
 * 4. LLSLSSi = L   L   SL   SSi
 *              - | - | -- | ---
 * ```
 */
dremel_data get_dremel_data(column_view h_col,
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

  // Check if there are empty lists with empty offsets in this column
  bool has_empty_list_offsets = false;
  {
    auto curr_col = h_col;
    while (is_nested(curr_col.type())) {
      if (curr_col.type().id() == type_id::LIST) {
        auto lcv = lists_column_view(curr_col);
        if (lcv.offsets().size() == 0) {
          has_empty_list_offsets = true;
          break;
        }
        curr_col = lcv.child();
      } else if (curr_col.type().id() == type_id::STRUCT) {
        curr_col = curr_col.child(0);
      }
    }
  }
  std::unique_ptr<column> empty_list_offset_col;
  if (has_empty_list_offsets) {
    empty_list_offset_col = make_fixed_width_column(data_type(type_id::INT32), 1);
    cudaMemsetAsync(empty_list_offset_col->mutable_view().head(), 0, sizeof(size_type), stream);
    std::function<column_view(column_view const&)> normalize_col = [&](column_view const& col) {
      auto children = [&]() -> std::vector<column_view> {
        if (col.type().id() == type_id::LIST) {
          auto lcol = lists_column_view(col);
          auto offset_col =
            lcol.offsets().head() == nullptr ? empty_list_offset_col->view() : lcol.offsets();
          return {offset_col, normalize_col(lcol.child())};
        } else if (col.type().id() == type_id::STRUCT) {
          return {normalize_col(col.child(0))};
        } else {
          return {col.child_begin(), col.child_end()};
        }
      }();
      return column_view(col.type(),
                         col.size(),
                         col.head(),
                         col.null_mask(),
                         UNKNOWN_NULL_COUNT,
                         col.offset(),
                         std::move(children));
    };
    h_col = normalize_col(h_col);
  }

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
    // At the end of all those structs is either a list column or the leaf. List column contributes
    // at least one def level. Leaf contributes 1 level only if it is nullable.
    def +=
      (col.type().id() == type_id::LIST ? 1 : 0) + (nullability[curr_nesting_level_idx] ? 1 : 0);
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

  auto [device_view_owners, d_nesting_levels] =
    contiguous_copy_column_device_views<column_device_view>(nesting_levels, stream);

  auto max_def_level = def_at_level.back();
  thrust::exclusive_scan(
    thrust::host, def_at_level.begin(), def_at_level.end(), def_at_level.begin());
  max_def_level += def_at_level.back();

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

  auto d_nullability = cudf::detail::make_device_uvector_async(nullability, stream);

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
    auto [empties, empties_idx, empties_size] =
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
    auto [empties, empties_idx, empties_size] =
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

  return dremel_data{std::move(new_offsets),
                     std::move(rep_level),
                     std::move(def_level),
                     leaf_data_size,
                     max_def_level};
}

}  // namespace cudf::detail
