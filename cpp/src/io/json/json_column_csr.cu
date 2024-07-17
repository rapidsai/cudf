/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "json_utils.hpp"
#include "nested_json.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace cudf::io::json::experimental::detail {

struct device_json_column_properties_size {
  rmm::device_uvector<NodeIndexT> outcol_nodes;
  size_t string_offsets_size = 0;
  size_t string_lengths_size = 0;
  size_t child_offsets_size  = 0;
  size_t num_rows_size       = 0;
};

device_json_column_properties_size estimate_device_json_column_size(
  rmm::device_uvector<NodeIndexT> const& rowidx,
  rmm::device_uvector<NodeIndexT> const& colidx,
  rmm::device_uvector<NodeT> const& categories,
  cudf::io::json_reader_options reader_options,
  rmm::cuda_stream_view stream)
{
  // What are the cases in which estimation works?
  CUDF_EXPECTS(reader_options.is_enabled_mixed_types_as_string() == false,
               "mixed type as string has not yet been implemented");
  CUDF_EXPECTS(reader_options.is_enabled_prune_columns() == false,
               "column pruning has not yet been implemented");
  // traverse the column tree
  auto num_columns = rowidx.size() - 1;

  // 1. TODO: removing NC_ERR nodes and their descendants i.e.
  // removing the entire subtree rooted at the nodes with category NC_ERR
  // for now, we just assert that there are indeed no error nodes
  auto num_err_nodes = thrust::count_if(
    rmm::exec_policy(stream), categories.begin(), categories.end(), [] __device__(auto const ctg) {
      return ctg == NC_ERR;
    });
  CUDF_EXPECTS(num_err_nodes == 0, "oops, there are some error nodes in the column tree!");

  // 2. Let's do some validation of the column tree based on its properties.
  // We will be using these properties to filter nodes later on.
  // ===========================================================================
  // (i) Every node v is of type string, val, field name, list or struct.
  // (ii) String and val cannot have any children i.e. they can only be leaf nodes
  // (iii) If v is a field name, it can have struct, list, string and val as children.
  // (iv) If v is a struct, it can have a field name as child
  // (v) If v is a list, it can have string, val, list or struct as child
  // (vi) There can only be at most one string and one val child for a given node, but many struct,
  // list and field name children. (vii) When mixed type support is disabled -
  //       (a) A mix of lists and structs in the same column is not supported i.e a field name and
  //       list node cannot have both list and struct as children (b) If there is a mix of str/val
  //       and list/struct in the same column, then str/val is discarded

  // Validation of (vii)(a)
  auto num_field_and_list_nodes = thrust::count_if(
    rmm::exec_policy(stream), categories.begin(), categories.end(), [] __device__(auto const ctg) {
      return ctg == NC_FN || ctg == NC_LIST;
    });
  rmm::device_uvector<NodeIndexT> field_and_list_nodes(num_field_and_list_nodes, stream);
  thrust::partition_copy(rmm::exec_policy(stream),
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(0) + num_columns,
                         field_and_list_nodes.begin(),
                         thrust::make_discard_iterator(),
                         [categories = categories.begin()] __device__(NodeIndexT node) {
                           return categories[node] == NC_LIST || categories[node] == NC_FN;
                         });
  bool is_valid_tree = thrust::all_of(
    rmm::exec_policy(stream),
    field_and_list_nodes.begin(),
    field_and_list_nodes.end(),
    [rowidx = rowidx.begin(), colidx = colidx.begin(), categories = categories.begin()] __device__(
      NodeIndexT node) {
      NodeIndexT first_child_pos = rowidx[node] + 1;
      NodeIndexT last_child_pos  = rowidx[node + 1] - 1;
      bool has_struct_child      = false;
      bool has_list_child        = false;
      for (NodeIndexT child_pos = first_child_pos; child_pos <= last_child_pos; child_pos++) {
        if (categories[colidx[child_pos]] == NC_STRUCT) has_struct_child = true;
        if (categories[colidx[child_pos]] == NC_LIST) has_list_child = true;
      }
      return !has_struct_child && !has_list_child;
    });

  CUDF_EXPECTS(is_valid_tree,
               "Invalidating property 7a i.e. mix of LIST and STRUCT in same column is not "
               "supported when mixed type support is disabled");

  // Validation of (vii)(b) i.e. ignore_vals in previous implementation
  // We need to identify leaf nodes that have non-leaf sibling nodes
  // i.e. we need to ignore leaf nodes at level above the last level
  // idea: leaf nodes have adjacency 1. So if there is an adjacency 1 inbetween non-one
  // adjacencies, then found the leaf node. Corner case: consider the last set of consecutive
  // ones. If the leftmost of those ones (say node u) has a non-leaf sibling
  // (can be found by looking at the adjacencies of the siblings
  // (which are in turn found from the colidx of the parent u), then this leaf node should be
  // ignored, otherwise all good.
  rmm::device_uvector<NodeIndexT> adjacency(
    num_columns + 1,
    stream);  // since adjacent_difference requires that the output have the same length as input
  thrust::adjacent_difference(
    rmm::exec_policy(stream), rowidx.begin(), rowidx.end(), adjacency.begin());
  auto num_leaf_nodes = thrust::count_if(rmm::exec_policy(stream),
                                         adjacency.begin() + 1,
                                         adjacency.end(),
                                         [] __device__(auto const adj) { return adj == 1; });
  rmm::device_uvector<NodeIndexT> leaf_nodes(num_leaf_nodes, stream);
  thrust::copy_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + num_columns,
    leaf_nodes.begin(),
    [adjacency = adjacency.begin()] __device__(size_t node) { return adjacency[node] == 1; });

  auto rev_node_it = thrust::make_reverse_iterator(thrust::make_counting_iterator(0) + num_columns);
  auto rev_leaf_nodes_it = thrust::make_reverse_iterator(leaf_nodes.begin());
  auto is_leftmost_leaf  = thrust::mismatch(
    rmm::exec_policy(stream), rev_node_it, rev_node_it + num_columns, rev_leaf_nodes_it);
  // the node number that could be the leftmost leaf node is given by u = *(is_leftmost_leaf.second
  // - 1)
  NodeIndexT leftmost_leaf_node = leaf_nodes.element(
    num_leaf_nodes - thrust::distance(rev_leaf_nodes_it, is_leftmost_leaf.second - 1) - 1, stream);

  // upper_bound search for u in rowidx for parent node v. Now check if any of the other child nodes
  // of v is non-leaf i.e check if u is the first child of v. If yes, then leafmost_leaf_node is
  // the leftmost leaf node. Otherwise, discard all children of v after and including u

  auto parent_it =
    thrust::upper_bound(rmm::exec_policy(stream), rowidx.begin(), rowidx.end(), leftmost_leaf_node);
  NodeIndexT parent           = thrust::distance(rowidx.begin(), parent_it - 1);
  NodeIndexT parent_adj_start = rowidx.element(parent, stream);
  NodeIndexT parent_adj_end   = rowidx.element(parent + 1, stream);
  auto childnum_it            = thrust::lower_bound(rmm::exec_policy(stream),
                                         colidx.begin() + parent_adj_start,
                                         colidx.begin() + parent_adj_end,
                                         leftmost_leaf_node);

  auto retained_leaf_nodes_it = leaf_nodes.begin() + num_leaf_nodes -
                                thrust::distance(rev_leaf_nodes_it, is_leftmost_leaf.second - 1) -
                                1;
  if (childnum_it != colidx.begin() + parent_adj_start + 1) {
    // discarding from u to last child of parent
    retained_leaf_nodes_it += thrust::distance(childnum_it, colidx.begin() + parent_adj_end);
  }
  // now, all nodes from leaf_nodes.begin() to retained_leaf_nodes_it need to be discarded i.e. they
  // are part of ignore_vals

  // (Optional?) TODO: Validation of the remaining column tree properties

  rmm::device_uvector<NodeIndexT> outcol_nodes(num_columns, stream);
  return device_json_column_properties_size{std::move(outcol_nodes)};
}

/**
 * @brief Reduces node tree representation to column tree CSR representation.
 *
 * @param tree Node tree representation of JSON string
 * @param original_col_ids Column ids of nodes
 * @param sorted_col_ids Sorted column ids of nodes
 * @param ordered_node_ids Node ids of nodes sorted by column ids
 * @param row_offsets Row offsets of nodes
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param row_array_parent_col_id Column id of row array, if is_array_of_arrays is true
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A tuple of column tree representation of JSON string, column ids of columns, and
 * max row offsets of columns
 */
std::tuple<column_tree_csr, rmm::device_uvector<size_type>> reduce_to_column_tree_csr(
  tree_meta_t& tree,
  device_span<NodeIndexT> original_col_ids,
  device_span<NodeIndexT> sorted_col_ids,
  device_span<NodeIndexT> ordered_node_ids,
  device_span<size_type> row_offsets,
  bool is_array_of_arrays,
  NodeIndexT const row_array_parent_col_id,
  cudf::io::json_reader_options const& reader_options,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // 1. column count for allocation
  auto const num_columns =
    thrust::unique_count(rmm::exec_policy(stream), sorted_col_ids.begin(), sorted_col_ids.end());

  rmm::device_uvector<size_type> unique_node_ids(num_columns, stream);
  rmm::device_uvector<size_type> csr_unique_node_ids(num_columns, stream);
  rmm::device_uvector<TreeDepthT> column_levels(num_columns, stream);
  thrust::unique_by_key_copy(rmm::exec_policy(stream),
                             sorted_col_ids.begin(),
                             sorted_col_ids.end(),
                             ordered_node_ids.begin(),
                             thrust::make_discard_iterator(),
                             unique_node_ids.begin());
  thrust::copy_n(
    rmm::exec_policy(stream),
    thrust::make_permutation_iterator(tree.node_levels.begin(), unique_node_ids.begin()),
    unique_node_ids.size(),
    column_levels.begin());
  auto [sorted_column_levels, sorted_column_levels_order] =
    cudf::io::json::detail::stable_sorted_key_order<size_t, TreeDepthT>(column_levels, stream);

  // 2. reduce_by_key {col_id}, {row_offset}, max.
  rmm::device_uvector<NodeIndexT> unique_col_ids(num_columns, stream);
  rmm::device_uvector<size_type> max_row_offsets(num_columns, stream);
  rmm::device_uvector<NodeIndexT> csr_unique_col_ids(num_columns, stream);
  rmm::device_uvector<size_type> csr_max_row_offsets(num_columns, stream);
  auto ordered_row_offsets =
    thrust::make_permutation_iterator(row_offsets.begin(), ordered_node_ids.begin());
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        sorted_col_ids.begin(),
                        sorted_col_ids.end(),
                        ordered_row_offsets,
                        unique_col_ids.begin(),
                        max_row_offsets.begin(),
                        thrust::equal_to<size_type>(),
                        thrust::maximum<size_type>());

  // 3. reduce_by_key {col_id}, {node_categories} - custom opp (*+v=*, v+v=v, *+#=E)
  rmm::device_uvector<NodeT> column_categories(num_columns, stream);
  rmm::device_uvector<NodeT> csr_column_categories(num_columns, stream);
  thrust::reduce_by_key(
    rmm::exec_policy(stream),
    sorted_col_ids.begin(),
    sorted_col_ids.end(),
    thrust::make_permutation_iterator(tree.node_categories.begin(), ordered_node_ids.begin()),
    unique_col_ids.begin(),
    column_categories.begin(),
    thrust::equal_to<size_type>(),
    [] __device__(NodeT type_a, NodeT type_b) -> NodeT {
      auto is_a_leaf = (type_a == NC_VAL || type_a == NC_STR);
      auto is_b_leaf = (type_b == NC_VAL || type_b == NC_STR);
      // (v+v=v, *+*=*,  *+v=*, *+#=E, NESTED+VAL=NESTED)
      // *+*=*, v+v=v
      if (type_a == type_b) {
        return type_a;
      } else if (is_a_leaf) {
        // *+v=*, N+V=N
        // STRUCT/LIST + STR/VAL = STRUCT/LIST, STR/VAL + FN = ERR, STR/VAL + STR = STR
        return type_b == NC_FN ? NC_ERR : (is_b_leaf ? NC_STR : type_b);
      } else if (is_b_leaf) {
        return type_a == NC_FN ? NC_ERR : (is_a_leaf ? NC_STR : type_a);
      }
      // *+#=E
      return NC_ERR;
    });

  auto csr_permutation_it = thrust::make_zip_iterator(
    thrust::make_permutation_iterator(unique_node_ids.begin(), sorted_column_levels_order.begin()),
    thrust::make_permutation_iterator(unique_col_ids.begin(), sorted_column_levels_order.begin()),
    thrust::make_permutation_iterator(max_row_offsets.begin(), sorted_column_levels_order.begin()),
    thrust::make_permutation_iterator(column_categories.begin(),
                                      sorted_column_levels_order.begin()));
  thrust::copy(rmm::exec_policy(stream),
               csr_permutation_it,
               csr_permutation_it + num_columns,
               thrust::make_zip_iterator(csr_unique_node_ids.begin(),
                                         csr_unique_col_ids.begin(),
                                         csr_max_row_offsets.begin(),
                                         csr_column_categories.begin()));

  // 4. unique_copy parent_node_ids, ranges
  rmm::device_uvector<NodeIndexT> csr_parent_col_ids(num_columns, stream);
  rmm::device_uvector<SymbolOffsetT> csr_col_range_begin(num_columns, stream);  // Field names
  rmm::device_uvector<SymbolOffsetT> csr_col_range_end(num_columns, stream);
  thrust::copy_n(
    rmm::exec_policy(stream),
    thrust::make_zip_iterator(
      thrust::make_permutation_iterator(tree.parent_node_ids.begin(), csr_unique_node_ids.begin()),
      thrust::make_permutation_iterator(tree.node_range_begin.begin(), csr_unique_node_ids.begin()),
      thrust::make_permutation_iterator(tree.node_range_end.begin(), csr_unique_node_ids.begin())),
    csr_unique_node_ids.size(),
    thrust::make_zip_iterator(
      csr_parent_col_ids.begin(), csr_col_range_begin.begin(), csr_col_range_end.begin()));

  // convert parent_node_ids to parent_col_ids
  thrust::transform(
    rmm::exec_policy(stream),
    csr_parent_col_ids.begin(),
    csr_parent_col_ids.end(),
    csr_parent_col_ids.begin(),
    [col_ids = original_col_ids.begin()] __device__(auto parent_node_id) -> size_type {
      return parent_node_id == parent_node_sentinel ? parent_node_sentinel
                                                    : col_ids[parent_node_id];
    });

  /*
    CSR construction:
    1. Sort column levels and get their ordering
    2. For each column node coln iterated according to sorted_column_levels; do
        a. Find nodes that have coln as the parent node -> set adj_coln
        b. row idx[coln] = size of adj_coln + 1
        c. col idx[coln] = adj_coln U {parent_col_id[coln]}
  */

  rmm::device_uvector<NodeIndexT> rowidx(num_columns + 1, stream);
  thrust::fill(rmm::exec_policy(stream), rowidx.begin(), rowidx.end(), 0);

  // Note that the first element of csr_parent_col_ids is -1 (parent_node_sentinel)
  // children adjacency
  auto num_non_leaf_columns = thrust::unique_count(
    rmm::exec_policy(stream), csr_parent_col_ids.begin() + 1, csr_parent_col_ids.end());
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        csr_parent_col_ids.begin() + 1,
                        csr_parent_col_ids.end(),
                        thrust::make_constant_iterator(1),
                        thrust::make_discard_iterator(),
                        rowidx.begin() + 1,
                        thrust::equal_to<TreeDepthT>());
  thrust::inclusive_scan(
    rmm::exec_policy(stream), rowidx.begin() + 1, rowidx.end(), rowidx.begin() + 1);
  // overwrite the csr_parent_col_ids with the col ids in the csr tree
  thrust::fill(rmm::exec_policy(stream), csr_parent_col_ids.begin(), csr_parent_col_ids.end(), -1);
  thrust::scatter(rmm::exec_policy(stream),
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + num_non_leaf_columns,
                  rowidx.begin(),
                  csr_parent_col_ids.begin() + 1);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         csr_parent_col_ids.begin(),
                         csr_parent_col_ids.end(),
                         csr_parent_col_ids.begin(),
                         thrust::maximum<NodeIndexT>{});
  // We are discarding the parent of the root node. Add the parent adjacency. Since we have already
  // performed the scan, we use a counting iterator to add
  thrust::transform(rmm::exec_policy(stream),
                    rowidx.begin() + 2,
                    rowidx.end(),
                    thrust::make_counting_iterator(1),
                    rowidx.begin() + 2,
                    thrust::plus<NodeIndexT>());

  rmm::device_uvector<NodeIndexT> colidx((num_columns - 1) * 2, stream);
  thrust::fill(rmm::exec_policy(stream), colidx.begin(), colidx.end(), 0);
  // Skip the parent of root node
  thrust::scatter(rmm::exec_policy(stream),
                  csr_parent_col_ids.begin() + 1,
                  csr_parent_col_ids.end(),
                  rowidx.begin() + 1,
                  colidx.begin());
  // excluding root node
  rmm::device_uvector<NodeIndexT> map(num_columns - 1, stream);
  thrust::fill(rmm::exec_policy(stream), map.begin(), map.end(), 1);
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                csr_parent_col_ids.begin() + 1,
                                csr_parent_col_ids.end(),
                                map.begin(),
                                map.begin());
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(1),
                   thrust::make_counting_iterator(1) + num_columns - 1,
                   [rowidx             = rowidx.begin(),
                    map                = map.begin(),
                    csr_parent_col_ids = csr_parent_col_ids.begin()] __device__(auto i) {
                     auto csr_parent_col_id = csr_parent_col_ids[i];
                     if (csr_parent_col_id == 0)
                       map[i - 1]--;
                     else
                       map[i - 1] += rowidx[csr_parent_col_id];
                   });
  thrust::scatter(rmm::exec_policy(stream),
                  thrust::make_counting_iterator(1),
                  thrust::make_counting_iterator(1) + num_columns - 1,
                  map.begin(),
                  colidx.begin());

  // condition is true if parent is not a list, or sentinel/root
  // Special case to return true if parent is a list and is_array_of_arrays is true
  auto is_non_list_parent = [column_categories = column_categories.begin(),
                             is_array_of_arrays,
                             row_array_parent_col_id] __device__(auto parent_col_id) -> bool {
    return !(parent_col_id == parent_node_sentinel ||
             column_categories[parent_col_id] == NC_LIST &&
               (!is_array_of_arrays || parent_col_id != row_array_parent_col_id));
  };
  // Mixed types in List children go to different columns,
  // so all immediate children of list column should have same max_row_offsets.
  //   create list's children max_row_offsets array. (initialize to zero)
  //   atomicMax on  children max_row_offsets array.
  //   gather the max_row_offsets from children row offset array.
  {
    rmm::device_uvector<NodeIndexT> list_parents_children_max_row_offsets(num_columns, stream);
    thrust::fill(rmm::exec_policy(stream),
                 list_parents_children_max_row_offsets.begin(),
                 list_parents_children_max_row_offsets.end(),
                 0);
    thrust::for_each(rmm::exec_policy(stream),
                     csr_unique_col_ids.begin(),
                     csr_unique_col_ids.end(),
                     [csr_column_categories = csr_column_categories.begin(),
                      csr_parent_col_ids    = csr_parent_col_ids.begin(),
                      csr_max_row_offsets   = csr_max_row_offsets.begin(),
                      list_parents_children_max_row_offsets =
                        list_parents_children_max_row_offsets.begin()] __device__(auto col_id) {
                       auto csr_parent_col_id = csr_parent_col_ids[col_id];
                       if (csr_parent_col_id != parent_node_sentinel and
                           csr_column_categories[csr_parent_col_id] == node_t::NC_LIST) {
                         cuda::atomic_ref<NodeIndexT, cuda::thread_scope_device> ref{
                           *(list_parents_children_max_row_offsets + csr_parent_col_id)};
                         ref.fetch_max(csr_max_row_offsets[col_id],
                                       cuda::std::memory_order_relaxed);
                       }
                     });
    thrust::gather_if(
      rmm::exec_policy(stream),
      csr_parent_col_ids.begin(),
      csr_parent_col_ids.end(),
      csr_parent_col_ids.begin(),
      list_parents_children_max_row_offsets.begin(),
      csr_max_row_offsets.begin(),
      [csr_column_categories = csr_column_categories.begin()] __device__(size_type parent_col_id) {
        return parent_col_id != parent_node_sentinel and
               csr_column_categories[parent_col_id] == node_t::NC_LIST;
      });
  }

  // copy lists' max_row_offsets to children.
  // all structs should have same size.
  thrust::transform_if(
    rmm::exec_policy(stream),
    csr_unique_col_ids.begin(),
    csr_unique_col_ids.end(),
    csr_max_row_offsets.begin(),
    [csr_column_categories = csr_column_categories.begin(),
     is_non_list_parent,
     csr_parent_col_ids  = csr_parent_col_ids.begin(),
     csr_max_row_offsets = csr_max_row_offsets.begin()] __device__(size_type col_id) {
      auto parent_col_id = csr_parent_col_ids[col_id];
      // condition is true if parent is not a list, or sentinel/root
      while (is_non_list_parent(parent_col_id)) {
        col_id        = parent_col_id;
        parent_col_id = csr_parent_col_ids[parent_col_id];
      }
      return csr_max_row_offsets[col_id];
    },
    [csr_column_categories = csr_column_categories.begin(),
     is_non_list_parent,
     parent_col_ids = csr_parent_col_ids.begin()] __device__(size_type col_id) {
      auto parent_col_id = parent_col_ids[col_id];
      // condition is true if parent is not a list, or sentinel/root
      return is_non_list_parent(parent_col_id);
    });

  // For Struct and List (to avoid copying entire strings when mixed type as string is enabled)
  thrust::transform_if(
    rmm::exec_policy(stream),
    csr_col_range_begin.begin(),
    csr_col_range_begin.end(),
    csr_column_categories.begin(),
    csr_col_range_end.begin(),
    [] __device__(auto i) { return i + 1; },
    [] __device__(NodeT type) { return type == NC_STRUCT || type == NC_LIST; });

  auto size_estimates =
    estimate_device_json_column_size(rowidx, colidx, csr_column_categories, reader_options, stream);

  return std::tuple{column_tree_csr{std::move(rowidx),
                                    std::move(colidx),
                                    std::move(csr_unique_col_ids),
                                    std::move(csr_column_categories),
                                    std::move(csr_col_range_begin),
                                    std::move(csr_col_range_end)},
                    std::move(csr_max_row_offsets)};
}

/**
 * @brief Constructs `d_json_column` from node tree representation
 * Newly constructed columns are insert into `root`'s children.
 * `root` must be a list type.
 *
 * @param input Input JSON string device data
 * @param tree Node tree representation of the JSON string
 * @param col_ids Column ids of the nodes in the tree
 * @param row_offsets Row offsets of the nodes in the tree
 * @param root Root node of the `d_json_column` tree
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param options Parsing options specifying the parsing behaviour
 * options affecting behaviour are
 *   is_enabled_lines: Whether the input is a line-delimited JSON
 *   is_enabled_mixed_types_as_string: Whether to enable reading mixed types as string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the device memory
 * of child_offets and validity members of `d_json_column`
 */
void make_device_json_column_csr(device_span<SymbolT const> input,
                                 tree_meta_t& tree,
                                 device_span<NodeIndexT> col_ids,
                                 device_span<size_type> row_offsets,
                                 device_json_column& root,
                                 bool is_array_of_arrays,
                                 cudf::io::json_reader_options const& options,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  bool const is_enabled_lines = options.is_enabled_lines();
  auto const num_nodes        = col_ids.size();
  rmm::device_uvector<NodeIndexT> sorted_col_ids(col_ids.size(), stream);  // make a copy
  thrust::copy(rmm::exec_policy(stream), col_ids.begin(), col_ids.end(), sorted_col_ids.begin());

  // sort by {col_id} on {node_ids} stable
  rmm::device_uvector<NodeIndexT> node_ids(col_ids.size(), stream);
  thrust::sequence(rmm::exec_policy(stream), node_ids.begin(), node_ids.end());
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), sorted_col_ids.begin(), sorted_col_ids.end(), node_ids.begin());

  NodeIndexT const row_array_parent_col_id = [&]() {
    NodeIndexT value = parent_node_sentinel;
    if (!col_ids.empty()) {
      auto const list_node_index = is_enabled_lines ? 0 : 1;
      CUDF_CUDA_TRY(cudaMemcpyAsync(&value,
                                    col_ids.data() + list_node_index,
                                    sizeof(NodeIndexT),
                                    cudaMemcpyDefault,
                                    stream.value()));
      stream.synchronize();
    }
    return value;
  }();

  // 1. gather column information.
  auto [d_column_tree, d_max_row_offsets] = reduce_to_column_tree_csr(tree,
                                                                      col_ids,
                                                                      sorted_col_ids,
                                                                      node_ids,
                                                                      row_offsets,
                                                                      is_array_of_arrays,
                                                                      row_array_parent_col_id,
                                                                      options,
                                                                      stream);

  CUDF_EXPECTS(is_array_of_arrays == false, "array of arrays has not yet been implemented");
  CUDF_EXPECTS(options.is_enabled_mixed_types_as_string() == false,
               "mixed type as string has not yet been implemented");
  CUDF_EXPECTS(options.is_enabled_prune_columns() == false,
               "column pruning has not yet been implemented");
}

}  // namespace cudf::io::json::experimental::detail
