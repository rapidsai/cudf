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
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace cudf::io::json::experimental::detail {

using row_offset_t = size_type;

struct unvalidated_column_tree {
  rmm::device_uvector<NodeIndexT> rowidx;
  rmm::device_uvector<NodeIndexT> colidx;
  rmm::device_uvector<row_offset_t> max_row_offsets;
  rmm::device_uvector<NodeT> column_categories;
};

struct level_ordering {
  device_span<TreeDepthT> node_levels;
  device_span<NodeIndexT> col_ids;
  __device__ bool operator()(NodeIndexT lhs_node_id, NodeIndexT rhs_node_id) const
  {
    return (node_levels[lhs_node_id] < node_levels[rhs_node_id]) ||
      (node_levels[lhs_node_id] == node_levels[rhs_node_id] && col_ids[lhs_node_id] < col_ids[rhs_node_id]);
  }
};

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
unvalidated_column_tree reduce_to_column_tree_csr(
  tree_meta_t& tree,
  device_span<NodeIndexT> col_ids,
  device_span<size_type> row_offsets,
  bool is_array_of_arrays,
  NodeIndexT const row_array_parent_col_id,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  rmm::device_uvector<NodeIndexT> level_ordered_col_ids(col_ids.size(), stream);
  rmm::device_uvector<NodeIndexT> level_ordered_node_ids(col_ids.size(), stream);
  thrust::copy(rmm::exec_policy_nosync(stream), col_ids.begin(), col_ids.end(), level_ordered_col_ids.begin());
  thrust::sequence(rmm::exec_policy_nosync(stream), level_ordered_node_ids.begin(), level_ordered_node_ids.end());

  // Reorder nodes and column ids in level-wise fashion
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream), level_ordered_node_ids.begin(), level_ordered_node_ids.end(), 
      level_ordered_col_ids.begin(), level_ordering{tree.node_levels, col_ids});

  // 1. get the number of columns in tree, mapping between node tree col ids and csr col ids, and the node id of first row in each column 
  auto const num_columns =
    thrust::unique_count(rmm::exec_policy_nosync(stream), level_ordered_col_ids.begin(), level_ordered_col_ids.end());
  rmm::device_uvector<NodeIndexT> level_ordered_unique_node_ids(num_columns, stream);
  rmm::device_uvector<NodeIndexT> mapped_col_ids(num_columns, stream);
  thrust::unique_by_key_copy(rmm::exec_policy_nosync(stream), level_ordered_col_ids.begin(), level_ordered_node_ids.end(), level_ordered_node_ids.begin(), mapped_col_ids.begin(), level_ordered_unique_node_ids.begin());
  auto rev_mapped_col_ids_it = thrust::make_permutation_iterator(thrust::make_counting_iterator(0), mapped_col_ids.begin());

  // 2. maximum number of rows per column: computed with reduce_by_key {col_id}, {row_offset}, max.
  // 3. category for each column node by aggregating all nodes in node tree corresponding to same column:
  //    reduce_by_key {col_id}, {node_categories} - custom opp (*+v=*, v+v=v, *+#=E)
  rmm::device_uvector<size_type> max_row_offsets(num_columns, stream);
  rmm::device_uvector<NodeT> column_categories(num_columns, stream);
  auto ordered_row_offsets =
    thrust::make_permutation_iterator(row_offsets.begin(), level_ordered_node_ids.begin());
  auto ordered_node_categories = thrust::make_permutation_iterator(tree.node_categories.begin(), level_ordered_node_ids.begin());
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        level_ordered_col_ids.begin(),
                        level_ordered_col_ids.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(ordered_row_offsets, ordered_node_categories)),
                        thrust::make_discard_iterator(),
                        thrust::make_zip_iterator(thrust::make_tuple(max_row_offsets.begin(), column_categories.begin())),
                        thrust::equal_to<NodeIndexT>(),
                        [] __device__(auto a, auto b) {
                          auto row_offset_a = thrust::get<0>(a);
                          auto row_offset_b = thrust::get<0>(b);
                          auto type_a = thrust::get<1>(a);
                          auto type_b = thrust::get<1>(b);
                        
                          NodeT max_offset;
                          auto is_a_leaf = (type_a == NC_VAL || type_a == NC_STR);
                          auto is_b_leaf = (type_b == NC_VAL || type_b == NC_STR);
                          // (v+v=v, *+*=*,  *+v=*, *+#=E, NESTED+VAL=NESTED)
                          // *+*=*, v+v=v
                          if (type_a == type_b) {
                            max_offset = type_a;
                          } else if (is_a_leaf) {
                            // *+v=*, N+V=N
                            // STRUCT/LIST + STR/VAL = STRUCT/LIST, STR/VAL + FN = ERR, STR/VAL + STR = STR
                            max_offset = type_b == NC_FN ? NC_ERR : (is_b_leaf ? NC_STR : type_b);
                          } else if (is_b_leaf) {
                            max_offset = type_a == NC_FN ? NC_ERR : (is_a_leaf ? NC_STR : type_a);
                          }
                          // *+#=E
                          max_offset = NC_ERR;

                          thrust::maximum<size_type> row_offset_op;
                          return thrust::make_tuple(row_offset_op(row_offset_a, row_offset_b), max_offset);
                        });

  // 4. construct parent_col_ids using permutation iterator
  rmm::device_uvector<NodeIndexT> parent_col_ids(num_columns, stream);
  thrust::copy_n(
    rmm::exec_policy(stream),
    thrust::make_permutation_iterator(tree.parent_node_ids.begin(), level_ordered_unique_node_ids.begin()),
    num_columns,
    thrust::make_transform_output_iterator(parent_col_ids.begin(), 
      [col_ids = col_ids.begin(), rev_mapped_col_ids_it] __device__(auto parent_node_id) -> NodeIndexT {
        return parent_node_id == parent_node_sentinel ? parent_node_sentinel : rev_mapped_col_ids_it[col_ids[parent_node_id]];
  }));

  /*
    5. CSR construction:
      a. Sort column levels and get their ordering
      b. For each column node coln iterated according to sorted_column_levels; do
          i. Find nodes that have coln as the parent node -> set adj_coln
          ii. row idx[coln] = size of adj_coln + 1
          iii. col idx[coln] = adj_coln U {parent_col_id[coln]}
  */

  rmm::device_uvector<NodeIndexT> rowidx(num_columns + 1, stream);
  thrust::fill(rmm::exec_policy(stream), rowidx.begin(), rowidx.end(), 0);

  // Note that the first element of csr_parent_col_ids is -1 (parent_node_sentinel)
  // children adjacency
  auto num_non_leaf_columns = thrust::unique_count(
    rmm::exec_policy(stream), parent_col_ids.begin() + 1, parent_col_ids.end());
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        parent_col_ids.begin() + 1,
                        parent_col_ids.end(),
                        thrust::make_constant_iterator(1),
                        thrust::make_discard_iterator(),
                        rowidx.begin() + 1,
                        thrust::equal_to<TreeDepthT>());
  thrust::inclusive_scan(
    rmm::exec_policy(stream), rowidx.begin() + 1, rowidx.end(), rowidx.begin() + 1);
  // We are discarding the parent of the root node. Add the parent adjacency. Since we have already
  // performed the scan, we use a counting iterator to add
  thrust::transform(rmm::exec_policy(stream),
                    rowidx.begin() + 2,
                    rowidx.end(),
                    thrust::make_counting_iterator(1),
                    rowidx.begin() + 2,
                    thrust::plus<NodeIndexT>());

  rmm::device_uvector<NodeIndexT> colidx((num_columns - 1) * 2, stream);

  // Skip the parent of root node
  thrust::scatter(rmm::exec_policy(stream),
                  parent_col_ids.begin() + 1,
                  parent_col_ids.end(),
                  rowidx.begin() + 1,
                  colidx.begin());
  // excluding root node, construct scatter map
  rmm::device_uvector<NodeIndexT> map(num_columns - 1, stream);
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                parent_col_ids.begin() + 1,
                                parent_col_ids.end(), 
                                thrust::make_constant_iterator(1),
                                map.begin());
  thrust::for_each_n(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(1),
                   num_columns - 1,
                   [rowidx             = rowidx.begin(),
                    map                = map.begin(),
                    parent_col_ids = parent_col_ids.begin()] __device__(auto i) {
                     auto parent_col_id = parent_col_ids[i];
                     if (parent_col_id == 0)
                       map[i - 1]--;
                     else
                       map[i - 1] += rowidx[parent_col_id];
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
    auto list_nodes = thrust::make_permutation_iterator

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

  return std::tuple{column_tree_csr{std::move(rowidx),
                                    std::move(colidx),
                                    std::move(csr_unique_col_ids),
                                    std::move(csr_column_categories),
                                    std::move(csr_col_range_begin),
                                    std::move(csr_col_range_end)},
                    std::move(csr_max_row_offsets)};
}

}  // namespace cudf::io::json::experimental::detail
