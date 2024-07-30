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

#include <cub/device/device_segmented_reduce.cuh>
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
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

namespace cudf::io::json::experimental::detail {

using row_offset_t = size_type;

struct level_ordering {
  device_span<TreeDepthT> node_levels;
  device_span<NodeIndexT> col_ids;
  __device__ bool operator()(NodeIndexT lhs_node_id, NodeIndexT rhs_node_id) const
  {
    return (node_levels[lhs_node_id] < node_levels[rhs_node_id]) ||
           (node_levels[lhs_node_id] == node_levels[rhs_node_id] &&
            col_ids[lhs_node_id] < col_ids[rhs_node_id]);
  }
};

struct parent_nodeids_to_colids {
  device_span<NodeIndexT> col_ids;
  device_span<NodeIndexT> rev_mapped_col_ids;
  __device__ auto operator()(NodeIndexT parent_node_id) -> NodeIndexT
  {
    return parent_node_id == parent_node_sentinel ? parent_node_sentinel
                                                  : rev_mapped_col_ids[col_ids[parent_node_id]];
  }
};

template <typename T>
void print(device_span<T const> d_vec, std::string name, rmm::cuda_stream_view stream)
{
  auto h_vec = cudf::detail::make_std_vector_async(d_vec, stream);
  std::cout << name << " = ";
  for (auto e : h_vec) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
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
std::tuple<csr, column_tree_properties> reduce_to_column_tree(
  tree_meta_t& tree,
  device_span<NodeIndexT> col_ids,
  device_span<row_offset_t> row_offsets,
  bool is_array_of_arrays,
  NodeIndexT const row_array_parent_col_id,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  rmm::device_uvector<NodeIndexT> level_ordered_col_ids(col_ids.size(), stream);
  rmm::device_uvector<NodeIndexT> level_ordered_node_ids(col_ids.size(), stream);
  thrust::copy(
    rmm::exec_policy_nosync(stream), col_ids.begin(), col_ids.end(), level_ordered_col_ids.begin());
  thrust::sequence(
    rmm::exec_policy_nosync(stream), level_ordered_node_ids.begin(), level_ordered_node_ids.end());

  // Reorder nodes and column ids in level-wise fashion
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                             level_ordered_node_ids.begin(),
                             level_ordered_node_ids.end(),
                             level_ordered_col_ids.begin(),
                             level_ordering{tree.node_levels, col_ids});

  // 1. get the number of columns in tree, mapping between node tree col ids and csr col ids, and
  // the node id of first row in each column
  auto const num_columns = thrust::unique_count(
    rmm::exec_policy_nosync(stream), level_ordered_col_ids.begin(), level_ordered_col_ids.end());
  rmm::device_uvector<NodeIndexT> level_ordered_unique_node_ids(num_columns, stream);
  rmm::device_uvector<NodeIndexT> mapped_col_ids(num_columns, stream);
  rmm::device_uvector<NodeIndexT> rev_mapped_col_ids(num_columns, stream);
  thrust::unique_by_key_copy(rmm::exec_policy_nosync(stream),
                             level_ordered_col_ids.begin(),
                             level_ordered_node_ids.end(),
                             level_ordered_node_ids.begin(),
                             mapped_col_ids.begin(),
                             level_ordered_unique_node_ids.begin());
  auto* dev_num_levels_ptr =
    thrust::max_element(rmm::exec_policy(stream), tree.node_levels.begin(), tree.node_levels.end());

  rmm::device_uvector<NodeIndexT> mapped_col_ids_copy(num_columns, stream);
  thrust::copy(rmm::exec_policy(stream),
               mapped_col_ids.begin(),
               mapped_col_ids.end(),
               mapped_col_ids_copy.begin());
  thrust::sequence(rmm::exec_policy(stream), rev_mapped_col_ids.begin(), rev_mapped_col_ids.end());
  thrust::sort_by_key(rmm::exec_policy(stream),
                      mapped_col_ids_copy.begin(),
                      mapped_col_ids_copy.end(),
                      rev_mapped_col_ids.begin());

  // 2. maximum number of rows per column: computed with reduce_by_key {col_id}, {row_offset}, max.
  // 3. category for each column node by aggregating all nodes in node tree corresponding to same
  // column:
  //    reduce_by_key {col_id}, {node_categories} - custom opp (*+v=*, v+v=v, *+#=E)
  rmm::device_uvector<row_offset_t> max_row_offsets(num_columns, stream);
  rmm::device_uvector<NodeT> column_categories(num_columns, stream);
  auto ordered_row_offsets =
    thrust::make_permutation_iterator(row_offsets.begin(), level_ordered_node_ids.begin());
  auto ordered_node_categories =
    thrust::make_permutation_iterator(tree.node_categories.begin(), level_ordered_node_ids.begin());
  thrust::reduce_by_key(
    rmm::exec_policy(stream),
    level_ordered_col_ids.begin(),
    level_ordered_col_ids.end(),
    thrust::make_zip_iterator(thrust::make_tuple(ordered_row_offsets, ordered_node_categories)),
    thrust::make_discard_iterator(),
    thrust::make_zip_iterator(
      thrust::make_tuple(max_row_offsets.begin(), column_categories.begin())),
    thrust::equal_to<NodeIndexT>(),
    [] __device__(auto a, auto b) {
      auto row_offset_a = thrust::get<0>(a);
      auto row_offset_b = thrust::get<0>(b);
      auto type_a       = thrust::get<1>(a);
      auto type_b       = thrust::get<1>(b);

      NodeT ctg;
      auto is_a_leaf = (type_a == NC_VAL || type_a == NC_STR);
      auto is_b_leaf = (type_b == NC_VAL || type_b == NC_STR);
      // (v+v=v, *+*=*,  *+v=*, *+#=E, NESTED+VAL=NESTED)
      // *+*=*, v+v=v
      if (type_a == type_b) {
        ctg = type_a;
      } else if (is_a_leaf) {
        // *+v=*, N+V=N
        // STRUCT/LIST + STR/VAL = STRUCT/LIST, STR/VAL + FN = ERR, STR/VAL + STR = STR
        ctg = (type_b == NC_FN ? NC_ERR : (is_b_leaf ? NC_STR : type_b));
      } else if (is_b_leaf) {
        ctg = (type_a == NC_FN ? NC_ERR : (is_a_leaf ? NC_STR : type_a));
      } else
        ctg = NC_ERR;

      thrust::maximum<size_type> row_offset_op;
      return thrust::make_tuple(row_offset_op(row_offset_a, row_offset_b), ctg);
    });

  // 4. construct parent_col_ids using permutation iterator
  rmm::device_uvector<NodeIndexT> parent_col_ids(num_columns, stream);
  thrust::transform_output_iterator parent_col_ids_it(
    parent_col_ids.begin(), parent_nodeids_to_colids{col_ids, rev_mapped_col_ids});
  thrust::copy_n(rmm::exec_policy(stream),
                 thrust::make_permutation_iterator(tree.parent_node_ids.begin(),
                                                   level_ordered_unique_node_ids.begin()),
                 num_columns,
                 parent_col_ids_it);

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
  thrust::transform_inclusive_scan(
    rmm::exec_policy(stream),
    thrust::make_zip_iterator(thrust::make_counting_iterator(1), rowidx.begin() + 1),
    thrust::make_zip_iterator(thrust::make_counting_iterator(1) + num_columns, rowidx.end()),
    rowidx.begin() + 1,
    cuda::proclaim_return_type<NodeIndexT>([] __device__(auto a) {
      auto n   = thrust::get<0>(a);
      auto idx = thrust::get<1>(a);
      return n == 1 ? idx : idx + 1;
    }),
    thrust::plus<NodeIndexT>{});

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
                     [rowidx         = rowidx.begin(),
                      map            = map.begin(),
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

  // Mixed types in List children go to different columns,
  // so all immediate children of list column should have same max_row_offsets.
  //   create list's children max_row_offsets array. (initialize to zero)
  //   atomicMax on  children max_row_offsets array.
  //   gather the max_row_offsets from children row offset array.
  {
    auto max_row_offsets_it =
      thrust::make_permutation_iterator(max_row_offsets.begin(), colidx.begin());
    rmm::device_uvector<row_offset_t> max_children_max_row_offsets(num_columns, stream);
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Max(nullptr,
                                    temp_storage_bytes,
                                    max_row_offsets_it,
                                    max_children_max_row_offsets.begin(),
                                    num_columns,
                                    rowidx.begin(),
                                    rowidx.begin() + 1,
                                    stream.value());
    rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);
    cub::DeviceSegmentedReduce::Max(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    max_row_offsets_it,
                                    max_children_max_row_offsets.begin(),
                                    num_columns,
                                    rowidx.begin(),
                                    rowidx.begin() + 1,
                                    stream.value());

    rmm::device_uvector<NodeIndexT> list_ancestors(num_columns, stream);
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      num_columns,
      [rowidx            = rowidx.begin(),
       colidx            = colidx.begin(),
       column_categories = column_categories.begin(),
       dev_num_levels_ptr,
       list_ancestors = list_ancestors.begin()] __device__(NodeIndexT node) {
        auto num_levels      = *dev_num_levels_ptr;
        list_ancestors[node] = node;
        for (int level = 0; level < num_levels; level++) {
          if (list_ancestors[node] > 0)
            list_ancestors[node] = colidx[rowidx[list_ancestors[node]]];
          else
            list_ancestors[node] = -1;
          if (list_ancestors[node] == -1 || column_categories[list_ancestors[node]] == NC_LIST)
            break;
        }
      });
    thrust::gather_if(rmm::exec_policy(stream),
                      list_ancestors.begin(),
                      list_ancestors.end(),
                      list_ancestors.begin(),
                      max_children_max_row_offsets.begin(),
                      max_row_offsets.begin(),
                      [] __device__(auto ancestor) { return ancestor != -1; });
  }

  return std::tuple{
    csr{std::move(rowidx), std::move(colidx)},
    column_tree_properties{
      std::move(column_categories), std::move(max_row_offsets), std::move(mapped_col_ids)}};
}

}  // namespace cudf::io::json::experimental::detail
