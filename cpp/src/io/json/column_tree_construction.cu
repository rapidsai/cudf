/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

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

namespace cudf::io::json {

using row_offset_t = size_type;

// debug printing
#ifndef CSR_DEBUG_PRINT
// #define CSR_DEBUG_PRINT
#endif

#ifdef CSR_DEBUG_PRINT
template <typename T>
void print(device_span<T const> d_vec, std::string name, rmm::cuda_stream_view stream)
{
  stream.synchronize();
  auto h_vec = cudf::detail::make_std_vector_sync(d_vec, stream);
  std::cout << name << " = ";
  for (auto e : h_vec) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
}
#endif

namespace experimental::detail {

struct level_ordering {
  device_span<TreeDepthT const> node_levels;
  device_span<NodeIndexT const> col_ids;
  device_span<NodeIndexT const> parent_node_ids;
  __device__ bool operator()(NodeIndexT lhs_node_id, NodeIndexT rhs_node_id) const
  {
    auto lhs_parent_col_id =
      parent_node_ids[lhs_node_id] == -1 ? -1 : col_ids[parent_node_ids[lhs_node_id]];
    auto rhs_parent_col_id =
      parent_node_ids[rhs_node_id] == -1 ? -1 : col_ids[parent_node_ids[rhs_node_id]];

    return (node_levels[lhs_node_id] < node_levels[rhs_node_id]) ||
           (node_levels[lhs_node_id] == node_levels[rhs_node_id] &&
            lhs_parent_col_id < rhs_parent_col_id) ||
           (node_levels[lhs_node_id] == node_levels[rhs_node_id] &&
            lhs_parent_col_id == rhs_parent_col_id && col_ids[lhs_node_id] < col_ids[rhs_node_id]);
  }
};

struct parent_nodeids_to_colids {
  device_span<NodeIndexT const> col_ids;
  device_span<NodeIndexT const> rev_mapped_col_ids;
  __device__ auto operator()(NodeIndexT parent_node_id) -> NodeIndexT
  {
    return parent_node_id == parent_node_sentinel ? parent_node_sentinel
                                                  : rev_mapped_col_ids[col_ids[parent_node_id]];
  }
};

/**
 * @brief Reduces node tree representation to column tree CSR representation.
 *
 * @param tree Node tree representation of JSON string
 * @param original_col_ids Column ids of nodes
 * @param row_offsets Row offsets of nodes
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param row_array_parent_col_id Column id of row array, if is_array_of_arrays is true
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A tuple of column tree representation of JSON string, column ids of columns, and
 * max row offsets of columns
 */
std::tuple<compressed_sparse_row, column_tree_properties> reduce_to_column_tree(
  tree_meta_t& tree,
  device_span<NodeIndexT const> col_ids,
  device_span<row_offset_t const> row_offsets,
  bool is_array_of_arrays,
  NodeIndexT row_array_parent_col_id,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (col_ids.empty()) {
    rmm::device_uvector<NodeIndexT> empty_rowidx(0, stream);
    rmm::device_uvector<NodeIndexT> empty_colidx(0, stream);
    rmm::device_uvector<NodeT> empty_column_categories(0, stream);
    rmm::device_uvector<row_offset_t> empty_max_row_offsets(0, stream);
    rmm::device_uvector<NodeIndexT> empty_mapped_col_ids(0, stream);
    return std::tuple{compressed_sparse_row{std::move(empty_rowidx), std::move(empty_colidx)},
                      column_tree_properties{std::move(empty_column_categories),
                                             std::move(empty_max_row_offsets),
                                             std::move(empty_mapped_col_ids)}};
  }

  auto level_orderings = [&tree, &col_ids, &stream]() {
    NodeIndexT num_columns;
    auto level_ordered_col_ids = cudf::detail::make_device_uvector_async(
      col_ids, stream, cudf::get_current_device_resource_ref());
    rmm::device_uvector<NodeIndexT> level_ordered_node_ids(col_ids.size(), stream);

    thrust::sequence(rmm::exec_policy_nosync(stream),
                     level_ordered_node_ids.begin(),
                     level_ordered_node_ids.end());
    // Reorder nodes and column ids in level-wise fashion
    thrust::sort_by_key(rmm::exec_policy_nosync(stream),
                        level_ordered_node_ids.begin(),
                        level_ordered_node_ids.end(),
                        level_ordered_col_ids.begin(),
                        level_ordering{tree.node_levels, col_ids, tree.parent_node_ids});

#ifdef CSR_DEBUG_PRINT
    print<NodeIndexT>(level_ordered_node_ids, "h_level_ordered_node_ids", stream);
    print<NodeIndexT>(col_ids, "h_col_ids", stream);
    print<NodeIndexT>(level_ordered_col_ids, "h_level_ordered_col_ids", stream);
#endif

    // 1. get the number of columns in tree, mapping between node tree col ids and csr col ids, and
    // the node id of first row in each column
    num_columns = thrust::unique_count(
      rmm::exec_policy_nosync(stream), level_ordered_col_ids.begin(), level_ordered_col_ids.end());

    return std::tuple{
      num_columns, std::move(level_ordered_node_ids), std::move(level_ordered_col_ids)};
  };

  auto col_tree_adjs = [&tree, &col_ids, &stream](
                         NodeIndexT num_columns,
                         device_span<NodeIndexT const> level_ordered_col_ids,
                         device_span<NodeIndexT const> level_ordered_node_ids) {
    rmm::device_uvector<NodeIndexT> mapped_col_ids(num_columns, stream);
    rmm::device_uvector<NodeIndexT> parent_col_ids(num_columns, stream);
    rmm::device_uvector<NodeIndexT> rev_mapped_col_ids(num_columns, stream);

    rmm::device_uvector<NodeIndexT> level_ordered_unique_node_ids(num_columns, stream);
    thrust::unique_by_key_copy(rmm::exec_policy_nosync(stream),
                               level_ordered_col_ids.begin(),
                               level_ordered_col_ids.end(),
                               level_ordered_node_ids.begin(),
                               mapped_col_ids.begin(),
                               level_ordered_unique_node_ids.begin());
    auto mapped_col_ids_copy = cudf::detail::make_device_uvector_async(
      mapped_col_ids, stream, cudf::get_current_device_resource_ref());
    thrust::sequence(
      rmm::exec_policy_nosync(stream), rev_mapped_col_ids.begin(), rev_mapped_col_ids.end());
    thrust::sort_by_key(rmm::exec_policy_nosync(stream),
                        mapped_col_ids_copy.begin(),
                        mapped_col_ids_copy.end(),
                        rev_mapped_col_ids.begin());
#ifdef CSR_DEBUG_PRINT
    print<NodeIndexT>(mapped_col_ids, "h_mapped_col_ids", stream);
    print<NodeIndexT>(level_ordered_unique_node_ids, "h_level_ordered_unique_node_ids", stream);
    print<NodeIndexT>(rev_mapped_col_ids, "h_rev_mapped_col_ids", stream);
#endif

    // 4. construct parent_col_ids using permutation iterator
    thrust::transform_output_iterator parent_col_ids_it(
      parent_col_ids.begin(), parent_nodeids_to_colids{col_ids, rev_mapped_col_ids});
    thrust::copy_n(rmm::exec_policy_nosync(stream),
                   thrust::make_permutation_iterator(tree.parent_node_ids.begin(),
                                                     level_ordered_unique_node_ids.begin()),
                   num_columns,
                   parent_col_ids_it);

    return std::tuple{
      std::move(mapped_col_ids), std::move(parent_col_ids), std::move(rev_mapped_col_ids)};
  };

  auto col_tree_props = [&tree, &row_offsets, &stream](
                          NodeIndexT num_columns,
                          device_span<NodeIndexT const> level_ordered_col_ids,
                          device_span<NodeIndexT const> level_ordered_node_ids) {
    rmm::device_uvector<row_offset_t> max_row_offsets(num_columns, stream);
    rmm::device_uvector<NodeT> column_categories(num_columns, stream);

    // 2. maximum number of rows per column: computed with reduce_by_key {col_id}, {row_offset},
    // max.
    // 3. category for each column node by aggregating all nodes in node tree corresponding to same
    // column:
    //    reduce_by_key {col_id}, {node_categories} - custom opp (*+v=*, v+v=v, *+#=E)
    cudf::io::json::detail::max_row_offsets_col_categories(
      level_ordered_col_ids.begin(),
      level_ordered_col_ids.end(),
      thrust::make_zip_iterator(
        thrust::make_permutation_iterator(row_offsets.begin(), level_ordered_node_ids.begin()),
        thrust::make_permutation_iterator(tree.node_categories.begin(),
                                          level_ordered_node_ids.begin())),
      thrust::make_discard_iterator(),
      thrust::make_zip_iterator(max_row_offsets.begin(), column_categories.begin()),
      stream);

    return std::tuple{std::move(max_row_offsets), std::move(column_categories)};
  };

  auto construct_rowidx = [&stream](NodeIndexT num_columns,
                                    device_span<NodeIndexT const> parent_col_ids) {
    auto rowidx = cudf::detail::make_zeroed_device_uvector_async<NodeIndexT>(
      static_cast<std::size_t>(num_columns + 1), stream, cudf::get_current_device_resource_ref());
    // Note that the first element of csr_parent_col_ids is -1 (parent_node_sentinel)
    // children adjacency

#ifdef CSR_DEBUG_PRINT
    print<NodeIndexT>(parent_col_ids, "h_parent_col_ids", stream);
#endif

    auto num_non_leaf_columns = thrust::unique_count(
      rmm::exec_policy_nosync(stream), parent_col_ids.begin() + 1, parent_col_ids.end());
    rmm::device_uvector<NodeIndexT> non_leaf_nodes(num_non_leaf_columns, stream);
    rmm::device_uvector<NodeIndexT> non_leaf_nodes_children(num_non_leaf_columns, stream);
    thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                          parent_col_ids.begin() + 1,
                          parent_col_ids.end(),
                          thrust::make_constant_iterator(1),
                          non_leaf_nodes.begin(),
                          non_leaf_nodes_children.begin(),
                          thrust::equal_to<TreeDepthT>());

    thrust::scatter(rmm::exec_policy_nosync(stream),
                    non_leaf_nodes_children.begin(),
                    non_leaf_nodes_children.end(),
                    non_leaf_nodes.begin(),
                    rowidx.begin() + 1);

    if (num_columns > 1) {
      thrust::transform_inclusive_scan(
        rmm::exec_policy_nosync(stream),
        thrust::make_zip_iterator(thrust::make_counting_iterator(1), rowidx.begin() + 1),
        thrust::make_zip_iterator(thrust::make_counting_iterator(1) + num_columns, rowidx.end()),
        rowidx.begin() + 1,
        cuda::proclaim_return_type<NodeIndexT>([] __device__(auto a) {
          auto n   = thrust::get<0>(a);
          auto idx = thrust::get<1>(a);
          return n == 1 ? idx : idx + 1;
          return idx + 1;
        }),
        thrust::plus<NodeIndexT>{});
    } else {
      auto single_node = 1;
      rowidx.set_element_async(1, single_node, stream);
    }

#ifdef CSR_DEBUG_PRINT
    print<NodeIndexT>(rowidx, "h_rowidx", stream);
#endif
    return rowidx;
  };

  auto partially_construct_colidx = [&stream](NodeIndexT num_columns,
                                              device_span<NodeIndexT const> parent_col_ids,
                                              device_span<NodeIndexT const> rowidx) {
    rmm::device_uvector<NodeIndexT> colidx((num_columns - 1) * 2, stream);
    thrust::fill(rmm::exec_policy_nosync(stream), colidx.begin(), colidx.end(), -1);
    // excluding root node, construct scatter map
    rmm::device_uvector<NodeIndexT> map(num_columns - 1, stream);
    thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream),
                                  parent_col_ids.begin() + 1,
                                  parent_col_ids.end(),
                                  thrust::make_constant_iterator(1),
                                  map.begin());
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(1),
                       num_columns - 1,
                       [rowidx         = rowidx.begin(),
                        map            = map.begin(),
                        parent_col_ids = parent_col_ids.begin()] __device__(auto i) {
                         auto parent_col_id = parent_col_ids[i];
                         if (parent_col_id == 0)
                           --map[i - 1];
                         else
                           map[i - 1] += rowidx[parent_col_id];
                       });
    thrust::scatter(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator(1),
                    thrust::make_counting_iterator(1) + num_columns - 1,
                    map.begin(),
                    colidx.begin());

#ifdef CSR_DEBUG_PRINT
    print<NodeIndexT>(colidx, "h_pre_colidx", stream);
    print<size_type>(max_row_offsets, "h_max_row_offsets", stream);
#endif

    return colidx;
  };

  auto [num_columns, level_ordered_node_ids, level_ordered_col_ids] = level_orderings();
  auto [mapped_col_ids, parent_col_ids, rev_mapped_col_ids] =
    col_tree_adjs(num_columns, level_ordered_col_ids, level_ordered_node_ids);
  auto [max_row_offsets, column_categories] =
    col_tree_props(num_columns, level_ordered_col_ids, level_ordered_node_ids);

  /*
    5. CSR construction:
      a. Sort column levels and get their ordering
      b. For each column node coln iterated according to sorted_column_levels; do
          i. Find nodes that have coln as the parent node -> set adj_coln
          ii. row idx[coln] = size of adj_coln + 1
          iii. col idx[coln] = adj_coln U {parent_col_id[coln]}
  */
  auto rowidx = construct_rowidx(num_columns, parent_col_ids);
  auto colidx = partially_construct_colidx(num_columns, parent_col_ids, rowidx);

  auto max_children_max_row_offsets_colidx_update =
    [&colidx, &stream](NodeIndexT num_columns,
                       device_span<NodeIndexT const> rowidx,
                       device_span<NodeIndexT const> parent_col_ids,
                       device_span<row_offset_t const> max_row_offsets) {
      auto max_row_offsets_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        cuda::proclaim_return_type<row_offset_t>(
          [colidx          = colidx.begin(),
           max_row_offsets = max_row_offsets.begin()] __device__(size_t i) {
            if (colidx[i] == -1)
              return -1;
            else
              return max_row_offsets[colidx[i]];
          }));
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
      cudf::detail::cuda_memcpy_async(max_children_max_row_offsets.data(),
                                      max_row_offsets.data(),
                                      sizeof(row_offset_t),
                                      cudf::detail::host_memory_kind::PAGEABLE,
                                      stream);

#ifdef CSR_DEBUG_PRINT
      print<row_offset_t>(max_children_max_row_offsets, "h_max_children_max_row_offsets", stream);
#endif

      thrust::transform_if(
        rmm::exec_policy_nosync(stream),
        thrust::make_zip_iterator(thrust::make_counting_iterator(0),
                                  max_children_max_row_offsets.begin()),
        thrust::make_zip_iterator(thrust::make_counting_iterator(0) + num_columns,
                                  max_children_max_row_offsets.end()),
        max_children_max_row_offsets.begin(),
        [max_row_offsets = max_row_offsets.begin()] __device__(auto tup) {
          auto n = thrust::get<0>(tup);
          return max_row_offsets[n];
        },
        [] __device__(auto tup) {
          auto e = thrust::get<1>(tup);
          return e == -1;
        });

#ifdef CSR_DEBUG_PRINT
      print<row_offset_t>(max_children_max_row_offsets, "h_max_children_max_row_offsets", stream);
#endif

      // Skip the parent of root node
      thrust::scatter(rmm::exec_policy_nosync(stream),
                      parent_col_ids.begin() + 1,
                      parent_col_ids.end(),
                      rowidx.begin() + 1,
                      colidx.begin());

#ifdef CSR_DEBUG_PRINT
      print<NodeIndexT>(colidx, "h_colidx", stream);
#endif

      return max_children_max_row_offsets;
    };

  row_array_parent_col_id = rev_mapped_col_ids.element(row_array_parent_col_id, stream);
  auto root_node = (column_categories.element(0, stream) == NC_LIST && !is_array_of_arrays) ||
                       (is_array_of_arrays && row_array_parent_col_id)
                     ? 1
                     : 0;

  auto update_max_row_offsets =
    [&tree, row_array_parent_col_id, is_array_of_arrays, root_node, &stream](
      NodeIndexT num_columns,
      device_span<NodeIndexT const> rowidx,
      device_span<NodeIndexT const> colidx,
      device_span<NodeT const> column_categories,
      device_span<row_offset_t> max_row_offsets,
      device_span<row_offset_t const> max_children_max_row_offsets) {
      // Vector to store the latest ancestor of LIST type. If no such ancestor is found,
      // store the root node of tree. Note that a node cannot be an ancestor of itself
      auto list_ancestors = cudf::detail::make_zeroed_device_uvector_async<NodeIndexT>(
        static_cast<std::size_t>(num_columns), stream, cudf::get_current_device_resource_ref());
      auto* dev_num_levels_ptr = thrust::max_element(
        rmm::exec_policy_nosync(stream), tree.node_levels.begin(), tree.node_levels.end());
      if (root_node) list_ancestors.set_element_async(root_node, root_node, stream);
      thrust::for_each_n(rmm::exec_policy_nosync(stream),
                         thrust::make_counting_iterator(root_node + 1),
                         num_columns - root_node - 1,
                         [rowidx            = rowidx.begin(),
                          colidx            = colidx.begin(),
                          column_categories = column_categories.begin(),
                          dev_num_levels_ptr,
                          is_array_of_arrays,
                          row_array_parent_col_id,
                          root_node,
                          list_ancestors = list_ancestors.begin()] __device__(NodeIndexT node) {
                           auto num_levels      = *dev_num_levels_ptr;
                           list_ancestors[node] = colidx[rowidx[node]];
                           for (int level = 0;
                                level <= num_levels && list_ancestors[node] != root_node &&
                                column_categories[list_ancestors[node]] != NC_LIST;
                                level++) {
                             list_ancestors[node] = colidx[rowidx[list_ancestors[node]]];
                           }
                         });

#ifdef CSR_DEBUG_PRINT
      print<NodeIndexT>(list_ancestors, "h_list_ancestors", stream);
#endif

      // exclude root node
      thrust::gather(rmm::exec_policy_nosync(stream),
                     list_ancestors.begin(),
                     list_ancestors.end(),
                     max_children_max_row_offsets.begin(),
                     max_row_offsets.begin());

#ifdef CSR_DEBUG_PRINT
      print<size_type>(max_row_offsets, "h_max_row_offsets", stream);
#endif
    };

  // Mixed types in List children go to different columns,
  // so all immediate children of list column should have same max_row_offsets.
  //   create list's children max_row_offsets array
  //   gather the max_row_offsets from children row offset array.
  if (num_columns > 1) {
    auto max_children_max_row_offsets = max_children_max_row_offsets_colidx_update(
      num_columns, rowidx, parent_col_ids, max_row_offsets);
    update_max_row_offsets(num_columns,
                           rowidx,
                           colidx,
                           column_categories,
                           max_row_offsets,
                           max_children_max_row_offsets);
  }

  return std::tuple{
    compressed_sparse_row{std::move(rowidx), std::move(colidx)},
    column_tree_properties{
      std::move(column_categories), std::move(max_row_offsets), std::move(mapped_col_ids)}};
}

}  // namespace experimental::detail
}  // namespace cudf::io::json
