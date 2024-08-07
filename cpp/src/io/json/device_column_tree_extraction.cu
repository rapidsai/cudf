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

#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/string_parsing.hpp"
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
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

namespace cudf::io::json {

namespace experimental::detail {

using row_offset_t = size_type;

rmm::device_uvector<row_offset_t> extract_device_column_subtree(
  const csr& adjacency,
  const column_tree_properties& props,
  cudf::io::json_reader_options reader_options,
  rmm::cuda_stream_view stream)
{
  // What are the cases in which estimation works?
  CUDF_EXPECTS(reader_options.is_enabled_mixed_types_as_string() == false,
               "mixed type as string has not yet been implemented");
  CUDF_EXPECTS(reader_options.is_enabled_prune_columns() == false,
               "column pruning has not yet been implemented");

  auto& rowidx          = adjacency.rowidx;
  auto& colidx          = adjacency.colidx;
  auto& categories      = props.categories;
  auto& max_row_offsets = props.max_row_offsets;
  auto& num_levels      = props.num_levels;

  // Traversing the column tree and annotating the device column subtree
  auto num_columns = rowidx.size() - 1;
  rmm::device_uvector<row_offset_t> subtree_nrows(max_row_offsets, stream);

  // 1. removing NC_ERR nodes and their descendants i.e.
  // removing the entire subtree rooted at the nodes with category NC_ERR
  {
    rmm::device_uvector<NodeIndexT> err_ancestors(num_columns, stream);
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(0),
      num_columns,
      [rowidx         = rowidx.begin(),
       colidx         = colidx.begin(),
       num_levels_ptr = num_levels.data(),
       categories     = categories.begin(),
       err_ancestors  = err_ancestors.begin()] __device__(NodeIndexT node) {
        auto num_levels     = *num_levels_ptr;
        err_ancestors[node] = node;
        for (int level = 0; level <= num_levels; level++) {
          if (err_ancestors[node] == -1 || categories[err_ancestors[node]] == NC_ERR) break;
          if (err_ancestors[node] > 0)
            err_ancestors[node] = colidx[rowidx[err_ancestors[node]]];
          else
            err_ancestors[node] = -1;
        }
      });
    thrust::gather_if(rmm::exec_policy_nosync(stream),
                      err_ancestors.begin(),
                      err_ancestors.end(),
                      err_ancestors.begin(),
                      thrust::make_constant_iterator(0),
                      subtree_nrows.begin(),
                      [] __device__(auto ancestor) { return ancestor != -1; });
  }

  // 2. Let's do some validation of the column tree based on its properties.
  // We will be using these properties to filter nodes later on.
  // ===========================================================================
  // (i) Every node v is of type string, val, field name, list or struct.
  // (ii) String and val cannot have any children i.e. they can only be leaf nodes
  // (iii) If v is a field name, it can have struct, list, string and val as children.
  // (iv) If v is a struct, it can have a field name as child
  // (v) If v is a list, it can have string, val, list or struct as child
  // (vi) There can only be at most one string and one val child for a given node, but many struct,
  //      list and field name children.
  // (vii) When mixed type support is disabled -
  //       (a) A mix of lists and structs in the same column is not supported i.e a field name and
  //       list node cannot have both list and struct as children
  //       (b) If there is a mix of str/val
  //       and list/struct in the same column, then str/val is discarded

  // Validation of (vii)(a)
  {
    if (!reader_options.is_enabled_mixed_types_as_string()) {
      auto num_field_and_list_nodes =
        thrust::count_if(rmm::exec_policy_nosync(stream),
                         categories.begin(),
                         categories.end(),
                         [] __device__(auto const ctg) { return ctg == NC_FN || ctg == NC_LIST; });
      rmm::device_uvector<NodeIndexT> field_and_list_nodes(num_field_and_list_nodes, stream);
      thrust::partition_copy(rmm::exec_policy_nosync(stream),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(0) + num_columns,
                             field_and_list_nodes.begin(),
                             thrust::make_discard_iterator(),
                             [categories = categories.begin()] __device__(NodeIndexT node) {
                               return categories[node] == NC_LIST || categories[node] == NC_FN;
                             });
      bool is_valid_tree = thrust::all_of(
        rmm::exec_policy_nosync(stream),
        field_and_list_nodes.begin(),
        field_and_list_nodes.end(),
        [rowidx     = rowidx.begin(),
         colidx     = colidx.begin(),
         categories = categories.begin()] __device__(NodeIndexT node) {
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
                   "Property 7a is not satisfied i.e. mix of LIST and STRUCT in same column is not "
                   "supported when mixed type support is disabled");
    }
  }

  // Validation of (vii)(b) i.e. ignore_vals in previous implementation
  // We need to identify leaf nodes that have non-leaf sibling nodes
  // i.e. we need to ignore leaf nodes at level above the last level
  // idea: leaf nodes have adjacency 1. So if there is an adjacency 1 inbetween non-one
  // adjacencies, then found the leaf node. Corner case: consider the last set of consecutive
  // ones. If the leftmost of those ones (say node u) has a non-leaf sibling
  // (can be found by looking at the adjacencies of the siblings
  // (which are in turn found from the colidx of the parent u), then this leaf node should be
  // ignored, otherwise all good.
  {
    if (!reader_options.is_enabled_mixed_types_as_string()) {
      // TODO: use cub segmented reduce here!
      rmm::device_uvector<NodeIndexT> num_adjacent_nodes(
        num_columns + 1,
        stream);  // since adjacent_difference requires that the output have the same length as
                  // input
      thrust::adjacent_difference(
        rmm::exec_policy_nosync(stream), rowidx.begin(), rowidx.end(), num_adjacent_nodes.begin());
      auto num_leaf_nodes = thrust::count_if(rmm::exec_policy_nosync(stream),
                                             num_adjacent_nodes.begin() + 1,
                                             num_adjacent_nodes.end(),
                                             [] __device__(auto const adj) { return adj == 1; });
      rmm::device_uvector<NodeIndexT> leaf_nodes(num_leaf_nodes, stream);
      thrust::copy_if(rmm::exec_policy_nosync(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + num_columns,
                      leaf_nodes.begin(),
                      [num_adjacent_nodes = num_adjacent_nodes.begin()] __device__(size_t node) {
                        return num_adjacent_nodes[node] == 1;
                      });

      auto rev_node_it =
        thrust::make_reverse_iterator(thrust::make_counting_iterator(0) + num_columns);
      auto rev_leaf_nodes_it = thrust::make_reverse_iterator(leaf_nodes.begin());
      // the node number that could be the leftmost leaf node is given by u =
      // *(is_leftmost_leaf.second
      // - 1)
      auto is_leftmost_leaf = thrust::mismatch(
        rmm::exec_policy_nosync(stream), rev_node_it, rev_node_it + num_columns, rev_leaf_nodes_it);
      NodeIndexT leftmost_leaf_node = leaf_nodes.element(
        num_leaf_nodes - thrust::distance(rev_leaf_nodes_it, is_leftmost_leaf.second - 1) - 1,
        stream);

      // upper_bound search for u in rowidx for parent node v. Now check if any of the other child
      // nodes of v is non-leaf i.e check if u is the first child of v. If yes, then
      // leafmost_leaf_node is the leftmost leaf node. Otherwise, discard all children of v after
      // and including u
      auto parent_it = thrust::upper_bound(
        rmm::exec_policy_nosync(stream), rowidx.begin(), rowidx.end(), leftmost_leaf_node);
      NodeIndexT parent           = thrust::distance(rowidx.begin(), parent_it - 1);
      NodeIndexT parent_adj_start = rowidx.element(parent, stream);
      NodeIndexT parent_adj_end   = rowidx.element(parent + 1, stream);
      auto childnum_it            = thrust::lower_bound(rmm::exec_policy_nosync(stream),
                                             colidx.begin() + parent_adj_start,
                                             colidx.begin() + parent_adj_end,
                                             leftmost_leaf_node);

      auto retained_leaf_nodes_it =
        leaf_nodes.begin() + num_leaf_nodes -
        thrust::distance(rev_leaf_nodes_it, is_leftmost_leaf.second - 1) - 1;
      if (childnum_it != colidx.begin() + parent_adj_start + 1) {
        // discarding from u to last child of parent
        retained_leaf_nodes_it += thrust::distance(childnum_it, colidx.begin() + parent_adj_end);
      }
      // now, all nodes from leaf_nodes.begin() to retained_leaf_nodes_it need to be discarded i.e.
      // they are part of ignore_vals
      thrust::scatter(rmm::exec_policy_nosync(stream),
                      thrust::make_constant_iterator(0),
                      thrust::make_constant_iterator(0) +
                        thrust::distance(leaf_nodes.begin(), retained_leaf_nodes_it),
                      leaf_nodes.begin(),
                      subtree_nrows.begin());
    }
  }

  // (Optional?) TODO: Validation of the remaining column tree properties

  return subtree_nrows;
}

device_column_subtree_properties allocate_device_column_subtree_properties(
  device_span<row_offset_t> subtree_nrows,
  const column_tree_properties& props,
  rmm::cuda_stream_view stream)
{
  auto num_columns      = subtree_nrows.size();
  auto& categories      = props.categories;
  auto& max_row_offsets = props.max_row_offsets;

  auto num_subtree_nodes = thrust::count_if(rmm::exec_policy_nosync(stream),
                                            subtree_nrows.begin(),
                                            subtree_nrows.end(),
                                            [] __device__(auto mro) { return mro != 0; });
  // For the subtree, we allocate memory for device column subtree properties
  rmm::device_uvector<NodeIndexT> subtree_properties_map(num_subtree_nodes, stream);
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + num_columns,
                  subtree_nrows.begin(),
                  subtree_properties_map.begin(),
                  [] __device__(auto mro) { return mro != 0; });
  // TODO: three way partitioning in cub::If
  auto str_partitioning_idx_it =
    thrust::partition(rmm::exec_policy(stream),
                      subtree_properties_map.begin(),
                      subtree_properties_map.end(),
                      [categories = categories.begin()] __device__(NodeIndexT node) {
                        return categories[node] == NC_STR || categories[node] == NC_VAL;
                      });
  auto str_val_end = thrust::distance(subtree_properties_map.begin(), str_partitioning_idx_it);
  auto max_row_offsets_it =
    thrust::make_permutation_iterator(max_row_offsets.begin(), subtree_properties_map.begin());
  size_type string_offsets_size =
    thrust::reduce(rmm::exec_policy(stream), max_row_offsets_it, max_row_offsets_it + str_val_end) +
    str_val_end;
  rmm::device_uvector<SymbolOffsetT> string_offsets(string_offsets_size, stream);
  rmm::device_uvector<SymbolOffsetT> string_lengths(string_offsets_size, stream);

  auto list_partitioning_idx_it =
    thrust::partition(rmm::exec_policy(stream),
                      str_partitioning_idx_it,
                      subtree_properties_map.end(),
                      [categories = categories.begin()] __device__(NodeIndexT node) {
                        return categories[node] == NC_LIST;
                      });
  auto list_end = thrust::distance(subtree_properties_map.begin(), list_partitioning_idx_it);
  max_row_offsets_it =
    thrust::make_permutation_iterator(max_row_offsets.begin(), subtree_properties_map.begin()) +
    str_val_end;
  size_type child_offsets_size =
    thrust::reduce(
      rmm::exec_policy(stream), max_row_offsets_it, max_row_offsets_it + (list_end - str_val_end)) +
    2 * (list_end - str_val_end);
  rmm::device_uvector<SymbolOffsetT> child_offsets(child_offsets_size, stream);

  auto validity_buffer_size =
    thrust::reduce(rmm::exec_policy(stream), subtree_nrows.begin(), subtree_nrows.end());
  auto validity = cudf::detail::create_null_mask(validity_buffer_size,
                                                 cudf::mask_state::ALL_NULL,
                                                 stream,
                                                 rmm::mr::get_current_device_resource());

  return device_column_subtree_properties{std::move(string_offsets),
                                          std::move(string_lengths),
                                          std::move(child_offsets),
                                          std::move(validity)};
}

void initialize_device_column_subtree_properties(device_column_subtree_properties& d_props,
                                                 device_span<row_offset_t> subtree_nrows,
                                                 tree_meta_t& tree,
                                                 device_span<NodeIndexT> original_col_ids,
                                                 device_span<row_offset_t> row_offsets,
                                                 column_tree_properties& c_props,
                                                 rmm::cuda_stream_view stream)
{
  auto num_nodes   = tree.node_levels.size();
  auto num_columns = c_props.categories.size();
  // now we actually do the annotation
  // relabel original_col_ids with the positions of the csr_unique_col_ids with same element. How do
  // we accomplish this? one idea is to sort the row offsets by node level. Just the way we did this
  // for the csr_column_ids sort original_col_ids, extract subtree based on the annotation above,
  // and then initialize.
  auto [sorted_node_levels, sorted_node_levels_order] =
    cudf::io::json::detail::stable_sorted_key_order<size_t, TreeDepthT>(tree.node_levels, stream);
  auto row_offsets_it =
    thrust::make_permutation_iterator(row_offsets.begin(), sorted_node_levels_order.begin());
  auto node_range_begin_it   = thrust::make_permutation_iterator(tree.node_range_begin.begin(),
                                                               sorted_node_levels_order.begin());
  auto node_range_end_it     = thrust::make_permutation_iterator(tree.node_range_end.begin(),
                                                             sorted_node_levels_order.begin());
  auto node_range_lengths_it = thrust::make_transform_iterator(
    thrust::make_zip_iterator(node_range_begin_it, node_range_end_it),
    cuda::proclaim_return_type<SymbolOffsetT>([] __device__(auto range_it) {
      return thrust::get<1>(range_it) - thrust::get<0>(range_it);
    }));

  auto node_col_ids_it =
    thrust::make_permutation_iterator(original_col_ids.begin(), sorted_node_levels_order.begin());
  auto node_categories_it = thrust::make_permutation_iterator(tree.node_categories.begin(),
                                                              sorted_node_levels_order.begin());

  rmm::device_uvector<row_offset_t> sorted_subtree_nrows(num_columns, stream);
  thrust::copy(rmm::exec_policy_nosync(stream),
               subtree_nrows.begin(),
               subtree_nrows.end(),
               sorted_subtree_nrows.begin());
  thrust::sort_by_key(rmm::exec_policy_nosync(stream),
                      c_props.mapped_ids.begin(),
                      c_props.mapped_ids.end(),
                      sorted_subtree_nrows.begin());

  thrust::copy_if(
    rmm::exec_policy_nosync(stream),
    thrust::make_zip_iterator(node_range_begin_it, node_range_lengths_it),
    thrust::make_zip_iterator(node_range_begin_it + num_nodes, node_range_lengths_it + num_nodes),
    thrust::make_counting_iterator(0),
    thrust::make_zip_iterator(d_props.string_offsets.begin(), d_props.string_lengths.begin()),
    [sorted_subtree_nrows = sorted_subtree_nrows.begin(),
     node_col_ids_it,
     node_categories_it] __device__(NodeIndexT node) {
      return sorted_subtree_nrows[node_col_ids_it[node]] &&
             (node_categories_it[node] == NC_STR || node_categories_it[node] == NC_VAL);
    });

  // row_offsets need to be prefix summed across columns for validity initialization
  // TODO: replace replace_if with a transform input iterator and pass that to inclusive scan
  thrust::replace_if(
    rmm::exec_policy_nosync(stream),
    row_offsets_it,
    row_offsets_it + num_nodes,
    thrust::make_counting_iterator(0),
    [sorted_subtree_nrows = sorted_subtree_nrows.begin(), node_col_ids_it] __device__(
      NodeIndexT node) { return sorted_subtree_nrows[node_col_ids_it[node]] == 0; },
    0);
  thrust::inclusive_scan(
    rmm::exec_policy_nosync(stream), row_offsets_it, row_offsets_it + num_nodes, row_offsets_it);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    num_nodes,
    [sorted_subtree_nrows = sorted_subtree_nrows.begin(),
     node_col_ids_it,
     node_categories_it,
     row_offsets_it,
     validity = static_cast<bitmask_type*>(d_props.validity.data())] __device__(NodeIndexT node) {
      if (sorted_subtree_nrows[node_col_ids_it[node]] && node_categories_it[node] != NC_LIST)
        cudf::set_bit(validity, row_offsets_it[node]);
    });

  // scatter list offsets
}

}  // namespace experimental::detail

namespace detail {
/**
 * @brief Checks if all strings in each string column in the tree are nulls.
 * For non-string columns, it's set as true. If any of rows in a string column is false, it's set as
 * false.
 *
 * @param input Input JSON string device data
 * @param d_column_tree column tree representation of JSON string
 * @param tree Node tree representation of the JSON string
 * @param col_ids Column ids of the nodes in the tree
 * @param options Parsing options specifying the parsing behaviour
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Array of bytes where each byte indicate if it is all nulls string column.
 */
rmm::device_uvector<uint8_t> is_all_nulls_each_column(device_span<SymbolT const> input,
                                                      tree_meta_t const& d_column_tree,
                                                      tree_meta_t const& tree,
                                                      device_span<NodeIndexT> col_ids,
                                                      cudf::io::json_reader_options const& options,
                                                      rmm::cuda_stream_view stream)
{
  auto const num_nodes = col_ids.size();
  auto const num_cols  = d_column_tree.node_categories.size();
  rmm::device_uvector<uint8_t> is_all_nulls(num_cols, stream);
  thrust::fill(rmm::exec_policy(stream), is_all_nulls.begin(), is_all_nulls.end(), true);

  auto parse_opt = parsing_options(options, stream);
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::counting_iterator<size_type>(0),
    num_nodes,
    [options           = parse_opt.view(),
     data              = input.data(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin(),
     range_begin       = tree.node_range_begin.begin(),
     range_end         = tree.node_range_end.begin(),
     is_all_nulls      = is_all_nulls.begin()] __device__(size_type i) {
      auto const node_category = column_categories[col_ids[i]];
      if (node_category == NC_STR or node_category == NC_VAL) {
        auto const is_null_literal = serialized_trie_contains(
          options.trie_na,
          {data + range_begin[i], static_cast<size_t>(range_end[i] - range_begin[i])});
        if (!is_null_literal) is_all_nulls[col_ids[i]] = false;
      }
    });
  return is_all_nulls;
}

/**
 * @brief Get the column indices for the values column for array of arrays rows
 *
 * @param row_array_children_level The level of the row array's children
 * @param d_tree The tree metadata
 * @param col_ids The column ids
 * @param num_columns The number of columns
 * @param stream The stream to use
 * @return The value columns' indices
 */
rmm::device_uvector<NodeIndexT> get_values_column_indices(TreeDepthT const row_array_children_level,
                                                          tree_meta_t const& d_tree,
                                                          device_span<NodeIndexT> col_ids,
                                                          size_type const num_columns,
                                                          rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto [level2_nodes, level2_indices] = get_array_children_indices(
    row_array_children_level, d_tree.node_levels, d_tree.parent_node_ids, stream);
  auto col_id_location = thrust::make_permutation_iterator(col_ids.begin(), level2_nodes.begin());
  rmm::device_uvector<NodeIndexT> values_column_indices(num_columns, stream);
  thrust::scatter(rmm::exec_policy(stream),
                  level2_indices.begin(),
                  level2_indices.end(),
                  col_id_location,
                  values_column_indices.begin());
  return values_column_indices;
}

/**
 * @brief Copies strings specified by pair of begin, end offsets to host vector of strings.
 *
 * @param input String device buffer
 * @param node_range_begin Begin offset of the strings
 * @param node_range_end End offset of the strings
 * @param stream CUDA stream
 * @return Vector of strings
 */
std::vector<std::string> copy_strings_to_host_sync(
  device_span<SymbolT const> input,
  device_span<SymbolOffsetT const> node_range_begin,
  device_span<SymbolOffsetT const> node_range_end,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto const num_strings = node_range_begin.size();
  rmm::device_uvector<size_type> string_offsets(num_strings, stream);
  rmm::device_uvector<size_type> string_lengths(num_strings, stream);
  auto d_offset_pairs = thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  thrust::transform(rmm::exec_policy(stream),
                    d_offset_pairs,
                    d_offset_pairs + num_strings,
                    thrust::make_zip_iterator(string_offsets.begin(), string_lengths.begin()),
                    [] __device__(auto const& offsets) {
                      // Note: first character for non-field columns
                      return thrust::make_tuple(
                        static_cast<size_type>(thrust::get<0>(offsets)),
                        static_cast<size_type>(thrust::get<1>(offsets) - thrust::get<0>(offsets)));
                    });

  cudf::io::parse_options_view options_view{};
  options_view.quotechar  = '\0';  // no quotes
  options_view.keepquotes = true;
  auto d_offset_length_it =
    thrust::make_zip_iterator(string_offsets.begin(), string_lengths.begin());
  auto d_column_names = parse_data(input.data(),
                                   d_offset_length_it,
                                   num_strings,
                                   data_type{type_id::STRING},
                                   rmm::device_buffer{},
                                   0,
                                   options_view,
                                   stream,
                                   rmm::mr::get_current_device_resource());
  auto to_host        = [stream](auto const& col) {
    if (col.is_empty()) return std::vector<std::string>{};
    auto const scv     = cudf::strings_column_view(col);
    auto const h_chars = cudf::detail::make_std_vector_async<char>(
      cudf::device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)), stream);
    auto const h_offsets = cudf::detail::make_std_vector_async(
      cudf::device_span<cudf::size_type const>(scv.offsets().data<cudf::size_type>() + scv.offset(),
                                               scv.size() + 1),
      stream);
    stream.synchronize();

    // build std::string vector from chars and offsets
    std::vector<std::string> host_data;
    host_data.reserve(col.size());
    std::transform(
      std::begin(h_offsets),
      std::end(h_offsets) - 1,
      std::begin(h_offsets) + 1,
      std::back_inserter(host_data),
      [&](auto start, auto end) { return std::string(h_chars.data() + start, end - start); });
    return host_data;
  };
  return to_host(d_column_names->view());
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
void make_device_json_column(device_span<SymbolT const> input,
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

  bool const is_enabled_lines                 = options.is_enabled_lines();
  bool const is_enabled_mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  auto const num_nodes                        = col_ids.size();
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
  auto [d_column_tree, d_unique_col_ids, d_max_row_offsets] =
    reduce_to_column_tree(tree,
                          col_ids,
                          sorted_col_ids,
                          node_ids,
                          row_offsets,
                          is_array_of_arrays,
                          row_array_parent_col_id,
                          stream);
  auto num_columns    = d_unique_col_ids.size();
  auto unique_col_ids = cudf::detail::make_std_vector_async(d_unique_col_ids, stream);
  auto column_categories =
    cudf::detail::make_std_vector_async(d_column_tree.node_categories, stream);
  auto column_parent_ids =
    cudf::detail::make_std_vector_async(d_column_tree.parent_node_ids, stream);
  auto column_range_beg =
    cudf::detail::make_std_vector_async(d_column_tree.node_range_begin, stream);
  auto max_row_offsets = cudf::detail::make_std_vector_async(d_max_row_offsets, stream);
  std::vector<std::string> column_names = copy_strings_to_host_sync(
    input, d_column_tree.node_range_begin, d_column_tree.node_range_end, stream);
  stream.synchronize();
  // array of arrays column names
  if (is_array_of_arrays) {
    TreeDepthT const row_array_children_level = is_enabled_lines ? 1 : 2;
    auto values_column_indices =
      get_values_column_indices(row_array_children_level, tree, col_ids, num_columns, stream);
    auto h_values_column_indices =
      cudf::detail::make_std_vector_async(values_column_indices, stream);
    stream.synchronize();
    std::transform(unique_col_ids.begin(),
                   unique_col_ids.end(),
                   column_names.begin(),
                   column_names.begin(),
                   [&h_values_column_indices, &column_parent_ids, row_array_parent_col_id](
                     auto col_id, auto name) mutable {
                     return column_parent_ids[col_id] == row_array_parent_col_id
                              ? std::to_string(h_values_column_indices[col_id])
                              : name;
                   });
  }

  auto to_json_col_type = [](auto category) {
    switch (category) {
      case NC_STRUCT: return json_col_t::StructColumn;
      case NC_LIST: return json_col_t::ListColumn;
      case NC_STR: [[fallthrough]];
      case NC_VAL: return json_col_t::StringColumn;
      default: return json_col_t::Unknown;
    }
  };
  auto init_to_zero = [stream](auto& v) {
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), v.begin(), v.end(), 0);
  };

  auto initialize_json_columns = [&](auto i, auto& col) {
    if (column_categories[i] == NC_ERR || column_categories[i] == NC_FN) {
      return;
    } else if (column_categories[i] == NC_VAL || column_categories[i] == NC_STR) {
      col.string_offsets.resize(max_row_offsets[i] + 1, stream);
      col.string_lengths.resize(max_row_offsets[i] + 1, stream);
      init_to_zero(col.string_offsets);
      init_to_zero(col.string_lengths);
    } else if (column_categories[i] == NC_LIST) {
      col.child_offsets.resize(max_row_offsets[i] + 2, stream);
      init_to_zero(col.child_offsets);
    }
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = to_json_col_type(column_categories[i]);
  };

  auto reinitialize_as_string = [&](auto i, auto& col) {
    col.string_offsets.resize(max_row_offsets[i] + 1, stream);
    col.string_lengths.resize(max_row_offsets[i] + 1, stream);
    init_to_zero(col.string_offsets);
    init_to_zero(col.string_lengths);
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = json_col_t::StringColumn;
    // destroy references of all child columns after this step, by calling remove_child_columns
  };

  path_from_tree tree_path{column_categories,
                           column_parent_ids,
                           column_names,
                           is_array_of_arrays,
                           row_array_parent_col_id};

  // 2. generate nested columns tree and its device_memory
  // reorder unique_col_ids w.r.t. column_range_begin for order of column to be in field order.
  auto h_range_col_id_it =
    thrust::make_zip_iterator(column_range_beg.begin(), unique_col_ids.begin());
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return thrust::get<0>(a) < thrust::get<0>(b);
  });

  std::vector<uint8_t> is_str_column_all_nulls{};
  if (is_enabled_mixed_types_as_string) {
    is_str_column_all_nulls = cudf::detail::make_std_vector_sync(
      is_all_nulls_each_column(input, d_column_tree, tree, col_ids, options, stream), stream);
  }

  // use hash map because we may skip field name's col_ids
  std::unordered_map<NodeIndexT, std::reference_wrapper<device_json_column>> columns;
  // map{parent_col_id, child_col_name}> = child_col_id, used for null value column tracking
  std::map<std::pair<NodeIndexT, std::string>, NodeIndexT> mapped_columns;
  // find column_ids which are values, but should be ignored in validity
  auto ignore_vals = cudf::detail::make_host_vector<uint8_t>(num_columns, stream);
  std::vector<uint8_t> is_mixed_type_column(num_columns, 0);
  std::vector<uint8_t> is_pruned(num_columns, 0);
  columns.try_emplace(parent_node_sentinel, std::ref(root));

  std::function<void(NodeIndexT, device_json_column&)> remove_child_columns =
    [&](NodeIndexT this_col_id, device_json_column& col) {
      for (auto col_name : col.column_order) {
        auto child_id                  = mapped_columns[{this_col_id, col_name}];
        is_mixed_type_column[child_id] = 1;
        remove_child_columns(child_id, col.child_columns.at(col_name));
        mapped_columns.erase({this_col_id, col_name});
        columns.erase(child_id);
      }
      col.child_columns.clear();  // their references are deleted above.
      col.column_order.clear();
    };

  auto name_and_parent_index = [&is_array_of_arrays,
                                &row_array_parent_col_id,
                                &column_parent_ids,
                                &column_categories,
                                &column_names](auto this_col_id) {
    std::string name   = "";
    auto parent_col_id = column_parent_ids[this_col_id];
    if (parent_col_id == parent_node_sentinel || column_categories[parent_col_id] == NC_LIST) {
      if (is_array_of_arrays && parent_col_id == row_array_parent_col_id) {
        name = column_names[this_col_id];
      } else {
        name = list_child_name;
      }
    } else if (column_categories[parent_col_id] == NC_FN) {
      auto field_name_col_id = parent_col_id;
      parent_col_id          = column_parent_ids[parent_col_id];
      name                   = column_names[field_name_col_id];
    } else {
      CUDF_FAIL("Unexpected parent column category");
    }
    return std::pair{name, parent_col_id};
  };

  // Prune columns that are not required to be parsed.
  if (options.is_enabled_prune_columns()) {
    for (auto const this_col_id : unique_col_ids) {
      if (column_categories[this_col_id] == NC_ERR || column_categories[this_col_id] == NC_FN) {
        continue;
      }
      // Struct, List, String, Value
      auto [name, parent_col_id] = name_and_parent_index(this_col_id);
      // get path of this column, and get its dtype if present in options
      auto const nt                             = tree_path.get_path(this_col_id);
      std::optional<data_type> const user_dtype = get_path_data_type(nt, options);
      if (!user_dtype.has_value() and parent_col_id != parent_node_sentinel) {
        is_pruned[this_col_id] = 1;
        continue;
      } else {
        // make sure all its parents are not pruned.
        while (parent_col_id != parent_node_sentinel and is_pruned[parent_col_id] == 1) {
          is_pruned[parent_col_id] = 0;
          parent_col_id            = column_parent_ids[parent_col_id];
        }
      }
    }
  }

  // Build the column tree, also, handles mixed types.
  for (auto const this_col_id : unique_col_ids) {
    if (column_categories[this_col_id] == NC_ERR || column_categories[this_col_id] == NC_FN) {
      continue;
    }
    // Struct, List, String, Value
    auto [name, parent_col_id] = name_and_parent_index(this_col_id);

    // if parent is mixed type column or this column is pruned, ignore this column.
    if (parent_col_id != parent_node_sentinel &&
        (is_mixed_type_column[parent_col_id] || is_pruned[this_col_id])) {
      ignore_vals[this_col_id] = 1;
      if (is_mixed_type_column[parent_col_id]) { is_mixed_type_column[this_col_id] = 1; }
      continue;
    }

    // If the child is already found,
    // replace if this column is a nested column and the existing was a value column
    // ignore this column if this column is a value column and the existing was a nested column
    auto it = columns.find(parent_col_id);
    CUDF_EXPECTS(it != columns.end(), "Parent column not found");
    auto& parent_col = it->second.get();
    bool replaced    = false;
    if (mapped_columns.count({parent_col_id, name}) > 0) {
      auto const old_col_id = mapped_columns[{parent_col_id, name}];
      // If mixed type as string is enabled, make both of them strings and merge them.
      // All child columns will be ignored when parsing.
      if (is_enabled_mixed_types_as_string) {
        bool const is_mixed_type = [&]() {
          // If new or old is STR and they are all not null, make it mixed type, else ignore.
          if (column_categories[this_col_id] == NC_VAL ||
              column_categories[this_col_id] == NC_STR) {
            if (is_str_column_all_nulls[this_col_id]) return false;
          }
          if (column_categories[old_col_id] == NC_VAL || column_categories[old_col_id] == NC_STR) {
            if (is_str_column_all_nulls[old_col_id]) return false;
          }
          return true;
        }();
        if (is_mixed_type) {
          is_mixed_type_column[this_col_id] = 1;
          is_mixed_type_column[old_col_id]  = 1;
          // if old col type (not cat) is list or struct, replace with string.
          auto& col = columns.at(old_col_id).get();
          if (col.type == json_col_t::ListColumn or col.type == json_col_t::StructColumn) {
            reinitialize_as_string(old_col_id, col);
            remove_child_columns(old_col_id, col);
            // all its children (which are already inserted) are ignored later.
          }
          col.forced_as_string_column = true;
          columns.try_emplace(this_col_id, columns.at(old_col_id));
          continue;
        }
      }

      if (column_categories[this_col_id] == NC_VAL || column_categories[this_col_id] == NC_STR) {
        ignore_vals[this_col_id] = 1;
        continue;
      }
      if (column_categories[old_col_id] == NC_VAL || column_categories[old_col_id] == NC_STR) {
        // remap
        ignore_vals[old_col_id] = 1;
        mapped_columns.erase({parent_col_id, name});
        columns.erase(old_col_id);
        parent_col.child_columns.erase(name);
        replaced = true;  // to skip duplicate name in column_order
      } else {
        // If this is a nested column but we're trying to insert either (a) a list node into a
        // struct column or (b) a struct node into a list column, we fail
        CUDF_EXPECTS(not((column_categories[old_col_id] == NC_LIST and
                          column_categories[this_col_id] == NC_STRUCT) or
                         (column_categories[old_col_id] == NC_STRUCT and
                          column_categories[this_col_id] == NC_LIST)),
                     "A mix of lists and structs within the same column is not supported");
      }
    }

    if (is_enabled_mixed_types_as_string) {
      // get path of this column, check if it is a struct forced as string, and enforce it
      auto const nt                             = tree_path.get_path(this_col_id);
      std::optional<data_type> const user_dtype = get_path_data_type(nt, options);
      if (column_categories[this_col_id] == NC_STRUCT and user_dtype.has_value() and
          user_dtype.value().id() == type_id::STRING) {
        is_mixed_type_column[this_col_id] = 1;
        column_categories[this_col_id]    = NC_STR;
      }
    }

    CUDF_EXPECTS(parent_col.child_columns.count(name) == 0, "duplicate column name: " + name);
    // move into parent
    device_json_column col(stream, mr);
    initialize_json_columns(this_col_id, col);
    auto inserted = parent_col.child_columns.try_emplace(name, std::move(col)).second;
    CUDF_EXPECTS(inserted, "child column insertion failed, duplicate column name in the parent");
    if (not replaced) parent_col.column_order.push_back(name);
    columns.try_emplace(this_col_id, std::ref(parent_col.child_columns.at(name)));
    mapped_columns.try_emplace(std::make_pair(parent_col_id, name), this_col_id);
  }

  if (is_enabled_mixed_types_as_string) {
    // ignore all children of mixed type columns
    for (auto const this_col_id : unique_col_ids) {
      auto parent_col_id = column_parent_ids[this_col_id];
      if (parent_col_id != parent_node_sentinel and is_mixed_type_column[parent_col_id] == 1) {
        is_mixed_type_column[this_col_id] = 1;
        ignore_vals[this_col_id]          = 1;
        columns.erase(this_col_id);
      }
      // Convert only mixed type columns as string (so to copy), but not its children
      if (parent_col_id != parent_node_sentinel and is_mixed_type_column[parent_col_id] == 0 and
          is_mixed_type_column[this_col_id] == 1)
        column_categories[this_col_id] = NC_STR;
    }
    cudaMemcpyAsync(d_column_tree.node_categories.begin(),
                    column_categories.data(),
                    column_categories.size() * sizeof(column_categories[0]),
                    cudaMemcpyDefault,
                    stream.value());
  }

  // restore unique_col_ids order
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return thrust::get<1>(a) < thrust::get<1>(b);
  });
  // move columns data to device.
  auto columns_data = cudf::detail::make_host_vector<json_column_data>(num_columns, stream);
  for (auto& [col_id, col_ref] : columns) {
    if (col_id == parent_node_sentinel) continue;
    auto& col            = col_ref.get();
    columns_data[col_id] = json_column_data{col.string_offsets.data(),
                                            col.string_lengths.data(),
                                            col.child_offsets.data(),
                                            static_cast<bitmask_type*>(col.validity.data())};
  }

  auto d_ignore_vals = cudf::detail::make_device_uvector_async(
    ignore_vals, stream, rmm::mr::get_current_device_resource());
  auto d_columns_data = cudf::detail::make_device_uvector_async(
    columns_data, stream, rmm::mr::get_current_device_resource());

  // 3. scatter string offsets to respective columns, set validity bits
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::counting_iterator<size_type>(0),
    num_nodes,
    [column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin(),
     row_offsets       = row_offsets.begin(),
     range_begin       = tree.node_range_begin.begin(),
     range_end         = tree.node_range_end.begin(),
     d_ignore_vals     = d_ignore_vals.begin(),
     d_columns_data    = d_columns_data.begin()] __device__(size_type i) {
      if (d_ignore_vals[col_ids[i]]) return;
      auto const node_category = column_categories[col_ids[i]];
      switch (node_category) {
        case NC_STRUCT: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_LIST: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_STR: [[fallthrough]];
        case NC_VAL:
          if (d_ignore_vals[col_ids[i]]) break;
          set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]);
          d_columns_data[col_ids[i]].string_offsets[row_offsets[i]] = range_begin[i];
          d_columns_data[col_ids[i]].string_lengths[row_offsets[i]] = range_end[i] - range_begin[i];
          break;
        default: break;
      }
    });

  // 4. scatter List offset
  // copy_if only node's whose parent is list, (node_id, parent_col_id)
  // stable_sort by parent_col_id of {node_id}.
  // For all unique parent_node_id of (i==0, i-1!=i), write start offset.
  //                                  (i==last, i+1!=i), write end offset.
  //    unique_copy_by_key {parent_node_id} {row_offset} to
  //    col[parent_col_id].child_offsets[row_offset[parent_node_id]]

  auto& parent_col_ids = sorted_col_ids;  // reuse sorted_col_ids
  auto parent_col_id   = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0),
    cuda::proclaim_return_type<NodeIndexT>(
      [col_ids         = col_ids.begin(),
       parent_node_ids = tree.parent_node_ids.begin()] __device__(size_type node_id) {
        return parent_node_ids[node_id] == parent_node_sentinel ? parent_node_sentinel
                                                                  : col_ids[parent_node_ids[node_id]];
      }));
  auto const list_children_end = thrust::copy_if(
    rmm::exec_policy(stream),
    thrust::make_zip_iterator(thrust::make_counting_iterator<size_type>(0), parent_col_id),
    thrust::make_zip_iterator(thrust::make_counting_iterator<size_type>(0), parent_col_id) +
      num_nodes,
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_zip_iterator(node_ids.begin(), parent_col_ids.begin()),
    [d_ignore_vals     = d_ignore_vals.begin(),
     parent_node_ids   = tree.parent_node_ids.begin(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin()] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      return parent_node_id != parent_node_sentinel and
             column_categories[col_ids[parent_node_id]] == NC_LIST and
             (!d_ignore_vals[col_ids[parent_node_id]]);
    });

  auto const num_list_children =
    list_children_end - thrust::make_zip_iterator(node_ids.begin(), parent_col_ids.begin());
  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             parent_col_ids.begin(),
                             parent_col_ids.begin() + num_list_children,
                             node_ids.begin());
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    num_list_children,
    [node_ids        = node_ids.begin(),
     parent_node_ids = tree.parent_node_ids.begin(),
     parent_col_ids  = parent_col_ids.begin(),
     row_offsets     = row_offsets.begin(),
     d_columns_data  = d_columns_data.begin(),
     num_list_children] __device__(size_type i) {
      auto const node_id        = node_ids[i];
      auto const parent_node_id = parent_node_ids[node_id];
      // scatter to list_offset
      if (i == 0 or parent_node_ids[node_ids[i - 1]] != parent_node_id) {
        d_columns_data[parent_col_ids[i]].child_offsets[row_offsets[parent_node_id]] =
          row_offsets[node_id];
      }
      // last value of list child_offset is its size.
      if (i == num_list_children - 1 or parent_node_ids[node_ids[i + 1]] != parent_node_id) {
        d_columns_data[parent_col_ids[i]].child_offsets[row_offsets[parent_node_id] + 1] =
          row_offsets[node_id] + 1;
      }
    });

  // 5. scan on offsets.
  for (auto& [id, col_ref] : columns) {
    auto& col = col_ref.get();
    if (col.type == json_col_t::StringColumn) {
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                             col.string_offsets.begin(),
                             col.string_offsets.end(),
                             col.string_offsets.begin(),
                             thrust::maximum<json_column::row_offset_t>{});
    } else if (col.type == json_col_t::ListColumn) {
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                             col.child_offsets.begin(),
                             col.child_offsets.end(),
                             col.child_offsets.begin(),
                             thrust::maximum<json_column::row_offset_t>{});
    }
  }
  stream.synchronize();
}

}  // namespace detail
}  // namespace cudf::io::json
