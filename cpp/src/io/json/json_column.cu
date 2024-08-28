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

#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/string_parsing.hpp"
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>

namespace cudf::io::json::detail {

// DEBUG prints
auto to_cat = [](auto v) -> std::string {
  switch (v) {
    case NC_STRUCT: return " S";
    case NC_LIST: return " L";
    case NC_STR: return " \"";
    case NC_VAL: return " V";
    case NC_FN: return " F";
    case NC_ERR: return "ER";
    default: return "UN";
  };
};
auto to_int    = [](auto v) { return std::to_string(static_cast<int>(v)); };
auto print_vec = [](auto const& cpu, auto const name, auto converter) {
  for (auto const& v : cpu)
    printf("%3s,", converter(v).c_str());
  std::cout << name << std::endl;
};

void print_tree(host_span<SymbolT const> input,
                tree_meta_t const& d_gpu_tree,
                rmm::cuda_stream_view stream)
{
  print_vec(cudf::detail::make_host_vector_sync(d_gpu_tree.node_categories, stream),
            "node_categories",
            to_cat);
  print_vec(cudf::detail::make_host_vector_sync(d_gpu_tree.parent_node_ids, stream),
            "parent_node_ids",
            to_int);
  print_vec(
    cudf::detail::make_host_vector_sync(d_gpu_tree.node_levels, stream), "node_levels", to_int);
  auto node_range_begin = cudf::detail::make_host_vector_sync(d_gpu_tree.node_range_begin, stream);
  auto node_range_end   = cudf::detail::make_host_vector_sync(d_gpu_tree.node_range_end, stream);
  print_vec(node_range_begin, "node_range_begin", to_int);
  print_vec(node_range_end, "node_range_end", to_int);
  for (int i = 0; i < int(node_range_begin.size()); i++) {
    printf("%3s ",
           std::string(input.data() + node_range_begin[i], node_range_end[i] - node_range_begin[i])
             .c_str());
  }
  printf(" (JSON)\n");
}

/**
 * @brief Reduces node tree representation to column tree representation.
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
std::tuple<tree_meta_t, rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
reduce_to_column_tree(tree_meta_t& tree,
                      device_span<NodeIndexT> original_col_ids,
                      device_span<NodeIndexT> sorted_col_ids,
                      device_span<NodeIndexT> ordered_node_ids,
                      device_span<size_type> row_offsets,
                      bool is_array_of_arrays,
                      NodeIndexT const row_array_parent_col_id,
                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // 1. column count for allocation
  auto const num_columns =
    thrust::unique_count(rmm::exec_policy(stream), sorted_col_ids.begin(), sorted_col_ids.end());

  // 2. reduce_by_key {col_id}, {row_offset}, max.
  rmm::device_uvector<NodeIndexT> unique_col_ids(num_columns, stream);
  rmm::device_uvector<size_type> max_row_offsets(num_columns, stream);
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

  // 4. unique_copy parent_node_ids, ranges
  rmm::device_uvector<TreeDepthT> column_levels(0, stream);  // not required
  rmm::device_uvector<NodeIndexT> parent_col_ids(num_columns, stream);
  rmm::device_uvector<SymbolOffsetT> col_range_begin(num_columns, stream);  // Field names
  rmm::device_uvector<SymbolOffsetT> col_range_end(num_columns, stream);
  rmm::device_uvector<size_type> unique_node_ids(num_columns, stream);
  thrust::unique_by_key_copy(rmm::exec_policy(stream),
                             sorted_col_ids.begin(),
                             sorted_col_ids.end(),
                             ordered_node_ids.begin(),
                             thrust::make_discard_iterator(),
                             unique_node_ids.begin());
  thrust::copy_n(
    rmm::exec_policy(stream),
    thrust::make_zip_iterator(
      thrust::make_permutation_iterator(tree.parent_node_ids.begin(), unique_node_ids.begin()),
      thrust::make_permutation_iterator(tree.node_range_begin.begin(), unique_node_ids.begin()),
      thrust::make_permutation_iterator(tree.node_range_end.begin(), unique_node_ids.begin())),
    unique_node_ids.size(),
    thrust::make_zip_iterator(
      parent_col_ids.begin(), col_range_begin.begin(), col_range_end.begin()));

  // convert parent_node_ids to parent_col_ids
  thrust::transform(
    rmm::exec_policy(stream),
    parent_col_ids.begin(),
    parent_col_ids.end(),
    parent_col_ids.begin(),
    [col_ids = original_col_ids.begin()] __device__(auto parent_node_id) -> size_type {
      return parent_node_id == parent_node_sentinel ? parent_node_sentinel
                                                    : col_ids[parent_node_id];
    });

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
                     unique_col_ids.begin(),
                     unique_col_ids.end(),
                     [column_categories = column_categories.begin(),
                      parent_col_ids    = parent_col_ids.begin(),
                      max_row_offsets   = max_row_offsets.begin(),
                      list_parents_children_max_row_offsets =
                        list_parents_children_max_row_offsets.begin()] __device__(auto col_id) {
                       auto parent_col_id = parent_col_ids[col_id];
                       if (parent_col_id != parent_node_sentinel and
                           column_categories[parent_col_id] == node_t::NC_LIST) {
                         cuda::atomic_ref<NodeIndexT, cuda::thread_scope_device> ref{
                           *(list_parents_children_max_row_offsets + parent_col_id)};
                         ref.fetch_max(max_row_offsets[col_id], cuda::std::memory_order_relaxed);
                       }
                     });
    thrust::gather_if(
      rmm::exec_policy(stream),
      parent_col_ids.begin(),
      parent_col_ids.end(),
      parent_col_ids.begin(),
      list_parents_children_max_row_offsets.begin(),
      max_row_offsets.begin(),
      [column_categories = column_categories.begin()] __device__(size_type parent_col_id) {
        return parent_col_id != parent_node_sentinel and
               column_categories[parent_col_id] == node_t::NC_LIST;
      });
  }

  // copy lists' max_row_offsets to children.
  // all structs should have same size.
  thrust::transform_if(
    rmm::exec_policy(stream),
    unique_col_ids.begin(),
    unique_col_ids.end(),
    max_row_offsets.begin(),
    [column_categories = column_categories.begin(),
     is_non_list_parent,
     parent_col_ids  = parent_col_ids.begin(),
     max_row_offsets = max_row_offsets.begin()] __device__(size_type col_id) {
      auto parent_col_id = parent_col_ids[col_id];
      // condition is true if parent is not a list, or sentinel/root
      while (is_non_list_parent(parent_col_id)) {
        col_id        = parent_col_id;
        parent_col_id = parent_col_ids[parent_col_id];
      }
      return max_row_offsets[col_id];
    },
    [column_categories = column_categories.begin(),
     is_non_list_parent,
     parent_col_ids = parent_col_ids.begin()] __device__(size_type col_id) {
      auto parent_col_id = parent_col_ids[col_id];
      // condition is true if parent is not a list, or sentinel/root
      return is_non_list_parent(parent_col_id);
    });

  // For Struct and List (to avoid copying entire strings when mixed type as string is enabled)
  thrust::transform_if(
    rmm::exec_policy(stream),
    col_range_begin.begin(),
    col_range_begin.end(),
    column_categories.begin(),
    col_range_end.begin(),
    [] __device__(auto i) { return i + 1; },
    [] __device__(NodeT type) { return type == NC_STRUCT || type == NC_LIST; });

  return std::tuple{tree_meta_t{std::move(column_categories),
                                std::move(parent_col_ids),
                                std::move(column_levels),
                                std::move(col_range_begin),
                                std::move(col_range_end)},
                    std::move(unique_col_ids),
                    std::move(max_row_offsets)};
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
    auto const h_chars = cudf::detail::make_host_vector_async<char>(
      cudf::device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)), stream);
    auto const h_offsets = cudf::detail::make_host_vector_async(
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
 * @brief Holds member data pointers of `d_json_column`
 *
 */
struct json_column_data {
  using row_offset_t = json_column::row_offset_t;
  row_offset_t* string_offsets;
  row_offset_t* string_lengths;
  row_offset_t* child_offsets;
  bitmask_type* validity;
};

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
  auto unique_col_ids = cudf::detail::make_host_vector_async(d_unique_col_ids, stream);
  auto column_categories =
    cudf::detail::make_host_vector_async(d_column_tree.node_categories, stream);
  auto const column_parent_ids =
    cudf::detail::make_host_vector_async(d_column_tree.parent_node_ids, stream);
  auto column_range_beg =
    cudf::detail::make_host_vector_async(d_column_tree.node_range_begin, stream);
  auto const max_row_offsets = cudf::detail::make_host_vector_async(d_max_row_offsets, stream);
  std::vector<std::string> column_names = copy_strings_to_host_sync(
    input, d_column_tree.node_range_begin, d_column_tree.node_range_end, stream);
  // array of arrays column names
  if (is_array_of_arrays) {
    TreeDepthT const row_array_children_level = is_enabled_lines ? 1 : 2;
    auto values_column_indices =
      get_values_column_indices(row_array_children_level, tree, col_ids, num_columns, stream);
    auto h_values_column_indices =
      cudf::detail::make_host_vector_sync(values_column_indices, stream);
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

  auto initialize_json_columns = [&](auto i, auto& col, auto column_category) {
    if (column_category == NC_ERR || column_category == NC_FN) {
      return;
    } else if (column_category == NC_VAL || column_category == NC_STR) {
      col.string_offsets.resize(max_row_offsets[i] + 1, stream);
      col.string_lengths.resize(max_row_offsets[i] + 1, stream);
      init_to_zero(col.string_offsets);
      init_to_zero(col.string_lengths);
    } else if (column_category == NC_LIST) {
      col.child_offsets.resize(max_row_offsets[i] + 2, stream);
      init_to_zero(col.child_offsets);
    }
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = to_json_col_type(column_category);
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

  auto const is_str_column_all_nulls = [&, &column_tree = d_column_tree]() {
    if (is_enabled_mixed_types_as_string) {
      return cudf::detail::make_host_vector_sync(
        is_all_nulls_each_column(input, column_tree, tree, col_ids, options, stream), stream);
    }
    return cudf::detail::make_empty_host_vector<uint8_t>(0, stream);
  }();

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

    auto this_column_category = column_categories[this_col_id];
    if (is_enabled_mixed_types_as_string) {
      // get path of this column, check if it is a struct/list forced as string, and enforce it
      auto const nt                             = tree_path.get_path(this_col_id);
      std::optional<data_type> const user_dtype = get_path_data_type(nt, options);
      if ((column_categories[this_col_id] == NC_STRUCT or
           column_categories[this_col_id] == NC_LIST) and
          user_dtype.has_value() and user_dtype.value().id() == type_id::STRING) {
        is_mixed_type_column[this_col_id] = 1;
        this_column_category              = NC_STR;
      }
    }

    CUDF_EXPECTS(parent_col.child_columns.count(name) == 0, "duplicate column name: " + name);
    // move into parent
    device_json_column col(stream, mr);
    initialize_json_columns(this_col_id, col, this_column_category);
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

std::pair<std::unique_ptr<column>, std::vector<column_name_info>> device_json_column_to_cudf_column(
  device_json_column& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::parse_options const& options,
  bool prune_columns,
  std::optional<schema_element> schema,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto validity_size_check = [](device_json_column& json_col) {
    CUDF_EXPECTS(json_col.validity.size() >= bitmask_allocation_size_bytes(json_col.num_rows),
                 "valid_count is too small");
  };
  auto make_validity = [stream, validity_size_check](
                         device_json_column& json_col) -> std::pair<rmm::device_buffer, size_type> {
    validity_size_check(json_col);
    auto null_count = cudf::detail::null_count(
      static_cast<bitmask_type*>(json_col.validity.data()), 0, json_col.num_rows, stream);
    // full null_mask is always required for parse_data
    return {std::move(json_col.validity), null_count};
    // Note: json_col modified here, moves this memory
  };

  auto get_child_schema = [schema](auto child_name) -> std::optional<schema_element> {
    if (schema.has_value()) {
      auto const result = schema.value().child_types.find(child_name);
      if (result != std::end(schema.value().child_types)) { return result->second; }
    }
    return {};
  };

  switch (json_col.type) {
    case json_col_t::StringColumn: {
      // move string_offsets to GPU and transform to string column
      auto const col_size      = json_col.string_offsets.size();
      using char_length_pair_t = thrust::pair<char const*, size_type>;
      CUDF_EXPECTS(json_col.string_offsets.size() == json_col.string_lengths.size(),
                   "string offset, string length mismatch");
      rmm::device_uvector<char_length_pair_t> d_string_data(col_size, stream);
      // TODO how about directly storing pair<char*, size_t> in json_column?
      auto offset_length_it =
        thrust::make_zip_iterator(json_col.string_offsets.begin(), json_col.string_lengths.begin());

      data_type target_type{};

      if (schema.has_value()) {
#ifdef NJP_DEBUG_PRINT
        std::cout << "-> explicit type: "
                  << (schema.has_value() ? std::to_string(static_cast<int>(schema->type.id()))
                                         : "n/a");
#endif
        target_type = schema.value().type;
      } else if (json_col.forced_as_string_column) {
        target_type = data_type{type_id::STRING};
      }
      // Infer column type, if we don't have an explicit type for it
      else {
        target_type = cudf::io::detail::infer_data_type(
          options.json_view(), d_input, offset_length_it, col_size, stream);
      }

      auto [result_bitmask, null_count] = make_validity(json_col);
      // Convert strings to the inferred data type
      auto col = parse_data(d_input.data(),
                            offset_length_it,
                            col_size,
                            target_type,
                            std::move(result_bitmask),
                            null_count,
                            options.view(),
                            stream,
                            mr);

      // Reset nullable if we do not have nulls
      // This is to match the existing JSON reader's behaviour:
      // - Non-string columns will always be returned as nullable
      // - String columns will be returned as nullable, iff there's at least one null entry
      if (col->null_count() == 0) { col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0); }

      // For string columns return ["offsets", "char"] schema
      if (target_type.id() == type_id::STRING) {
        return {std::move(col), std::vector<column_name_info>{{"offsets"}, {"chars"}}};
      }
      // Non-string leaf-columns (e.g., numeric) do not have child columns in the schema
      return {std::move(col), std::vector<column_name_info>{}};
    }
    case json_col_t::StructColumn: {
      std::vector<std::unique_ptr<column>> child_columns;
      std::vector<column_name_info> column_names{};
      size_type num_rows{json_col.num_rows};
      // Create children columns
      for (auto const& col_name : json_col.column_order) {
        auto const& col = json_col.child_columns.find(col_name);
        column_names.emplace_back(col->first);
        auto& child_col           = col->second;
        auto child_schema_element = get_child_schema(col_name);
        if (!prune_columns or child_schema_element.has_value()) {
          auto [child_column, names] = device_json_column_to_cudf_column(
            child_col, d_input, options, prune_columns, child_schema_element, stream, mr);
          CUDF_EXPECTS(num_rows == child_column->size(),
                       "All children columns must have the same size");
          child_columns.push_back(std::move(child_column));
          column_names.back().children = names;
        }
      }
      auto [result_bitmask, null_count] = make_validity(json_col);
      // The null_mask is set after creation of struct column is to skip the superimpose_nulls and
      // null validation applied in make_structs_column factory, which is not needed for json
      auto ret_col = make_structs_column(num_rows, std::move(child_columns), 0, {}, stream, mr);
      if (null_count != 0) { ret_col->set_null_mask(std::move(result_bitmask), null_count); }
      return {std::move(ret_col), column_names};
    }
    case json_col_t::ListColumn: {
      size_type num_rows = json_col.child_offsets.size() - 1;
      std::vector<column_name_info> column_names{};
      column_names.emplace_back("offsets");
      column_names.emplace_back(
        json_col.child_columns.empty() ? list_child_name : json_col.child_columns.begin()->first);

      // Note: json_col modified here, reuse the memory
      auto offsets_column = std::make_unique<column>(data_type{type_id::INT32},
                                                     num_rows + 1,
                                                     json_col.child_offsets.release(),
                                                     rmm::device_buffer{},
                                                     0);
      // Create children column
      auto child_schema_element = json_col.child_columns.empty()
                                    ? std::optional<schema_element>{}
                                    : get_child_schema(json_col.child_columns.begin()->first);
      auto [child_column, names] =
        json_col.child_columns.empty() or (prune_columns and !child_schema_element.has_value())
          ? std::pair<std::unique_ptr<column>,
                      // EMPTY type could not used because gather throws exception on EMPTY type.
                      std::vector<column_name_info>>{std::make_unique<column>(
                                                       data_type{type_id::INT8},
                                                       0,
                                                       rmm::device_buffer{},
                                                       rmm::device_buffer{},
                                                       0),
                                                     std::vector<column_name_info>{}}
          : device_json_column_to_cudf_column(json_col.child_columns.begin()->second,
                                              d_input,
                                              options,
                                              prune_columns,
                                              child_schema_element,
                                              stream,
                                              mr);
      column_names.back().children      = names;
      auto [result_bitmask, null_count] = make_validity(json_col);
      auto ret_col                      = make_lists_column(num_rows,
                                       std::move(offsets_column),
                                       std::move(child_column),
                                       0,
                                       rmm::device_buffer{0, stream, mr},
                                       stream,
                                       mr);
      // The null_mask is set after creation of list column is to skip the purge_nonempty_nulls and
      // null validation applied in make_lists_column factory, which is not needed for json
      // parent column cannot be null when its children is non-empty in JSON
      if (null_count != 0) { ret_col->set_null_mask(std::move(result_bitmask), null_count); }
      return {std::move(ret_col), std::move(column_names)};
    }
    default: CUDF_FAIL("Unsupported column type"); break;
  }
}

table_with_metadata device_parse_nested_json(device_span<SymbolT const> d_input,
                                             cudf::io::json_reader_options const& options,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto gpu_tree = [&]() {
    // Parse the JSON and get the token stream
    const auto [tokens_gpu, token_indices_gpu] =
      get_token_stream(d_input, options, stream, rmm::mr::get_current_device_resource());
    // gpu tree generation
    return get_tree_representation(tokens_gpu,
                                   token_indices_gpu,
                                   options.is_enabled_mixed_types_as_string(),
                                   stream,
                                   rmm::mr::get_current_device_resource());
  }();  // IILE used to free memory of token data.
#ifdef NJP_DEBUG_PRINT
  auto h_input = cudf::detail::make_host_vector_async(d_input, stream);
  print_tree(h_input, gpu_tree, stream);
#endif

  bool const is_array_of_arrays = [&]() {
    std::array<node_t, 2> h_node_categories = {NC_ERR, NC_ERR};
    auto const size_to_copy                 = std::min(size_t{2}, gpu_tree.node_categories.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_node_categories.data(),
                                  gpu_tree.node_categories.data(),
                                  sizeof(node_t) * size_to_copy,
                                  cudaMemcpyDefault,
                                  stream.value()));
    stream.synchronize();
    if (options.is_enabled_lines()) return h_node_categories[0] == NC_LIST;
    return h_node_categories[0] == NC_LIST and h_node_categories[1] == NC_LIST;
  }();

  auto [gpu_col_id, gpu_row_offsets] =
    records_orient_tree_traversal(d_input,
                                  gpu_tree,
                                  is_array_of_arrays,
                                  options.is_enabled_lines(),
                                  stream,
                                  rmm::mr::get_current_device_resource());

  device_json_column root_column(stream, mr);
  root_column.type = json_col_t::ListColumn;
  root_column.child_offsets.resize(2, stream);
  thrust::fill(rmm::exec_policy(stream),
               root_column.child_offsets.begin(),
               root_column.child_offsets.end(),
               0);

  // Get internal JSON column
  make_device_json_column(d_input,
                          gpu_tree,
                          gpu_col_id,
                          gpu_row_offsets,
                          root_column,
                          is_array_of_arrays,
                          options,
                          stream,
                          mr);

  // data_root refers to the root column of the data represented by the given JSON string
  auto& data_root =
    options.is_enabled_lines() ? root_column : root_column.child_columns.begin()->second;

  // Zero row entries
  if (data_root.type == json_col_t::ListColumn && data_root.child_columns.empty()) {
    return table_with_metadata{std::make_unique<table>(std::vector<std::unique_ptr<column>>{})};
  }

  // Verify that we were in fact given a list of structs (or in JSON speech: an array of objects)
  auto constexpr single_child_col_count = 1;
  CUDF_EXPECTS(data_root.type == json_col_t::ListColumn and
                 data_root.child_columns.size() == single_child_col_count and
                 data_root.child_columns.begin()->second.type ==
                   (is_array_of_arrays ? json_col_t::ListColumn : json_col_t::StructColumn),
               "Input needs to be an array of arrays or an array of (nested) objects");

  // Slice off the root list column, which has only a single row that contains all the structs
  auto& root_struct_col = data_root.child_columns.begin()->second;

  // Initialize meta data to be populated while recursing through the tree of columns
  std::vector<std::unique_ptr<column>> out_columns;
  std::vector<column_name_info> out_column_names;
  auto parse_opt = parsing_options(options, stream);

  // Iterate over the struct's child columns and convert to cudf column
  size_type column_index = 0;
  for (auto const& col_name : root_struct_col.column_order) {
    auto& json_col = root_struct_col.child_columns.find(col_name)->second;

    std::optional<schema_element> child_schema_element = std::visit(
      cudf::detail::visitor_overload{
        [column_index](std::vector<data_type> const& user_dtypes) -> std::optional<schema_element> {
          return (static_cast<std::size_t>(column_index) < user_dtypes.size())
                   ? std::optional<schema_element>{{user_dtypes[column_index]}}
                   : std::optional<schema_element>{};
        },
        [col_name](
          std::map<std::string, data_type> const& user_dtypes) -> std::optional<schema_element> {
          return (user_dtypes.find(col_name) != std::end(user_dtypes))
                   ? std::optional<schema_element>{{user_dtypes.find(col_name)->second}}
                   : std::optional<schema_element>{};
        },
        [col_name](std::map<std::string, schema_element> const& user_dtypes)
          -> std::optional<schema_element> {
          return (user_dtypes.find(col_name) != std::end(user_dtypes))
                   ? user_dtypes.find(col_name)->second
                   : std::optional<schema_element>{};
        }},
      options.get_dtypes());
#ifdef NJP_DEBUG_PRINT
    auto debug_schema_print = [](auto ret) {
      std::cout << ", type id: "
                << (ret.has_value() ? std::to_string(static_cast<int>(ret->type.id())) : "n/a")
                << ", with " << (ret.has_value() ? ret->child_types.size() : 0) << " children"
                << "\n";
    };
    std::visit(
      cudf::detail::visitor_overload{[column_index](std::vector<data_type> const&) {
                                       std::cout << "Column by index: #" << column_index;
                                     },
                                     [col_name](std::map<std::string, data_type> const&) {
                                       std::cout << "Column by flat name: '" << col_name;
                                     },
                                     [col_name](std::map<std::string, schema_element> const&) {
                                       std::cout << "Column by nested name: #" << col_name;
                                     }},
      options.get_dtypes());
    debug_schema_print(child_schema_element);
#endif

    if (!options.is_enabled_prune_columns() or child_schema_element.has_value()) {
      // Get this JSON column's cudf column and schema info, (modifies json_col)
      auto [cudf_col, col_name_info] =
        device_json_column_to_cudf_column(json_col,
                                          d_input,
                                          parse_opt,
                                          options.is_enabled_prune_columns(),
                                          child_schema_element,
                                          stream,
                                          mr);
      // Insert this column's name into the schema
      out_column_names.emplace_back(col_name);
      // TODO: RangeIndex as DataFrame.columns names for array of arrays
      // if (is_array_of_arrays) {
      //   col_name_info.back().name = "";
      // }

      out_column_names.back().children = std::move(col_name_info);
      out_columns.emplace_back(std::move(cudf_col));

      column_index++;
    }
  }

  return table_with_metadata{std::make_unique<table>(std::move(out_columns)), {out_column_names}};
}

}  // namespace cudf::io::json::detail
