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
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <queue>

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
  print_vec(cudf::detail::make_std_vector_sync(d_gpu_tree.node_categories, stream),
            "node_categories",
            to_cat);
  print_vec(cudf::detail::make_std_vector_sync(d_gpu_tree.parent_node_ids, stream),
            "parent_node_ids",
            to_int);
  print_vec(
    cudf::detail::make_std_vector_sync(d_gpu_tree.node_levels, stream), "node_levels", to_int);
  auto node_range_begin = cudf::detail::make_std_vector_sync(d_gpu_tree.node_range_begin, stream);
  auto node_range_end   = cudf::detail::make_std_vector_sync(d_gpu_tree.node_range_end, stream);
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
  using row_offset_t           = json_column::row_offset_t;
  row_offset_t* string_offsets = nullptr;
  row_offset_t* string_lengths = nullptr;
  row_offset_t* child_offsets  = nullptr;
  bitmask_type* validity       = nullptr;
};

namespace experimental {

std::optional<schema_element> child_schema_element(std::string const& col_name,
                                                   cudf::io::json_reader_options const& options)
{
  return std::visit(
    cudf::detail::visitor_overload{
      [col_name](std::vector<data_type> const& user_dtypes) -> std::optional<schema_element> {
        auto column_index = atol(col_name.data());
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
      [col_name](
        std::map<std::string, schema_element> const& user_dtypes) -> std::optional<schema_element> {
        return (user_dtypes.find(col_name) != std::end(user_dtypes))
                 ? user_dtypes.find(col_name)->second
                 : std::optional<schema_element>{};
      }},
    options.get_dtypes());
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
  auto column_levels = cudf::detail::make_std_vector_async(d_column_tree.node_levels, stream);
  auto column_range_beg =
    cudf::detail::make_std_vector_async(d_column_tree.node_range_begin, stream);
  auto column_range_end = cudf::detail::make_std_vector_async(d_column_tree.node_range_end, stream);
  auto max_row_offsets  = cudf::detail::make_std_vector_async(d_max_row_offsets, stream);
  std::vector<std::string> column_names = copy_strings_to_host_sync(
    input, d_column_tree.node_range_begin, d_column_tree.node_range_end, stream);
  stream.synchronize();

  // 2. generate nested columns tree and its device_memory
  // reorder unique_col_ids w.r.t. column_range_begin for order of column to be in field order.
  std::vector<NodeIndexT> node_to_col_mapping(num_columns);
  std::iota(node_to_col_mapping.begin(), node_to_col_mapping.end(), 0);
  std::sort(node_to_col_mapping.begin(),
            node_to_col_mapping.end(),
            [&column_range_beg](auto const& a, auto const& b) {
              return column_range_beg[a] < column_range_beg[b];
            });
  std::vector<NodeIndexT> col_to_node_mapping(num_columns);
  std::iota(col_to_node_mapping.begin(), col_to_node_mapping.end(), 0);
  std::sort(col_to_node_mapping.begin(),
            col_to_node_mapping.end(),
            [&node_to_col_mapping](auto const& a, auto const& b) {
              return node_to_col_mapping[a] < node_to_col_mapping[b];
            });

  // adjacency list construction
  // (i) remove parent_node_sentinel i.e. parent of root node
  // (ii) parent of each node is first element of adjacency list
  std::vector<std::vector<NodeIndexT>> adj(num_columns);
  for (uint32_t node = 0; node < num_columns; node++) {
    auto col    = unique_col_ids[node];
    auto parent = column_parent_ids[node];
    if (parent != parent_node_sentinel)
      adj[col_to_node_mapping[col]].push_back(col_to_node_mapping[parent]);
  }
  for (uint32_t node = 0; node < num_columns; node++) {
    auto col    = unique_col_ids[node];
    auto parent = column_parent_ids[node];
    if (parent != parent_node_sentinel)
      adj[col_to_node_mapping[parent]].push_back(col_to_node_mapping[col]);
  }

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

  auto initialize_json_columns = [&](auto i, auto& col, auto column_category) {
    if (column_category == NC_ERR || column_category == NC_FN) {
      return;
    } else if (column_category == NC_VAL || column_category == NC_STR) {
      col.string_offsets.resize(max_row_offsets[i] + 1, stream);
      col.string_lengths.resize(max_row_offsets[i] + 1, stream);
      thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), col.string_offsets.begin(), col.string_offsets.end(), 0);
      thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), col.string_lengths.begin(), col.string_lengths.end(), 0);
    } else if (column_category == NC_LIST) {
      col.child_offsets.resize(max_row_offsets[i] + 2, stream);
      thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), col.child_offsets.begin(), col.child_offsets.end(), 0);
    }
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = to_json_col_type(column_category);
  };

  std::vector<bool> is_pruned(num_columns, true);
  std::queue<NodeIndexT> optq;
  if (options.is_enabled_prune_columns()) {
    optq.push(0);
    while (!optq.empty()) {
      auto level_size = optq.size();
      for (uint32_t n = 0; n < level_size; n++) {
        auto node          = optq.front();
        auto col_id        = node_to_col_mapping[node];
        auto parent_col_id = node == 0 ? parent_node_sentinel : node_to_col_mapping[adj[node][0]];
        optq.pop();

        std::string col_name = "";
        if (parent_col_id == parent_node_sentinel) col_name = "root";
        if (column_categories[col_id] == NC_LIST || column_categories[parent_col_id] == NC_LIST) {
          if (is_array_of_arrays && parent_col_id == row_array_parent_col_id)
            col_name = column_names[col_id];
          else
            col_name = std::string(list_child_name);
        } else if (column_categories[col_id] == NC_FN)
          col_name = column_names[col_id];
        else if (column_categories[parent_col_id] == NC_FN)
          col_name = column_names[parent_col_id];

        if (!col_name.empty()) {
          std::optional<schema_element> col_schema = child_schema_element(col_name, options);
          if (col_schema.has_value() || parent_col_id == parent_node_sentinel) {
            is_pruned[node] = false;
            for (uint32_t v = 1; v < adj[node].size(); v++)
              optq.push(adj[node][v]);
          }
        }
      }
    }
  }

  std::vector<bool> is_mixed_type(num_columns, false);
  std::vector<bool> ignore_vals(num_columns, false);
  if(options.is_enabled_mixed_types_as_string()) {
    optq.push(0);
    while(!optq.empty()) {
      auto level_size = optq.size();
      for (uint32_t n = 0; n < level_size; n++) {
        auto node          = optq.front();
        auto col_id        = node_to_col_mapping[node];
        auto parent_col_id = node == 0 ? parent_node_sentinel : node_to_col_mapping[adj[node][0]];
        optq.pop();

        if(column_categories[col_id] == NC_FN || column_categories[col_id] == NC_LIST) {
          bool has_list_child = false;
          bool has_struct_child = false;
          for(uint32_t v = 1; v < adj[node].size(); v++) {
            auto child_node_id = node_to_col_mapping[adj[node][v]];
            if(column_categories[child_node_id] == NC_LIST) has_list_child = true;
            else if(column_categories[child_node_id] == NC_STRUCT) has_struct_child = true;
          }
          if(has_list_child && has_struct_child) is_mixed_type[node] = true;
        }
        for(uint32_t v = 1; v < adj[node].size(); v++)
          optq.push(adj[node][v]);
      }
    }
  }
  else {
    optq.push(0);
    while(!optq.empty()) {
      auto level_size = optq.size();
      for (uint32_t n = 0; n < level_size; n++) {
        auto node          = optq.front();
        auto col_id        = node_to_col_mapping[node];
        auto parent_col_id = node == 0 ? parent_node_sentinel : node_to_col_mapping[adj[node][0]];
        optq.pop();

        if(column_categories[col_id] == NC_FN || column_categories[col_id] == NC_LIST) {
          bool has_list_child = false;
          bool has_struct_child = false;
          NodeIndexT val_child_node = -1;
          for(uint32_t v = 1; v < adj[node].size(); v++) {
            auto child_node_id = node_to_col_mapping[adj[node][v]];
            if(column_categories[child_node_id] == NC_LIST) has_list_child = true;
            else if(column_categories[child_node_id] == NC_STRUCT) has_struct_child = true;
            else if(column_categories[child_node_id] == NC_STR || column_categories[child_node_id] == NC_VAL) val_child_node = adj[node][v];
          }
          CUDF_EXPECTS(!(has_list_child && has_struct_child), "A mix of lists and structs within the same column is not supported");
          if((has_list_child || has_struct_child) && val_child_node != -1) ignore_vals[val_child_node] = true;
        }
        for(uint32_t v = 1; v < adj[node].size(); v++)
          optq.push(adj[node][v]);
      }
    }
}

  std::queue<std::pair<NodeIndexT, std::reference_wrapper<device_json_column>>> bfsq;
  auto columns_data = cudf::detail::make_host_vector<json_column_data>(num_columns, stream);
  bfsq.emplace(std::make_pair(0, std::ref(root)));
  while (!bfsq.empty()) {
    auto level_size = bfsq.size();
    for (uint32_t n = 0; n < level_size; n++) {
      auto [node, parent_col_ref] = bfsq.front();
      auto& parent_col            = parent_col_ref.get();
      auto col_id                 = node_to_col_mapping[node];
      auto parent_col_id = node == 0 ? parent_node_sentinel : node_to_col_mapping[adj[node][0]];
      auto parent_node   = node == 0 ? -1 : adj[node][0];
      bfsq.pop();

      if (!is_pruned[node]) {
        std::string col_name = "";
        if (parent_col_id == parent_node_sentinel) col_name = "root";
        if (column_categories[col_id] == NC_LIST || column_categories[parent_col_id] == NC_LIST) {
          if (is_array_of_arrays && parent_col_id == row_array_parent_col_id)
            col_name = column_names[col_id];
          else
            col_name = std::string(list_child_name);
        } else if (column_categories[col_id] == NC_FN)
          col_name = column_names[col_id];
        else if (column_categories[parent_col_id] == NC_FN)
          col_name = column_names[parent_col_id];

        if (column_categories[col_id] != NC_FN && column_categories[col_id] != NC_ERR && !ignore_vals[node]) {
          device_json_column col(stream, mr);
          if(!is_mixed_type[node]) initialize_json_columns(col_id, col, column_categories[col_id]);
          else {
            initialize_json_columns(col_id, col, NC_STR);
            col.forced_as_string_column = true;
          }
          columns_data[col_id]        = json_column_data{col.string_offsets.data(),
                                                  col.string_lengths.data(),
                                                  col.child_offsets.data(),
                                                  static_cast<bitmask_type*>(col.validity.data())};
          auto inserted = parent_col.child_columns.try_emplace(col_name, std::move(col)).second;
          CUDF_EXPECTS(inserted, "child column insertion failed, duplicate column name in the parent");
          parent_col.column_order.push_back(col_name);
          for (uint32_t v = adj[node][1]; v < adj[node].size(); v++)
            bfsq.emplace(std::make_pair(v, std::ref(col)));
        } else if (is_mixed_type[node]) { // this is for NC_FN
          device_json_column col(stream, mr);
          initialize_json_columns(col_id, col, NC_STR);
          col.forced_as_string_column = true;
          columns_data[col_id]        = json_column_data{col.string_offsets.data(),
                                                  col.string_lengths.data(),
                                                  col.child_offsets.data(),
                                                  static_cast<bitmask_type*>(col.validity.data())};
          auto inserted = parent_col.child_columns.try_emplace(col_name, std::move(col)).second;
          CUDF_EXPECTS(inserted, "child column insertion failed, duplicate column name in the parent");
          parent_col.column_order.push_back(col_name);
          for (uint32_t v = adj[node][1]; v < adj[node].size(); v++)
            bfsq.emplace(std::make_pair(v, std::ref(parent_col)));
        }
      }
    }
  }

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
     d_columns_data    = d_columns_data.begin()] __device__(size_type i) {
      auto const node_category = column_categories[col_ids[i]];
      switch (node_category) {
        case NC_STRUCT: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_LIST: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_STR: [[fallthrough]];
        case NC_VAL:
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
    [parent_node_ids   = tree.parent_node_ids.begin(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin()] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      return parent_node_id != parent_node_sentinel and
             column_categories[col_ids[parent_node_id]] == NC_LIST;
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
  std::queue<std::reference_wrapper<device_json_column>> scanq;
  scanq.emplace(std::ref(root));
  while (!scanq.empty()) {
    auto& col = scanq.front().get();
    scanq.pop();
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

}  // namespace experimental

/**
 * @brief Constructs `d_json_column` from node tree representation
 * Newly constructed columns are insert into `root`'s children.
 * `root` must be a list type.

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

}  // namespace cudf::io::json::detail
