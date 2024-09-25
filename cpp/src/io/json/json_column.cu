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
#include <cudf/io/detail/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace cudf::io::json::detail {

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
                      device_span<NodeIndexT const> original_col_ids,
                      device_span<NodeIndexT const> sorted_col_ids,
                      device_span<NodeIndexT const> ordered_node_ids,
                      device_span<size_type const> row_offsets,
                      bool is_array_of_arrays,
                      NodeIndexT const row_array_parent_col_id,
                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // 1. column count for allocation
  auto const num_columns = thrust::unique_count(
    rmm::exec_policy_nosync(stream), sorted_col_ids.begin(), sorted_col_ids.end());

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
  rmm::device_uvector<TreeDepthT> column_levels(num_columns, stream);  // not required
  rmm::device_uvector<NodeIndexT> parent_col_ids(num_columns, stream);
  rmm::device_uvector<SymbolOffsetT> col_range_begin(num_columns, stream);  // Field names
  rmm::device_uvector<SymbolOffsetT> col_range_end(num_columns, stream);
  rmm::device_uvector<size_type> unique_node_ids(num_columns, stream);
  thrust::unique_by_key_copy(rmm::exec_policy_nosync(stream),
                             sorted_col_ids.begin(),
                             sorted_col_ids.end(),
                             ordered_node_ids.begin(),
                             thrust::make_discard_iterator(),
                             unique_node_ids.begin());

  thrust::copy_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_zip_iterator(
      thrust::make_permutation_iterator(tree.node_levels.begin(), unique_node_ids.begin()),
      thrust::make_permutation_iterator(tree.parent_node_ids.begin(), unique_node_ids.begin()),
      thrust::make_permutation_iterator(tree.node_range_begin.begin(), unique_node_ids.begin()),
      thrust::make_permutation_iterator(tree.node_range_end.begin(), unique_node_ids.begin())),
    unique_node_ids.size(),
    thrust::make_zip_iterator(column_levels.begin(),
                              parent_col_ids.begin(),
                              col_range_begin.begin(),
                              col_range_end.begin()));

  // convert parent_node_ids to parent_col_ids
  thrust::transform(
    rmm::exec_policy_nosync(stream),
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
    auto list_parents_children_max_row_offsets =
      cudf::detail::make_zeroed_device_uvector_async<NodeIndexT>(
        static_cast<std::size_t>(num_columns), stream, cudf::get_current_device_resource_ref());
    thrust::for_each(rmm::exec_policy_nosync(stream),
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
      rmm::exec_policy_nosync(stream),
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
    rmm::exec_policy_nosync(stream),
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
    rmm::exec_policy_nosync(stream),
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

      auto [result_bitmask, null_count] = make_validity(json_col);

      data_type target_type{};
      std::unique_ptr<column> col{};
      if (options.normalize_whitespace && json_col.forced_as_string_column) {
        CUDF_EXPECTS(prune_columns || options.mixed_types_as_string,
                     "Whitespace normalization of nested columns requested as string requires "
                     "either prune_columns or mixed_types_as_string to be enabled");
        auto [normalized_d_input, col_offsets, col_lengths] =
          cudf::io::json::detail::normalize_whitespace(
            d_input, json_col.string_offsets, json_col.string_lengths, stream, mr);
        auto offset_length_it = thrust::make_zip_iterator(col_offsets.begin(), col_lengths.begin());
        target_type           = data_type{type_id::STRING};
        // Convert strings to the inferred data type
        col = parse_data(normalized_d_input.data(),
                         offset_length_it,
                         col_size,
                         target_type,
                         std::move(result_bitmask),
                         null_count,
                         options.view(),
                         stream,
                         mr);
      } else {
        auto offset_length_it = thrust::make_zip_iterator(json_col.string_offsets.begin(),
                                                          json_col.string_lengths.begin());
        if (schema.has_value()) {
#ifdef NJP_DEBUG_PRINT
          std::cout << "-> explicit type: "
                    << (schema.has_value() ? std::to_string(static_cast<int>(schema->type.id()))
                                           : "n/a");
#endif
          target_type = schema.value().type;
        }
        // Infer column type, if we don't have an explicit type for it
        else {
          target_type = cudf::io::detail::infer_data_type(
            options.json_view(), d_input, offset_length_it, col_size, stream);
        }
        // Convert strings to the inferred data type
        col = parse_data(d_input.data(),
                         offset_length_it,
                         col_size,
                         target_type,
                         std::move(result_bitmask),
                         null_count,
                         options.view(),
                         stream,
                         mr);
      }

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
      get_token_stream(d_input, options, stream, cudf::get_current_device_resource_ref());
    // gpu tree generation
    // Note that to normalize whitespaces in nested columns coerced to be string, we need the column
    // to either be of mixed type or we need to request the column to be returned as string by
    // pruning it with the STRING dtype
    return get_tree_representation(
      tokens_gpu,
      token_indices_gpu,
      options.is_enabled_mixed_types_as_string() || options.is_enabled_prune_columns(),
      stream,
      cudf::get_current_device_resource_ref());
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
                                  cudf::get_current_device_resource_ref());

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
