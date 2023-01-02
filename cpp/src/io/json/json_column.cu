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

#include "nested_json.hpp"
#include <io/utilities/parsing_utils.cuh>
#include <io/utilities/type_inference.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/io/detail/data_casting.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
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

namespace cudf::io::json {
namespace detail {

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
  print_vec(cudf::detail::make_std_vector_async(d_gpu_tree.node_categories, stream),
            "node_categories",
            to_cat);
  print_vec(cudf::detail::make_std_vector_async(d_gpu_tree.parent_node_ids, stream),
            "parent_node_ids",
            to_int);
  print_vec(
    cudf::detail::make_std_vector_async(d_gpu_tree.node_levels, stream), "node_levels", to_int);
  auto node_range_begin = cudf::detail::make_std_vector_async(d_gpu_tree.node_range_begin, stream);
  auto node_range_end   = cudf::detail::make_std_vector_async(d_gpu_tree.node_range_end, stream);
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
 * @param col_ids Column ids of nodes
 * @param row_offsets Row offsets of nodes
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A tuple of column tree representation of JSON string, column ids of columns, and
 * max row offsets of columns
 */
std::tuple<tree_meta_t, rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
reduce_to_column_tree(tree_meta_t& tree,
                      device_span<NodeIndexT> col_ids,
                      device_span<size_type> row_offsets,
                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  //   1. sort_by_key {col_id}, {row_offset} stable
  rmm::device_uvector<NodeIndexT> node_ids(row_offsets.size(), stream);
  thrust::sequence(rmm::exec_policy(stream), node_ids.begin(), node_ids.end());
  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             col_ids.begin(),
                             col_ids.end(),
                             thrust::make_zip_iterator(node_ids.begin(), row_offsets.begin()));
  auto num_columns = thrust::unique_count(rmm::exec_policy(stream), col_ids.begin(), col_ids.end());

  // 2. reduce_by_key {col_id}, {row_offset}, max.
  rmm::device_uvector<NodeIndexT> unique_col_ids(num_columns, stream);
  rmm::device_uvector<size_type> max_row_offsets(num_columns, stream);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        col_ids.begin(),
                        col_ids.end(),
                        row_offsets.begin(),
                        unique_col_ids.begin(),
                        max_row_offsets.begin(),
                        thrust::equal_to<size_type>(),
                        thrust::maximum<size_type>());

  // 3. reduce_by_key {col_id}, {node_categories} - custom opp (*+v=*, v+v=v, *+#=E)
  rmm::device_uvector<NodeT> column_categories(num_columns, stream);
  thrust::reduce_by_key(
    rmm::exec_policy(stream),
    col_ids.begin(),
    col_ids.end(),
    thrust::make_permutation_iterator(tree.node_categories.begin(), node_ids.begin()),
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
                             col_ids.begin(),
                             col_ids.end(),
                             node_ids.begin(),
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

  // Restore the order
  {
    // use scatter to restore the order
    rmm::device_uvector<NodeIndexT> temp_col_ids(col_ids.size(), stream);
    rmm::device_uvector<size_type> temp_row_offsets(row_offsets.size(), stream);
    thrust::scatter(rmm::exec_policy(stream),
                    thrust::make_zip_iterator(col_ids.begin(), row_offsets.begin()),
                    thrust::make_zip_iterator(col_ids.end(), row_offsets.end()),
                    node_ids.begin(),
                    thrust::make_zip_iterator(temp_col_ids.begin(), temp_row_offsets.begin()));
    thrust::copy(rmm::exec_policy(stream),
                 thrust::make_zip_iterator(temp_col_ids.begin(), temp_row_offsets.begin()),
                 thrust::make_zip_iterator(temp_col_ids.end(), temp_row_offsets.end()),
                 thrust::make_zip_iterator(col_ids.begin(), row_offsets.begin()));
  }

  // convert parent_node_ids to parent_col_ids
  thrust::transform(rmm::exec_policy(stream),
                    parent_col_ids.begin(),
                    parent_col_ids.end(),
                    parent_col_ids.begin(),
                    [col_ids = col_ids.begin()] __device__(auto parent_node_id) -> size_type {
                      return parent_node_id == parent_node_sentinel ? parent_node_sentinel
                                                                    : col_ids[parent_node_id];
                    });

  // copy lists' max_row_offsets to children.
  // all structs should have same size.
  thrust::transform_if(
    rmm::exec_policy(stream),
    unique_col_ids.begin(),
    unique_col_ids.end(),
    max_row_offsets.begin(),
    [column_categories = column_categories.begin(),
     parent_col_ids    = parent_col_ids.begin(),
     max_row_offsets   = max_row_offsets.begin()] __device__(size_type col_id) {
      auto parent_col_id = parent_col_ids[col_id];
      while (parent_col_id != parent_node_sentinel and
             column_categories[parent_col_id] != node_t::NC_LIST) {
        col_id        = parent_col_id;
        parent_col_id = parent_col_ids[parent_col_id];
      }
      return max_row_offsets[col_id];
    },
    [column_categories = column_categories.begin(),
     parent_col_ids    = parent_col_ids.begin()] __device__(size_type col_id) {
      auto parent_col_id = parent_col_ids[col_id];
      return parent_col_id != parent_node_sentinel and
             (column_categories[parent_col_id] != node_t::NC_LIST);
      // Parent is not a list, or sentinel/root
    });

  return std::tuple{tree_meta_t{std::move(column_categories),
                                std::move(parent_col_ids),
                                std::move(column_levels),
                                std::move(col_range_begin),
                                std::move(col_range_end)},
                    std::move(unique_col_ids),
                    std::move(max_row_offsets)};
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
std::vector<std::string> copy_strings_to_host(device_span<SymbolT const> input,
                                              device_span<SymbolOffsetT const> node_range_begin,
                                              device_span<SymbolOffsetT const> node_range_end,
                                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto const num_strings = node_range_begin.size();
  rmm::device_uvector<thrust::pair<const char*, size_type>> string_views(num_strings, stream);
  auto d_offset_pairs = thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  thrust::transform(rmm::exec_policy(stream),
                    d_offset_pairs,
                    d_offset_pairs + num_strings,
                    string_views.begin(),
                    [data = input.data()] __device__(auto const& offsets) {
                      // Note: first character for non-field columns
                      return thrust::make_pair(
                        data + thrust::get<0>(offsets),
                        static_cast<size_type>(thrust::get<1>(offsets) - thrust::get<0>(offsets)));
                    });
  auto d_column_names = cudf::make_strings_column(string_views, stream);
  auto to_host        = [](auto const& col) {
    if (col.is_empty()) return std::vector<std::string>{};
    auto const scv     = cudf::strings_column_view(col);
    auto const h_chars = cudf::detail::make_std_vector_sync<char>(
      cudf::device_span<char const>(scv.chars().data<char>(), scv.chars().size()),
      cudf::get_default_stream());
    auto const h_offsets = cudf::detail::make_std_vector_sync(
      cudf::device_span<cudf::offset_type const>(
        scv.offsets().data<cudf::offset_type>() + scv.offset(), scv.size() + 1),
      cudf::get_default_stream());

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
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the device memory
 * of child_offets and validity members of `d_json_column`
 */
void make_device_json_column(device_span<SymbolT const> input,
                             tree_meta_t& tree,
                             device_span<NodeIndexT> col_ids,
                             device_span<size_type> row_offsets,
                             device_json_column& root,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // 1. gather column information.
  auto [d_column_tree, d_unique_col_ids, d_max_row_offsets] =
    reduce_to_column_tree(tree, col_ids, row_offsets, stream);
  auto num_columns    = d_unique_col_ids.size();
  auto unique_col_ids = cudf::detail::make_std_vector_async(d_unique_col_ids, stream);
  auto column_categories =
    cudf::detail::make_std_vector_async(d_column_tree.node_categories, stream);
  auto column_parent_ids =
    cudf::detail::make_std_vector_async(d_column_tree.parent_node_ids, stream);
  auto column_range_beg =
    cudf::detail::make_std_vector_async(d_column_tree.node_range_begin, stream);
  auto max_row_offsets = cudf::detail::make_std_vector_async(d_max_row_offsets, stream);
  std::vector<std::string> column_names = copy_strings_to_host(
    input, d_column_tree.node_range_begin, d_column_tree.node_range_end, stream);

  auto to_json_col_type = [](auto category) {
    switch (category) {
      case NC_STRUCT: return json_col_t::StructColumn;
      case NC_LIST: return json_col_t::ListColumn;
      case NC_STR:
      case NC_VAL: return json_col_t::StringColumn;
      default: return json_col_t::Unknown;
    }
  };
  auto init_to_zero = [stream](auto& v) {
    thrust::uninitialized_fill(rmm::exec_policy(stream), v.begin(), v.end(), 0);
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
    col.validity.resize(bitmask_allocation_size_bytes(max_row_offsets[i] + 1), stream);
    init_to_zero(col.validity);
    col.type = to_json_col_type(column_categories[i]);
  };

  // 2. generate nested columns tree and its device_memory
  // reorder unique_col_ids w.r.t. column_range_begin for order of column to be in field order.
  auto h_range_col_id_it =
    thrust::make_zip_iterator(column_range_beg.begin(), unique_col_ids.begin());
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return thrust::get<0>(a) < thrust::get<0>(b);
  });

  // use hash map because we may skip field name's col_ids
  std::unordered_map<NodeIndexT, std::reference_wrapper<device_json_column>> columns;
  // map{parent_col_id, child_col_name}> = child_col_id, used for null value column tracking
  std::map<std::pair<NodeIndexT, std::string>, NodeIndexT> mapped_columns;
  // find column_ids which are values, but should be ignored in validity
  std::vector<uint8_t> ignore_vals(num_columns, 0);
  columns.try_emplace(parent_node_sentinel, std::ref(root));

  for (auto const this_col_id : unique_col_ids) {
    if (column_categories[this_col_id] == NC_ERR || column_categories[this_col_id] == NC_FN) {
      continue;
    }
    // Struct, List, String, Value
    std::string name   = "";
    auto parent_col_id = column_parent_ids[this_col_id];
    if (parent_col_id == parent_node_sentinel || column_categories[parent_col_id] == NC_LIST) {
      name = list_child_name;
    } else if (column_categories[parent_col_id] == NC_FN) {
      auto field_name_col_id = parent_col_id;
      parent_col_id          = column_parent_ids[parent_col_id];
      name                   = column_names[field_name_col_id];
    } else {
      CUDF_FAIL("Unexpected parent column category");
    }
    // If the child is already found,
    // replace if this column is a nested column and the existing was a value column
    // ignore this column if this column is a value column and the existing was a nested column
    auto it = columns.find(parent_col_id);
    CUDF_EXPECTS(it != columns.end(), "Parent column not found");
    auto& parent_col = it->second.get();
    bool replaced    = false;
    if (mapped_columns.count({parent_col_id, name}) > 0) {
      if (column_categories[this_col_id] == NC_VAL || column_categories[this_col_id] == NC_STR) {
        ignore_vals[this_col_id] = 1;
        continue;
      }
      auto old_col_id = mapped_columns[{parent_col_id, name}];
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
    CUDF_EXPECTS(parent_col.child_columns.count(name) == 0, "duplicate column name");
    // move into parent
    device_json_column col(stream, mr);
    initialize_json_columns(this_col_id, col);
    auto inserted = parent_col.child_columns.try_emplace(name, std::move(col)).second;
    CUDF_EXPECTS(inserted, "child column insertion failed, duplicate column name in the parent");
    if (not replaced) parent_col.column_order.push_back(name);
    columns.try_emplace(this_col_id, std::ref(parent_col.child_columns.at(name)));
    mapped_columns.try_emplace(std::make_pair(parent_col_id, name), this_col_id);
  }
  // restore unique_col_ids order
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return thrust::get<1>(a) < thrust::get<1>(b);
  });
  // move columns data to device.
  std::vector<json_column_data> columns_data(num_columns);
  for (auto& [col_id, col_ref] : columns) {
    if (col_id == parent_node_sentinel) continue;
    auto& col            = col_ref.get();
    columns_data[col_id] = json_column_data{col.string_offsets.data(),
                                            col.string_lengths.data(),
                                            col.child_offsets.data(),
                                            col.validity.data()};
  }

  // 3. scatter string offsets to respective columns, set validity bits
  auto d_ignore_vals  = cudf::detail::make_device_uvector_async(ignore_vals, stream);
  auto d_columns_data = cudf::detail::make_device_uvector_async(columns_data, stream);
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::counting_iterator<size_type>(0),
    col_ids.size(),
    [node_categories = tree.node_categories.begin(),
     col_ids         = col_ids.begin(),
     row_offsets     = row_offsets.begin(),
     range_begin     = tree.node_range_begin.begin(),
     range_end       = tree.node_range_end.begin(),
     d_ignore_vals   = d_ignore_vals.begin(),
     d_columns_data  = d_columns_data.begin()] __device__(size_type i) {
      switch (node_categories[i]) {
        case NC_STRUCT: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_LIST: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_VAL:
        case NC_STR:
          if (d_ignore_vals[col_ids[i]]) break;
          set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]);
          d_columns_data[col_ids[i]].string_offsets[row_offsets[i]] = range_begin[i];
          d_columns_data[col_ids[i]].string_lengths[row_offsets[i]] = range_end[i] - range_begin[i];
          break;
        default: break;
      }
    });

  // 4. scatter List offset
  //   sort_by_key {col_id}, {node_id}
  //   unique_copy_by_key {parent_node_id} {row_offset} to
  //   col[parent_col_id].child_offsets[row_offset[parent_node_id]]

  rmm::device_uvector<NodeIndexT> original_col_ids(col_ids.size(), stream);  // make a copy
  thrust::copy(rmm::exec_policy(stream), col_ids.begin(), col_ids.end(), original_col_ids.begin());
  rmm::device_uvector<size_type> node_ids(row_offsets.size(), stream);
  thrust::sequence(rmm::exec_policy(stream), node_ids.begin(), node_ids.end());
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), col_ids.begin(), col_ids.end(), node_ids.begin());

  auto ordered_parent_node_ids =
    thrust::make_permutation_iterator(tree.parent_node_ids.begin(), node_ids.begin());
  auto ordered_row_offsets =
    thrust::make_permutation_iterator(row_offsets.begin(), node_ids.begin());
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::counting_iterator<size_type>(0),
    col_ids.size(),
    [num_nodes = col_ids.size(),
     ordered_parent_node_ids,
     ordered_row_offsets,
     original_col_ids = original_col_ids.begin(),
     col_ids          = col_ids.begin(),
     row_offsets      = row_offsets.begin(),
     node_categories  = tree.node_categories.begin(),
     d_columns_data   = d_columns_data.begin()] __device__(size_type i) {
      auto parent_node_id = ordered_parent_node_ids[i];
      if (parent_node_id != parent_node_sentinel and node_categories[parent_node_id] == NC_LIST) {
        // unique item
        if (i == 0 or
            (col_ids[i - 1] != col_ids[i] or ordered_parent_node_ids[i - 1] != parent_node_id)) {
          // scatter to list_offset
          d_columns_data[original_col_ids[parent_node_id]]
            .child_offsets[row_offsets[parent_node_id]] = ordered_row_offsets[i];
        }
        // TODO: verify if this code is right. check with more test cases.
        if (i == num_nodes - 1 or
            (col_ids[i] != col_ids[i + 1] or ordered_parent_node_ids[i + 1] != parent_node_id)) {
          // last value of list child_offset is its size.
          d_columns_data[original_col_ids[parent_node_id]]
            .child_offsets[row_offsets[parent_node_id] + 1] = ordered_row_offsets[i] + 1;
        }
      }
    });

  // restore col_ids, TODO is this required?
  // thrust::copy(
  //   rmm::exec_policy(stream), original_col_ids.begin(), original_col_ids.end(), col_ids.begin());

  // 5. scan on offsets.
  for (auto& [id, col_ref] : columns) {
    auto& col = col_ref.get();
    if (col.type == json_col_t::StringColumn) {
      thrust::inclusive_scan(rmm::exec_policy(stream),
                             col.string_offsets.begin(),
                             col.string_offsets.end(),
                             col.string_offsets.begin(),
                             thrust::maximum<json_column::row_offset_t>{});
    } else if (col.type == json_col_t::ListColumn) {
      thrust::inclusive_scan(rmm::exec_policy(stream),
                             col.child_offsets.begin(),
                             col.child_offsets.end(),
                             col.child_offsets.begin(),
                             thrust::maximum<json_column::row_offset_t>{});
    }
  }
}

/**
 * @brief Retrieves the parse_options to be used for type inference and type casting
 *
 * @param options The reader options to influence the relevant type inference and type casting
 * options
 */
cudf::io::parse_options parsing_options(cudf::io::json_reader_options const& options);

std::pair<std::unique_ptr<column>, std::vector<column_name_info>> device_json_column_to_cudf_column(
  device_json_column& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::json_reader_options const& options,
  std::optional<schema_element> schema,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto make_validity =
    [stream](device_json_column& json_col) -> std::pair<rmm::device_buffer, size_type> {
    CUDF_EXPECTS(json_col.validity.size() >= bitmask_allocation_size_bytes(json_col.num_rows),
                 "valid_count is too small");
    auto null_count =
      cudf::detail::null_count(json_col.validity.data(), 0, json_col.num_rows, stream);
    // full null_mask is always required for parse_data
    return {json_col.validity.release(), null_count};
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
      using char_length_pair_t = thrust::pair<const char*, size_type>;
      CUDF_EXPECTS(json_col.string_offsets.size() == json_col.string_lengths.size(),
                   "string offset, string length mismatch");
      rmm::device_uvector<char_length_pair_t> d_string_data(col_size, stream);
      // TODO how about directly storing pair<char*, size_t> in json_column?
      auto offset_length_it =
        thrust::make_zip_iterator(json_col.string_offsets.begin(), json_col.string_lengths.begin());
      // Prepare iterator that returns (string_offset, string_length)-pairs needed by inference
      auto string_ranges_it =
        thrust::make_transform_iterator(offset_length_it, [] __device__(auto ip) {
          return thrust::pair<json_column::row_offset_t, std::size_t>{
            thrust::get<0>(ip), static_cast<std::size_t>(thrust::get<1>(ip))};
        });

      // Prepare iterator that returns (string_ptr, string_length)-pairs needed by type conversion
      auto string_spans_it = thrust::make_transform_iterator(
        offset_length_it, [data = d_input.data()] __device__(auto ip) {
          return thrust::pair<const char*, std::size_t>{
            data + thrust::get<0>(ip), static_cast<std::size_t>(thrust::get<1>(ip))};
        });

      data_type target_type{};

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
          parsing_options(options).json_view(), d_input, string_ranges_it, col_size, stream);
      }
      // Convert strings to the inferred data type
      auto col = experimental::detail::parse_data(string_spans_it,
                                                  col_size,
                                                  target_type,
                                                  make_validity(json_col).first,
                                                  parsing_options(options).view(),
                                                  stream,
                                                  mr);

      // Reset nullable if we do not have nulls
      // This is to match the existing JSON reader's behaviour:
      // - Non-string columns will always be returned as nullable
      // - String columns will be returned as nullable, iff there's at least one null entry
      if (target_type.id() == type_id::STRING and col->null_count() == 0) {
        col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
      }

      // For string columns return ["offsets", "char"] schema
      if (target_type.id() == type_id::STRING) {
        return {std::move(col), {{"offsets"}, {"chars"}}};
      }
      // Non-string leaf-columns (e.g., numeric) do not have child columns in the schema
      return {std::move(col), {}};
    }
    case json_col_t::StructColumn: {
      std::vector<std::unique_ptr<column>> child_columns;
      std::vector<column_name_info> column_names{};
      size_type num_rows{json_col.num_rows};
      // Create children columns
      for (auto const& col_name : json_col.column_order) {
        auto const& col = json_col.child_columns.find(col_name);
        column_names.emplace_back(col->first);
        auto& child_col            = col->second;
        auto [child_column, names] = device_json_column_to_cudf_column(
          child_col, d_input, options, get_child_schema(col_name), stream, mr);
        CUDF_EXPECTS(num_rows == child_column->size(),
                     "All children columns must have the same size");
        child_columns.push_back(std::move(child_column));
        column_names.back().children = names;
      }
      auto [result_bitmask, null_count] = make_validity(json_col);
      return {
        make_structs_column(
          num_rows, std::move(child_columns), null_count, std::move(result_bitmask), stream, mr),
        column_names};
    }
    case json_col_t::ListColumn: {
      size_type num_rows = json_col.child_offsets.size() - 1;
      std::vector<column_name_info> column_names{};
      column_names.emplace_back("offsets");
      column_names.emplace_back(
        json_col.child_columns.empty() ? list_child_name : json_col.child_columns.begin()->first);

      // Note: json_col modified here, reuse the memory
      auto offsets_column = std::make_unique<column>(
        data_type{type_id::INT32}, num_rows + 1, json_col.child_offsets.release());
      // Create children column
      auto [child_column, names] =
        json_col.child_columns.empty()
          ? std::pair<std::unique_ptr<column>,
                      std::vector<column_name_info>>{std::make_unique<column>(), {}}
          : device_json_column_to_cudf_column(
              json_col.child_columns.begin()->second,
              d_input,
              options,
              get_child_schema(json_col.child_columns.begin()->first),
              stream,
              mr);
      column_names.back().children      = names;
      auto [result_bitmask, null_count] = make_validity(json_col);
      return {make_lists_column(num_rows,
                                std::move(offsets_column),
                                std::move(child_column),
                                null_count,
                                std::move(result_bitmask),
                                stream,
                                mr),
              std::move(column_names)};
    }
    default: CUDF_FAIL("Unsupported column type"); break;
  }
}

table_with_metadata device_parse_nested_json(device_span<SymbolT const> d_input,
                                             cudf::io::json_reader_options const& options,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  auto gpu_tree = [&]() {
    // Parse the JSON and get the token stream
    const auto [tokens_gpu, token_indices_gpu] = get_token_stream(d_input, options, stream);
    // gpu tree generation
    return get_tree_representation(tokens_gpu, token_indices_gpu, stream);
  }();  // IILE used to free memory of token data.
#ifdef NJP_DEBUG_PRINT
  auto h_input = cudf::detail::make_host_vector_async(d_input, stream);
  print_tree(h_input, gpu_tree, stream);
#endif

  auto [gpu_col_id, gpu_row_offsets] = records_orient_tree_traversal(d_input, gpu_tree, stream);

  device_json_column root_column(stream, mr);
  root_column.type = json_col_t::ListColumn;
  root_column.child_offsets.resize(2, stream);
  thrust::fill(rmm::exec_policy(stream),
               root_column.child_offsets.begin(),
               root_column.child_offsets.end(),
               0);

  // Get internal JSON column
  make_device_json_column(d_input, gpu_tree, gpu_col_id, gpu_row_offsets, root_column, stream, mr);

  // data_root refers to the root column of the data represented by the given JSON string
  auto& data_root =
    options.is_enabled_lines() ? root_column : root_column.child_columns.begin()->second;

  // Zero row entries
  if (data_root.type == json_col_t::ListColumn && data_root.child_columns.size() == 0) {
    return table_with_metadata{std::make_unique<table>(std::vector<std::unique_ptr<column>>{}),
                               {{}, std::vector<column_name_info>{}}};
  }

  // Verify that we were in fact given a list of structs (or in JSON speech: an array of objects)
  auto constexpr single_child_col_count = 1;
  CUDF_EXPECTS(data_root.type == json_col_t::ListColumn and
                 data_root.child_columns.size() == single_child_col_count and
                 data_root.child_columns.begin()->second.type == json_col_t::StructColumn,
               "Currently the nested JSON parser only supports an array of (nested) objects");

  // Slice off the root list column, which has only a single row that contains all the structs
  auto& root_struct_col = data_root.child_columns.begin()->second;

  // Initialize meta data to be populated while recursing through the tree of columns
  std::vector<std::unique_ptr<column>> out_columns;
  std::vector<column_name_info> out_column_names;

  // Iterate over the struct's child columns and convert to cudf column
  size_type column_index = 0;
  for (auto const& col_name : root_struct_col.column_order) {
    auto& json_col = root_struct_col.child_columns.find(col_name)->second;
    // Insert this columns name into the schema
    out_column_names.emplace_back(col_name);

    std::optional<schema_element> child_schema_element = std::visit(
      cudf::detail::visitor_overload{
        [column_index](const std::vector<data_type>& user_dtypes) -> std::optional<schema_element> {
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
      cudf::detail::visitor_overload{[column_index](const std::vector<data_type>&) {
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

    // Get this JSON column's cudf column and schema info, (modifies json_col)
    auto [cudf_col, col_name_info] = device_json_column_to_cudf_column(
      json_col, d_input, options, child_schema_element, stream, mr);

    out_column_names.back().children = std::move(col_name_info);
    out_columns.emplace_back(std::move(cudf_col));

    column_index++;
  }

  return table_with_metadata{std::make_unique<table>(std::move(out_columns)),
                             {{}, out_column_names}};
}

table_with_metadata device_parse_nested_json(host_span<SymbolT const> input,
                                             cudf::io::json_reader_options const& options,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  // Allocate device memory for the JSON input & copy over to device
  rmm::device_uvector<SymbolT> d_input = cudf::detail::make_device_uvector_async(input, stream);

  return device_parse_nested_json(device_span<SymbolT const>{d_input}, options, stream, mr);
}
}  // namespace detail
}  // namespace cudf::io::json
