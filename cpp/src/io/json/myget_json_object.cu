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

#include "cudf/detail/iterator.cuh"
#include "cudf/io/detail/tokenize_json.hpp"
#include "cudf/io/new_json_object.hpp"
#include "cudf/lists/extract.hpp"
#include "cudf/lists/lists_column_view.hpp"
#include "cudf/strings/combine.hpp"
#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/string_parsing.hpp"
#include "nested_json.hpp"
#include <cudf/scalar/scalar.hpp>
#include "lists/utilities.hpp"
#include <cudf/detail/valid_if.cuh>

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
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cudf::io::json::detail {

std::tuple<tree_meta_t, rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
reduce_to_column_tree(tree_meta_t& tree,
                      device_span<NodeIndexT> original_col_ids,
                      device_span<NodeIndexT> sorted_col_ids,
                      device_span<NodeIndexT> ordered_node_ids,
                      device_span<size_type> row_offsets,
                      bool is_array_of_arrays,
                      NodeIndexT const row_array_parent_col_id,
                      rmm::cuda_stream_view stream);

std::vector<std::string> copy_strings_to_host_sync(
  device_span<SymbolT const> input,
  device_span<SymbolOffsetT const> node_range_begin,
  device_span<SymbolOffsetT const> node_range_end,
  rmm::cuda_stream_view stream);

/**
 * @brief Holds member data pointers of `d_json_column`
 *
 */
struct json_column_data {
  using row_offset_t = json_column::row_offset_t;
  row_offset_t* string_offsets;
  row_offset_t* string_lengths;
  row_offset_t* child_offsets;
  bitmask_type* struct_validity;
  bitmask_type* string_validity;
  bitmask_type* list_validity;
  json_col_t type;
};
template<typename Iter>
struct IteratorRange {
Iter _beg;
Iter _end;
IteratorRange(Iter beg, Iter end) : _beg(std::move(beg)), _end(std::move(end)) { }
Iter begin() { return _beg; }
Iter end() { return _end; }
};

void make_device_json_column2(device_span<SymbolT const> input,
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
    }
    col.num_rows = max_row_offsets[i] + 1;
    if (column_category == NC_VAL || column_category == NC_STR) {
      col.string_offsets.resize(max_row_offsets[i] + 1, stream);
      col.string_lengths.resize(max_row_offsets[i] + 1, stream);
      init_to_zero(col.string_offsets);
      init_to_zero(col.string_lengths);
      col.string_validity =
        cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    } else if (column_category == NC_LIST) {
      col.child_offsets.resize(max_row_offsets[i] + 2, stream);
      init_to_zero(col.child_offsets);
      col.list_validity =
        cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    } else if (column_category == NC_STRUCT) {
    col.struct_validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    }
    if(col.type == json_col_t::Unknown)
      col.type = to_json_col_type(column_category);
    else
      col.type = json_col_t::MixedColumn;
  };

  // auto reinitialize_as_string = [&](auto i, auto& col) {
  //   col.string_offsets.resize(max_row_offsets[i] + 1, stream);
  //   col.string_lengths.resize(max_row_offsets[i] + 1, stream);
  //   init_to_zero(col.string_offsets);
  //   init_to_zero(col.string_lengths);
  //   col.num_rows = max_row_offsets[i] + 1;
  //   col.string_validity =
  //     cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
  //   col.type = json_col_t::StringColumn;
  //   // destroy references of all child columns after this step, by calling remove_child_columns
  // };

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
  // find column_ids which are values, but should be ignored in validity
  auto ignore_vals = cudf::detail::make_host_vector<uint8_t>(num_columns, stream);
  std::vector<uint8_t> is_pruned(num_columns, 0);

  // get_json_object
  // if (std::holds_alternative<std::vector<json_path_t>>(options.get_dtypes()));
    // Build adjacency list.
    std::map<NodeIndexT, std::vector<NodeIndexT>> adj;
    for (auto const this_col_id : unique_col_ids) {
      auto parent_col_id = column_parent_ids[this_col_id];
      adj[parent_col_id].push_back(this_col_id);
    }
    std::fill(ignore_vals.begin(), ignore_vals.end(), 1);
    std::fill(is_pruned.begin(), is_pruned.end(), 1);
    auto& json_paths = std::get<std::vector<json_path_t>>(options.get_dtypes());
    std::vector<NodeIndexT> is_path_found(json_paths.size(), -1);
    std::vector<NodeIndexT> is_string(num_columns, 0);
    auto range = IteratorRange(thrust::make_zip_iterator(thrust::make_counting_iterator(0ul), json_paths.begin()),
    thrust::make_zip_iterator(thrust::make_counting_iterator(json_paths.size()), json_paths.end()));
    // does is_array_of_arrays matter here?
    // struct, field, list, value/string,
    // Mark necessary columns, & mark forced string types.
    // Prune columns that are not required to be parsed.
    for (auto [path_index, path] : range) {
      auto parent = parent_node_sentinel;
      std::deque<NodeIndexT> tree_path;
      bool is_path_found = false;
      for (size_t pi=0ul; pi<path.size(); ++pi) {
        auto& [path_type, name, index] = path[pi];
        if(path_type == path_instruction_type::NAMED) {
          auto col_name = name;
          // first find struct children, then find name in it.
          auto it = std::find_if(adj[parent].begin(), adj[parent].end(),
            [column_categories](auto col_id) { return column_categories[col_id] == NC_STRUCT; });
          if(it == adj[parent].end()) {
            break;
          }
          auto struct_col_id = *it;
          auto it2 = std::find_if(adj[struct_col_id].begin(), adj[struct_col_id].end(),
            [col_name, column_names](auto col_id) { return column_names[col_id] == col_name; });
          if (it2 == adj[struct_col_id].end()) {
            break;
          } else {
            auto field_column_id = *it2; // this is field
            // CUDF_EXPECTS(adj[field_column_id].size()==1, "Field column should have only 1 children!");

            // FIXME: could be mixed type!?
            // If it has multiple children, choose the right child if next one is list, or struct, or string.
            // TODO: if this is last of the path, all of the child should be read as single string column
            // this_column_id = adj[this_column_id].at(0); // struct child column
            parent = field_column_id;
            tree_path.push_back(struct_col_id);
            tree_path.push_back(field_column_id);
            if (pi == path.size()-1) {
              std::cout<<"found path: "<<col_name<<std::endl;
              is_path_found = true;
              // if only 1, no problem. if mixed type, we need all. how?
              for(auto ch_i : adj[field_column_id]) {
                tree_path.push_back(ch_i); // TODO? only string column how? remap types?(might mess up others)
                is_string[ch_i]=1;
                // everything is string!
              }
            }
          }
        } else if(path_type == path_instruction_type::WILDCARD) {
            auto it = std::find_if(adj[parent].begin(), adj[parent].end(),
              [column_categories](auto col_id) { return column_categories[col_id] == NC_LIST; });
            if(it == adj[parent].end()) {
              break;
            }
            auto list_col_id = *it;
            // elements // find the child node, that is list.
            // CUDF_EXPECTS(adj[parent].size()==1, "List column should have only 1 children!");
            // FIXME: could be mixed type!?
            parent = list_col_id;
            tree_path.push_back(list_col_id);
            if (pi == path.size()-1) {
              is_path_found = true;
              is_string[list_col_id]=1;
            }
        } else if(path_type == path_instruction_type::INDEX) {
          auto it = std::find_if(adj[parent].begin(), adj[parent].end(),
              [column_categories](auto col_id) { return column_categories[col_id] == NC_LIST; });
            if(it == adj[parent].end()) {
              break;
            }
            auto list_col_id = *it;
            parent = list_col_id;
            tree_path.push_back(list_col_id);
            if (pi == path.size()-1) {
              is_path_found = true;
              for(auto ch_i : adj[list_col_id]) {
                tree_path.push_back(ch_i); // TODO? only string column how? remap types?(might mess up others)
                is_string[ch_i]=1;
                // everything is string!
              }
            }
        } else {
          CUDF_FAIL("Unexpected path type in json path dtypes");
        }
      }
      if(is_path_found) {
        for(auto col : tree_path)
          is_pruned[col] = 0;
      }
    }
    // is_string should have all string+struct+list boundaries too.
  columns.try_emplace(parent_node_sentinel, std::ref(root));

  // // // go through adjacency list and build the device_json_column
  // std::deque<NodeIndexT> bfs_q;
  // bfs_q.push_back(parent_node_sentinel); //parent node -> struct or list.
  // device_json_column* base = &root; //type is struct/list/string. // struct=1, list=2, string=4;
  // while(not bfs_q.empty()) {
  //   auto this_col_id = bfs_q.front(); bfs_q.pop_front();
  //   auto it = columns.find(this_col_id);
  //   if(it == columns.end()) CUDF_FAIL("Can't find column id");
  //   auto& parent = it->second;
  //   for(auto child_col_id : adj[this_col_id]) {
  //     if(is_pruned[child_col_id]) continue;
  //     if(column_categories[child_col_id] == NC_ERR) continue;
  //     if(column_categories[child_col_id] == NC_FN) { continue;
  //       // construct the column.
  //       // handle mixed types here.
  //       for(auto fld_col_id : adj[child_col_id]) {
  //         auto column_name = column_names[fld_col_id];
  //         parent.child_columns
  //       }
  //     }
  //     if(column_categories[child_col_id] == NC_STRUCT) {
  //       initialize_json_columns(child_col_id, *base, NC_STRUCT);
  //       for(auto fld_col_id : adj[child_col_id]) {
  //         if(is_pruned[fld_col_id]) continue;
  //         auto column_name = column_names[fld_col_id];
  //         columns.try_emplace(fld_col_id, std::ref(base->child_columns[column_name]));
  //         base->column_order.push_back(column_name);
  //       }
  //     } else if(column_categories[child_col_id] == NC_LIST) {
  //     } else if(column_categories[child_col_id] == NC_STR) {
  //     }
  //   }
  // }

  auto name_and_parent_index = [&is_array_of_arrays,
                                &row_array_parent_col_id,
                                &column_parent_ids,
                                &column_categories,
                                &column_names](auto this_col_id) {
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
    return std::pair{name, parent_col_id};
  };


  // Build the column tree, also, handles mixed types.
  for (auto const this_col_id : unique_col_ids) {
    if (column_categories[this_col_id] == NC_ERR || column_categories[this_col_id] == NC_FN) {
      continue;
    }
    // Struct, List, String, Value
    auto [name, parent_col_id] = name_and_parent_index(this_col_id);

    if (is_pruned[this_col_id]) continue;
    // if (is_string[this_col_id]) col.forced_as_string_column = true;

    // If the child is already found,
    // replace if this column is a nested column and the existing was a value column
    // ignore this column if this column is a value column and the existing was a nested column
    auto it = columns.find(parent_col_id);
    CUDF_EXPECTS(it != columns.end(), "Parent column not found");
    auto& parent_col = it->second.get();
    // bool replaced    = false;
    // if (mapped_columns.count({parent_col_id, name}) > 0) {
    //   auto const old_col_id = mapped_columns[{parent_col_id, name}];
    //   // If mixed type as string is enabled, make both of them strings and merge them.
    //   // All child columns will be ignored when parsing.
    //   if (is_enabled_mixed_types_as_string) {
    //     auto is_mixed_type = false;
    //     if (is_mixed_type) {
    //       is_mixed_type_column[this_col_id] = 1;
    //       is_mixed_type_column[old_col_id]  = 1;
    //       // if old col type (not cat) is list or struct, replace with string.
    //       auto& col = columns.at(old_col_id).get();
    //       if (col.type == json_col_t::ListColumn or col.type == json_col_t::StructColumn) {
    //         reinitialize_as_string(old_col_id, col);
    //         remove_child_columns(old_col_id, col);
    //         // all its children (which are already inserted) are ignored later.
    //       }
    //       col.forced_as_string_column = true;
    //       columns.try_emplace(this_col_id, columns.at(old_col_id));
    //       continue;
    //     }
    //   }

    //   if (column_categories[this_col_id] == NC_VAL || column_categories[this_col_id] == NC_STR) {
    //     ignore_vals[this_col_id] = 1;
    //     continue;
    //   }
    //   if (column_categories[old_col_id] == NC_VAL || column_categories[old_col_id] == NC_STR) {
    //     // remap
    //     ignore_vals[old_col_id] = 1;
    //     mapped_columns.erase({parent_col_id, name});
    //     columns.erase(old_col_id);
    //     parent_col.child_columns.erase(name);
    //     replaced = true;  // to skip duplicate name in column_order
    //   } else {
    //     // If this is a nested column but we're trying to insert either (a) a list node into a
    //     // struct column or (b) a struct node into a list column, we fail
    //     CUDF_EXPECTS(not((column_categories[old_col_id] == NC_LIST and
    //                       column_categories[this_col_id] == NC_STRUCT) or
    //                      (column_categories[old_col_id] == NC_STRUCT and
    //                       column_categories[this_col_id] == NC_LIST)),
    //                  "A mix of lists and structs within the same column is not supported");
    //   }
    // }

    // auto this_column_category = column_categories[this_col_id];
    // if (is_enabled_mixed_types_as_string) {
    //   // get path of this column, check if it is a struct/list forced as string, and enforce it
    //   auto const nt                             = tree_path.get_path(this_col_id);
    //   std::optional<data_type> const user_dtype = get_path_data_type(nt, options);
    //   if ((column_categories[this_col_id] == NC_STRUCT or
    //        column_categories[this_col_id] == NC_LIST) and
    //       user_dtype.has_value() and user_dtype.value().id() == type_id::STRING) {
    //     is_mixed_type_column[this_col_id] = 1;
    //     this_column_category              = NC_STR;
    //   }
    // }

    // LEAF = anything
    // PARENT = LIST, STRUCT(name)
    // if parent is list, add this child to child_columns. (get if already present, and update it)
    // if parent is struct, field multiple times can occur. get that field column, from parent and update.
    std::cout<<parent_col_id<<" "<<name<<std::endl;
    if (parent_col_id == parent_node_sentinel or column_categories[parent_col_id] == NC_LIST) {
      if(parent_col.list_child_columns.empty()) {
        parent_col.list_child_columns.emplace_back(device_json_column(stream, mr));
      }
      // name is "element";
      // device_json_column col(stream, mr);
      auto& this_col = parent_col.list_child_columns[0];
      initialize_json_columns(this_col_id, this_col, column_categories[this_col_id]); // as string, struct, list
      if(is_string[this_col_id])
        initialize_json_columns(this_col_id, this_col, NC_STR);
      columns.try_emplace(this_col_id, std::ref(this_col));
      ignore_vals[this_col_id] = 0;
    } else if (column_categories[parent_col_id] == NC_STRUCT) {
      std::cout<< "inserting "<< name << std::endl;
      if(parent_col.child_columns.count(name) == 0) {
        parent_col.child_columns.emplace(name, device_json_column(stream, mr));
        parent_col.column_order.push_back(name);
      }
      auto& this_col = parent_col.child_columns.at(name);
      initialize_json_columns(this_col_id, this_col, column_categories[this_col_id]);
      if(is_string[this_col_id])
        initialize_json_columns(this_col_id, this_col, NC_STR);
      columns.try_emplace(this_col_id, std::ref(this_col));
      ignore_vals[this_col_id] = 0;
      // 2 features:
      // if mixed type is not enabled: if current is value/string, ignore. (these should have been fixed in adj list traversal itself.)
      // if mixed type is enabled, if current is value/string and all nulls, ignore, or mix it (force it as string and ignore all children)
      //      
    } else {
      CUDF_FAIL("Not expecting non-nested parent");
    }
  }

  // if (is_enabled_mixed_types_as_string) {
  //   // ignore all children of mixed type columns
  //   for (auto const this_col_id : unique_col_ids) {
  //     auto parent_col_id = column_parent_ids[this_col_id];
  //     if (parent_col_id != parent_node_sentinel and is_mixed_type_column[parent_col_id] == 1) {
  //       is_mixed_type_column[this_col_id] = 1;
  //       ignore_vals[this_col_id]          = 1;
  //       columns.erase(this_col_id);
  //     }
  //     // Convert only mixed type columns as string (so to copy), but not its children
  //     if (parent_col_id != parent_node_sentinel and is_mixed_type_column[parent_col_id] == 0 and
  //         is_mixed_type_column[this_col_id] == 1)
  //       column_categories[this_col_id] = NC_STR;
  //   }
  //   cudaMemcpyAsync(d_column_tree.node_categories.begin(),
  //                   column_categories.data(),
  //                   column_categories.size() * sizeof(column_categories[0]),
  //                   cudaMemcpyDefault,
  //                   stream.value());
  // }
  // // }

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
                                            static_cast<bitmask_type*>(col.struct_validity.data()),
                                            static_cast<bitmask_type*>(col.string_validity.data()),
                                            static_cast<bitmask_type*>(col.list_validity.data()),
                                            col.type};
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
        case NC_STRUCT: 
          set_bit(d_columns_data[col_ids[i]].struct_validity, row_offsets[i]);
        case NC_LIST:
          if(node_category==NC_LIST)
            set_bit(d_columns_data[col_ids[i]].list_validity, row_offsets[i]);
          if(d_columns_data[col_ids[i]].type != json_col_t::MixedColumn) break;
        case NC_STR: [[fallthrough]];
        case NC_VAL:
          if (d_ignore_vals[col_ids[i]]) break;
          set_bit(d_columns_data[col_ids[i]].string_validity, row_offsets[i]);
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
std::unique_ptr<column> join_list_of_strings(lists_column_view const& lists_strings,
                                             string_view const list_prefix,
                                             string_view const list_suffix,
                                             string_view const element_separator,
                                             string_view const element_narep,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  /*
  create string_views of the list elements, and the list separators and list prefix/suffix.
  then concatenates them all together.
  gather offset of first string_view of each row as offsets for output string column.
  Algorithm:
    calculate #strviews per list using null mask, and list_offsets.
    scan #strviews to get strviews_offset
    create label segments.
    sublist_index = index - offsets[label]
    strviews_offset[label] + sublist_index = string_view index +1, +2
    use above 2 to scatter element, element_seperator
    scatter list_prefix, list_suffix to the right place using list_offsets
    make_strings_column() and gather offsets, based on strviews_offset.
  */
  auto const offsets          = lists_strings.offsets();
  auto const strings_children = lists_strings.get_sliced_child(stream);
  auto const num_lists        = lists_strings.size();
  auto const num_strings      = strings_children.size();
  auto const num_offsets      = offsets.size();

  rmm::device_uvector<size_type> d_strview_offsets(num_offsets, stream);
  auto num_strings_per_list = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [offsets = offsets.begin<size_type>(), num_offsets] __device__(size_type idx) {
        if (idx + 1 >= num_offsets) return 0;
        auto const length = offsets[idx + 1] - offsets[idx];
        return length == 0 ? 2 : (2 + length + length - 1);
      }));

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         num_strings_per_list,
                         num_strings_per_list + num_offsets,
                         d_strview_offsets.begin());
  auto const total_strings = d_strview_offsets.back_element(stream);

  rmm::device_uvector<string_view> d_strviews(total_strings, stream);
  // scatter null_list and list_prefix, list_suffix
  auto col_device_view = cudf::column_device_view::create(lists_strings.parent(), stream);
  auto d_strings_children = cudf::column_device_view::create(strings_children, stream);

  // // single list element if it is null.
    auto [null_mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_lists),
    [col                = *col_device_view,
    d_strings_children  = *d_strings_children,
     offsets = offsets.begin<size_type>()] __device__(size_type idx) {
      auto const length = offsets[idx + 1] - offsets[idx];
      if (length == 0) return false;
      else if (length == 1) return d_strings_children.is_valid(offsets[idx]);
      return true;
    },
    stream,
    mr);
  std::vector<bitmask_type const*> masks{static_cast<bitmask_type const*>(null_mask.data())};
  std::vector<size_type> begin_bits{0};
    if(lists_strings.parent().null_mask()) {
    masks.push_back(lists_strings.parent().null_mask());
    begin_bits.push_back(lists_strings.parent().offset());
  }
  auto [new_nullmask, new_nullcount] = cudf::detail::bitmask_and(masks, begin_bits, num_lists, stream, mr);

  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(num_lists),
                   [col = *col_device_view,
                    list_prefix,
                    list_suffix,
                    d_strview_offsets = d_strview_offsets.begin(),
                    d_strviews        = d_strviews.begin()] __device__(auto idx) {
                      // Find size of list element
                      auto const list_offsets = col.child(lists_column_view::offsets_column_index).template data<size_type>() + col.offset();
                      auto const list_size = list_offsets[idx + 1] - list_offsets[idx]; //  or list_size <= 1
                     if (col.is_null(idx)) {
                       d_strviews[d_strview_offsets[idx]]     = string_view{};
                       d_strviews[d_strview_offsets[idx] + 1] = string_view{};
                     } else {
                       // [ ]
                       d_strviews[d_strview_offsets[idx]]         = (list_size == 1) ? string_view{}: list_prefix;
                       d_strviews[d_strview_offsets[idx + 1] - 1] = (list_size == 1) ? string_view{}: list_suffix;
                     }
                   });

  // scatter string and separator
  auto labels = cudf::lists::detail::generate_labels(
    lists_strings, num_strings, stream, rmm::mr::get_current_device_resource());
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(num_strings),
                   [col                = *col_device_view,
                    d_strview_offsets  = d_strview_offsets.begin(),
                    d_strviews         = d_strviews.begin(),
                    labels             = labels->view().begin<size_type>(),
                    list_offsets       = offsets.begin<size_type>(),
                    d_strings_children = *d_strings_children,
                    element_separator,
                    element_narep] __device__(auto idx) {
                     auto const label         = labels[idx];
                     auto const sublist_index = idx - list_offsets[label];
                     auto const strview_index = d_strview_offsets[label] + sublist_index * 2 + 1;
                     // value or na_rep
                     auto const strview = d_strings_children.element<cudf::string_view>(idx);
                     d_strviews[strview_index] =
                       d_strings_children.is_null(idx) ? string_view{} : strview;
                     // separator
                     if (sublist_index != 0) { d_strviews[strview_index - 1] = element_separator; }
                   });

  auto joined_col = make_strings_column(d_strviews, string_view{nullptr, 0}, stream, mr);

  // gather from offset and create a new string column
  auto old_offsets = strings_column_view(joined_col->view()).offsets();
  rmm::device_uvector<size_type> row_string_offsets(num_offsets, stream, mr);
  thrust::gather(rmm::exec_policy(stream),
                 d_strview_offsets.begin(),
                 d_strview_offsets.end(),
                 old_offsets.begin<size_type>(),
                 row_string_offsets.begin());
  auto chars_data = joined_col->release().data;
  return make_strings_column(
    num_lists,
    std::make_unique<cudf::column>(std::move(row_string_offsets), rmm::device_buffer{}, 0),
    std::move(chars_data.release()[0]),
    new_nullcount, std::move(new_nullmask));
    // lists_strings.null_count(),
    // cudf::detail::copy_bitmask(lists_strings.parent(), stream, mr));
}


// declaration
std::pair<std::unique_ptr<column>, std::vector<column_name_info>> extract_cudf_column_from_device_json_column(
  device_json_column& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::parse_options const& options,
  host_span<json_location_t> schema,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr, bool is_wild=false);

table_with_metadata extract_result_columns(
  device_json_column& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) {
    std::cout<<"extracting result columns"<<std::endl;
    //  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const& json_paths,
  auto& json_paths = std::get<std::vector<json_path_t>>(options.get_dtypes());
  std::vector<std::unique_ptr<column>> out_columns;
  std::vector<column_name_info> out_column_names;
  out_columns.reserve(json_paths.size());
  auto parse_opt = parsing_options(options, stream);
  size_t i =0;
  for (auto path : json_paths) {
    auto col_name = std::to_string(i++);
    out_column_names.emplace_back(col_name);
    auto col = extract_cudf_column_from_device_json_column(json_col, d_input, parse_opt, path, stream, mr).first;
    out_columns.push_back(std::move(col));
  }
  return table_with_metadata{std::make_unique<table>(std::move(out_columns)), {out_column_names}};
}

// TODO pass mr.
template<json_col_t col_type>
auto make_validity(device_json_column& json_col, rmm::cuda_stream_view stream) {
  auto& validity = [&]() -> auto& {
    if constexpr(col_type == json_col_t::StringColumn)
      return json_col.string_validity;
    else if constexpr(col_type == json_col_t::ListColumn)
      return json_col.list_validity;
    else if constexpr(col_type == json_col_t::StructColumn)
      return json_col.struct_validity;
    return json_col.string_validity;
  }();
  CUDF_EXPECTS(validity.size() >= bitmask_allocation_size_bytes(json_col.num_rows),
               "valid_count is too small");
  auto null_count = cudf::detail::null_count(
    static_cast<bitmask_type*>(validity.data()), 0, json_col.num_rows, stream);
  // full null_mask is always required for parse_data (TODO: std::move(validity) ?)
  return std::pair<rmm::device_buffer, size_type>{rmm::device_buffer{validity, stream}, null_count};
}

std::unique_ptr<column> extract_string_column(
  device_json_column& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::parse_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  bool is_wild) {
    std::cout<<"extracting string column"<<json_col.num_rows<<std::endl;
    // CUDF_EXPECTS(json_col.type==json_col_t::StringColumn or json_col.type==json_col_t::MixedColumn, "Not a string or mixed column");
    CUDF_EXPECTS(json_col.string_offsets.size() > 0 and json_col.string_lengths.size() > 0, "string column is empty");

    // move string_offsets to GPU and transform to string column
    auto const col_size      = json_col.string_offsets.size();
    using char_length_pair_t = thrust::pair<char const*, size_type>;
    CUDF_EXPECTS(json_col.string_offsets.size() == json_col.string_lengths.size(),
                  "string offset, string length mismatch");
    rmm::device_uvector<char_length_pair_t> d_string_data(col_size, stream);
    auto [result_bitmask, null_count] = make_validity<json_col_t::StringColumn>(json_col, stream);
    // TODO how about directly storing pair<char*, size_t> in json_column?
    auto offset_length_it =
      thrust::make_zip_iterator(json_col.string_offsets.begin(), json_col.string_lengths.begin());
    data_type target_type = data_type{type_id::STRING};
    // Convert strings to the inferred data type
    auto col = is_wild == false? parse_data(d_input.data(),
                          offset_length_it,
                          col_size,
                          target_type,
                          std::move(result_bitmask),
                          null_count,
                          options.view(),
                          stream,
                          mr):
    // bool skip_double_quotes = false;
    // auto col = 
    parse_raw_strings(d_input.data(),
                                        json_col.string_offsets,
                                        json_col.string_lengths,
                                        std::move(result_bitmask),
                                        null_count,
                                        false,
                                        stream,
                                        mr);
    // Reset nullable if we do not have nulls
    // This is to match the existing JSON reader's behaviour:
    // - Non-string columns will always be returned as nullable
    // - String columns will be returned as nullable, iff there's at least one null entry
    if (col->null_count() == 0) { col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0); }

    return col;
    // // For string columns return ["offsets", "char"] schema
    // if (target_type.id() == type_id::STRING) {
    //   return {std::move(col), std::vector<column_name_info>{{"offsets"}, {"chars"}}};
    // }
    // // Non-string leaf-columns (e.g., numeric) do not have child columns in the schema
    // return {std::move(col), std::vector<column_name_info>{}};
}

std::unique_ptr<column> extract_list_column(
  device_json_column& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::parse_options const& options,
  host_span<json_location_t> schema,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr, bool is_wild=false) {
    CUDF_EXPECTS(json_col.type==json_col_t::ListColumn or json_col.type==json_col_t::MixedColumn, "Not a list or mixed column");
    CUDF_EXPECTS(json_col.child_offsets.size() > 0, "list column is empty");
    CUDF_EXPECTS(json_col.list_child_columns.size() > 0, "list column has no children");

    size_type num_rows = json_col.child_offsets.size() - 1;
    std::vector<column_name_info> column_names{};
    column_names.emplace_back("offsets");
    column_names.emplace_back(list_child_name);

    // Note: json_col modified here, reuse the memory
    //column(rmm::device_uvector<T>&& other, rmm::device_buffer&& null_mask, size_type null_count)
    auto offsets_column = std::make_unique<column>(rmm::device_uvector<int32_t>(json_col.child_offsets, stream, mr),
                                                    rmm::device_buffer{},
                                                    0);
    // auto offsets_column = std::make_unique<column>(data_type{type_id::INT32},
    //                                                 num_rows + 1,
    //                                                 json_col.child_offsets.release(),
    //                                                 rmm::device_buffer{},
    //                                                 0);
    // Create children column
    auto child_schema_element = schema.subspan(1, schema.size() - 1);
    auto [child_column, names] = extract_cudf_column_from_device_json_column(json_col.list_child_columns[0],
                                            d_input,
                                            options,
                                            child_schema_element,
                                            stream,
                                            mr, is_wild);
    column_names.back().children      = names;
    auto [result_bitmask, null_count] = make_validity<json_col_t::ListColumn>(json_col, stream);
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
    return ret_col;
    // return {std::move(ret_col), std::move(column_names)};
  }

std::pair<std::unique_ptr<column>, std::vector<column_name_info>> extract_cudf_column_from_device_json_column(
  device_json_column& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::parse_options const& options,
  host_span<json_location_t> schema,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr, bool is_wild)
{
  // may need separate validity for struct, list, string.
  CUDF_FUNC_RANGE();
  if(schema.empty()) CUDF_FAIL("Empty json path schema!");

  auto [path_type, name, index] = schema.front();
  if (path_type == path_instruction_type::WILDCARD) {
    std::cout<<"WILDCARD"<<std::endl;
    // Join list elements idea.
    auto list_col = extract_list_column(json_col, d_input, options, schema, stream, mr, true);
    // auto join_list = join_list_elements(list_col->view(),
    //                                        string_scalar{",", true, stream},
    //                                        string_scalar{"", false, stream},
    //                                        cudf::strings::separator_on_nulls::NO,
    //                                        cudf::strings::output_if_empty_list::EMPTY_STRING,
    //                                        stream,
    //                                       mr);
    cudf::string_scalar list_row_begin_wrap{"[", true, stream, mr};
    cudf::string_scalar list_row_end_wrap{"]", true, stream, mr};
    cudf::string_scalar list_value_separator{",", true, stream, mr};
    cudf::string_scalar narep{"null", true, stream, mr};
    auto join_list = join_list_of_strings(list_col->view(),
                                list_row_begin_wrap.value(stream),
                                list_row_end_wrap.value(stream),
                                list_value_separator.value(stream),
                                narep.value(stream),
                                stream,
                                mr);
    return {std::move(join_list), {std::string("[*]")}};
  } else if (path_type == path_instruction_type::INDEX) {
    std::cout<<"INDEX:"<<index<<std::endl;
    // Extract list element
    auto list_col = extract_list_column(json_col, d_input, options, schema, stream, mr);
    return {cudf::lists::extract_list_element(list_col->view(), index, stream, mr), {"["+std::to_string(index)+"]"}};
  } else if (path_type == path_instruction_type::NAMED) {
      std::cout<<"NAMED:"<<name<<".\n";
    // Extract struct element
    CUDF_EXPECTS(json_col.type==json_col_t::StructColumn or json_col.type==json_col_t::MixedColumn, "Not a struct or mixed column");
    auto expected_col = json_col.child_columns.find(name);
    if (expected_col == json_col.child_columns.end()) {
      std::cout<< "can't find struct column "<< name << std::endl;
      return {make_column_from_scalar(cudf::string_scalar("", false, stream, mr), json_col.num_rows, stream, mr), {name}};
    } else {
      // Not sure what to do with this bitmask, (propagate nulls?)
      // auto [result_bitmask, null_count] = make_validity<json_col_t::StructColumn>(json_col, stream);
      if (schema.size()==1)
        return {extract_string_column(expected_col->second, d_input, options, stream, mr, is_wild), {name}};
      return extract_cudf_column_from_device_json_column(expected_col->second, d_input, options, schema.subspan(1, schema.size()-1), stream, mr, is_wild);
    }
  } else {
    CUDF_FAIL("Unexpected path type in json path schema");
  }
}

}  // namespace cudf::io::json::detail
