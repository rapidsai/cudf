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

#include "cudf/io/new_json_object.hpp"
#include "cudf/utilities/error.hpp"
#include "nested_json.hpp"

#include <cudf/detail/utilities/visitor_overload.hpp>

#include <optional>
#include <string>
#include <vector>

namespace cudf::io::json::detail {

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
      },
      [col_name](
        std::vector<json_path_t> const& user_dtypes) -> std::optional<schema_element> {
        CUDF_FAIL("Unsupported option in this mode, use spark mode");
        return std::optional<schema_element>{};
      }},
    options.get_dtypes());
}

// example schema and its path.
// "a": int             {"a", int}
// "a": [ int ]         {"a", list}, {"element", int}
// "a": { "b": int}     {"a", struct}, {"b", int}
// "a": [ {"b": int }]  {"a", list}, {"element", struct}, {"b", int}
// "a": [ null]         {"a", list}, {"element", str}
// back() is root.
// front() is leaf.
/**
 * @brief Get the path data type of a column by path if present in input schema
 *
 * @param path path of the json column
 * @param root root of input schema element
 * @return data type of the column if present, otherwise std::nullopt
 */
std::optional<data_type> get_path_data_type(
  host_span<std::pair<std::string, cudf::io::json::NodeT> const> path, schema_element const& root)
{
  if (path.empty() || path.size() == 1) {
    return root.type;
  } else {
    if (path.back().second == NC_STRUCT && root.type.id() == type_id::STRUCT) {
      auto const child_name      = path.first(path.size() - 1).back().first;
      auto const child_schema_it = root.child_types.find(child_name);
      return (child_schema_it != std::end(root.child_types))
               ? get_path_data_type(path.first(path.size() - 1), child_schema_it->second)
               : std::optional<data_type>{};
    } else if (path.back().second == NC_LIST && root.type.id() == type_id::LIST) {
      auto const child_schema_it = root.child_types.find(list_child_name);
      return (child_schema_it != std::end(root.child_types))
               ? get_path_data_type(path.first(path.size() - 1), child_schema_it->second)
               : std::optional<data_type>{};
    }
    return std::optional<data_type>{};
  }
}

std::optional<data_type> get_path_data_type(
  host_span<std::pair<std::string, cudf::io::json::NodeT> const> path,
  cudf::io::json_reader_options const& options)
{
  if (path.empty()) return {};
  std::optional<schema_element> col_schema = child_schema_element(path.back().first, options);
  // check if it has value, then do recursive call and return.
  if (col_schema.has_value()) {
    return get_path_data_type(path, col_schema.value());
  } else {
    return {};
  }
}

// idea: write a memoizer using template and lambda?, then call recursively.
std::vector<path_from_tree::path_rep> path_from_tree::get_path(NodeIndexT this_col_id)
{
  std::vector<path_rep> path;
  // stops at root.
  while (this_col_id != parent_node_sentinel) {
    auto type        = column_categories[this_col_id];
    std::string name = "";
    // code same as name_and_parent_index lambda.
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
    }
    // "name": type/schema
    path.emplace_back(name, type);
    this_col_id = parent_col_id;
    if (this_col_id == row_array_parent_col_id) return path;
  }
  return {};
}

}  // namespace cudf::io::json::detail
