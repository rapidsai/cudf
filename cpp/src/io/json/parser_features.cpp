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

#include "nested_json.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <optional>
#include <string>
#include <vector>

namespace cudf::io {
namespace {
bool validate_column_order(schema_element const& types)
{
  // For struct types, check if column_order size matches child_types size and all elements in
  // column_order are in child_types, in child_types, call this function recursively.
  // For list types, check if child_types size is 1 and call this function recursively.
  if (types.type.id() == type_id::STRUCT) {
    if (types.column_order.has_value()) {
      if (types.column_order.value().size() != types.child_types.size()) { return false; }
      for (auto const& column_name : types.column_order.value()) {
        auto it = types.child_types.find(column_name);
        if (it == types.child_types.end()) { return false; }
        if (it->second.type.id() == type_id::STRUCT or it->second.type.id() == type_id::LIST) {
          if (!validate_column_order(it->second)) { return false; }
        }
      }
    }
  } else if (types.type.id() == type_id::LIST) {
    if (types.child_types.size() != 1) { return false; }
    auto it = types.child_types.begin();
    if (it->second.type.id() == type_id::STRUCT or it->second.type.id() == type_id::LIST) {
      if (!validate_column_order(it->second)) { return false; }
    }
  }
  return true;
}
}  // namespace

void json_reader_options::set_dtypes(schema_element types)
{
  CUDF_EXPECTS(
    validate_column_order(types), "Column order does not match child types", std::invalid_argument);
  _dtypes = std::move(types);
}
}  // namespace cudf::io

namespace cudf::io::json::detail {

/// Created an empty column of the specified schema
struct empty_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, CUDF_ENABLE_IF(!cudf::is_nested<T>())>
  std::unique_ptr<column> operator()(schema_element const& schema) const
  {
    return make_empty_column(schema.type);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema) const
  {
    CUDF_EXPECTS(schema.child_types.size() == 1, "List column should have only one child");
    auto const& child_name        = schema.child_types.begin()->first;
    std::unique_ptr<column> child = cudf::type_dispatcher(
      schema.child_types.at(child_name).type, *this, schema.child_types.at(child_name));
    auto offsets = make_empty_column(data_type(type_to_id<size_type>()));
    return make_lists_column(0, std::move(offsets), std::move(child), 0, {}, stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema) const
  {
    std::vector<std::unique_ptr<column>> child_columns;
    for (auto const& child_name : schema.column_order.value_or(std::vector<std::string>{})) {
      child_columns.push_back(cudf::type_dispatcher(
        schema.child_types.at(child_name).type, *this, schema.child_types.at(child_name)));
    }
    return make_structs_column(0, std::move(child_columns), 0, {}, stream, mr);
  }
};

/// Created all null column of the specified schema
struct allnull_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

 private:
  [[nodiscard]] auto make_zeroed_offsets(size_type size) const
  {
    auto offsets_buff =
      cudf::detail::make_zeroed_device_uvector_async<size_type>(size + 1, stream, mr);
    return std::make_unique<column>(std::move(offsets_buff), rmm::device_buffer{}, 0);
  }

 public:
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    return make_fixed_width_column(schema.type, size, mask_state::ALL_NULL, stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_dictionary<T>())>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    CUDF_EXPECTS(schema.child_types.size() == 1, "Dictionary column should have only one child");
    auto const& child_name        = schema.child_types.begin()->first;
    std::unique_ptr<column> child = cudf::type_dispatcher(schema.child_types.at(child_name).type,
                                                          empty_column_functor{stream, mr},
                                                          schema.child_types.at(child_name));
    return make_fixed_width_column(schema.type, size, mask_state::ALL_NULL, stream, mr);
    auto indices   = make_zeroed_offsets(size - 1);
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    return make_dictionary_column(
      std::move(child), std::move(indices), std::move(null_mask), size, stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    auto offsets   = make_zeroed_offsets(size);
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    return make_strings_column(
      size, std::move(offsets), rmm::device_buffer{}, size, std::move(null_mask));
  }
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    CUDF_EXPECTS(schema.child_types.size() == 1, "List column should have only one child");
    auto const& child_name        = schema.child_types.begin()->first;
    std::unique_ptr<column> child = cudf::type_dispatcher(schema.child_types.at(child_name).type,
                                                          empty_column_functor{stream, mr},
                                                          schema.child_types.at(child_name));
    auto offsets                  = make_zeroed_offsets(size);
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    return make_lists_column(
      size, std::move(offsets), std::move(child), size, std::move(null_mask), stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    std::vector<std::unique_ptr<column>> child_columns;
    for (auto const& child_name : schema.column_order.value_or(std::vector<std::string>{})) {
      child_columns.push_back(cudf::type_dispatcher(
        schema.child_types.at(child_name).type, *this, schema.child_types.at(child_name), size));
    }
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    return make_structs_column(
      size, std::move(child_columns), size, std::move(null_mask), stream, mr);
  }
};

std::unique_ptr<column> make_all_nulls_column(schema_element const& schema,
                                              size_type num_rows,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(schema.type, allnull_column_functor{stream, mr}, schema, num_rows);
}

column_name_info make_column_name_info(schema_element const& schema, std::string const& col_name)
{
  column_name_info info;
  info.name = col_name;
  switch (schema.type.id()) {
    case type_id::STRUCT:
      for (auto const& child_name : schema.column_order.value_or(std::vector<std::string>{})) {
        info.children.push_back(
          make_column_name_info(schema.child_types.at(child_name), child_name));
      }
      break;
    case type_id::LIST:
      info.children.emplace_back("offsets");
      for (auto const& [child_name, child_schema] : schema.child_types) {
        info.children.push_back(make_column_name_info(child_schema, child_name));
      }
      break;
    case type_id::DICTIONARY32:
      info.children.emplace_back("indices");
      for (auto const& [child_name, child_schema] : schema.child_types) {
        info.children.push_back(make_column_name_info(child_schema, child_name));
      }
      break;
    case type_id::STRING: info.children.emplace_back("offsets"); break;
    default: break;
  }
  return info;
}

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
      [col_name](schema_element const& user_dtypes) -> std::optional<schema_element> {
        return (user_dtypes.child_types.find(col_name) != std::end(user_dtypes.child_types))
                 ? user_dtypes.child_types.find(col_name)->second
                 : std::optional<schema_element>{};
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
