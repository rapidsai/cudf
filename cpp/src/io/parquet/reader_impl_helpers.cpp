/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "reader_impl_helpers.hpp"

#include <io/utilities/row_selection.hpp>

#include <numeric>
#include <regex>

namespace cudf::io::detail::parquet {

namespace {

ConvertedType logical_type_to_converted_type(LogicalType const& logical)
{
  if (logical.isset.STRING) {
    return parquet::UTF8;
  } else if (logical.isset.MAP) {
    return parquet::MAP;
  } else if (logical.isset.LIST) {
    return parquet::LIST;
  } else if (logical.isset.ENUM) {
    return parquet::ENUM;
  } else if (logical.isset.DECIMAL) {
    return parquet::DECIMAL;  // TODO set decimal values
  } else if (logical.isset.DATE) {
    return parquet::DATE;
  } else if (logical.isset.TIME) {
    if (logical.TIME.unit.isset.MILLIS)
      return parquet::TIME_MILLIS;
    else if (logical.TIME.unit.isset.MICROS)
      return parquet::TIME_MICROS;
  } else if (logical.isset.TIMESTAMP) {
    if (logical.TIMESTAMP.unit.isset.MILLIS)
      return parquet::TIMESTAMP_MILLIS;
    else if (logical.TIMESTAMP.unit.isset.MICROS)
      return parquet::TIMESTAMP_MICROS;
  } else if (logical.isset.INTEGER) {
    switch (logical.INTEGER.bitWidth) {
      case 8: return logical.INTEGER.isSigned ? INT_8 : UINT_8;
      case 16: return logical.INTEGER.isSigned ? INT_16 : UINT_16;
      case 32: return logical.INTEGER.isSigned ? INT_32 : UINT_32;
      case 64: return logical.INTEGER.isSigned ? INT_64 : UINT_64;
      default: break;
    }
  } else if (logical.isset.UNKNOWN) {
    return parquet::NA;
  } else if (logical.isset.JSON) {
    return parquet::JSON;
  } else if (logical.isset.BSON) {
    return parquet::BSON;
  }
  return parquet::UNKNOWN;
}

}  // namespace

/**
 * @brief Function that translates Parquet datatype to cuDF type enum
 */
type_id to_type_id(SchemaElement const& schema,
                   bool strings_to_categorical,
                   type_id timestamp_type_id)
{
  parquet::Type const physical            = schema.type;
  parquet::LogicalType const logical_type = schema.logical_type;
  parquet::ConvertedType converted_type   = schema.converted_type;
  int32_t decimal_precision               = schema.decimal_precision;

  // Logical type used for actual data interpretation; the legacy converted type
  // is superseded by 'logical' type whenever available.
  auto const inferred_converted_type = logical_type_to_converted_type(logical_type);
  if (inferred_converted_type != parquet::UNKNOWN) { converted_type = inferred_converted_type; }
  if (inferred_converted_type == parquet::DECIMAL) {
    decimal_precision = schema.logical_type.DECIMAL.precision;
  }

  switch (converted_type) {
    case parquet::UINT_8: return type_id::UINT8;
    case parquet::INT_8: return type_id::INT8;
    case parquet::UINT_16: return type_id::UINT16;
    case parquet::INT_16: return type_id::INT16;
    case parquet::UINT_32: return type_id::UINT32;
    case parquet::UINT_64: return type_id::UINT64;
    case parquet::DATE: return type_id::TIMESTAMP_DAYS;
    case parquet::TIME_MILLIS: return type_id::DURATION_MILLISECONDS;
    case parquet::TIME_MICROS: return type_id::DURATION_MICROSECONDS;
    case parquet::TIMESTAMP_MILLIS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MILLISECONDS;
    case parquet::TIMESTAMP_MICROS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MICROSECONDS;
    case parquet::DECIMAL:
      if (physical == parquet::INT32) { return type_id::DECIMAL32; }
      if (physical == parquet::INT64) { return type_id::DECIMAL64; }
      if (physical == parquet::FIXED_LEN_BYTE_ARRAY) {
        if (schema.type_length <= static_cast<int32_t>(sizeof(int32_t))) {
          return type_id::DECIMAL32;
        }
        if (schema.type_length <= static_cast<int32_t>(sizeof(int64_t))) {
          return type_id::DECIMAL64;
        }
        if (schema.type_length <= static_cast<int32_t>(sizeof(__int128_t))) {
          return type_id::DECIMAL128;
        }
      }
      if (physical == parquet::BYTE_ARRAY) {
        CUDF_EXPECTS(decimal_precision <= MAX_DECIMAL128_PRECISION, "Invalid decimal precision");
        if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
          return type_id::DECIMAL32;
        } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
          return type_id::DECIMAL64;
        } else {
          return type_id::DECIMAL128;
        }
      }
      CUDF_FAIL("Invalid representation of decimal type");
      break;

    // maps are just List<Struct<>>.
    case parquet::MAP:
    case parquet::LIST: return type_id::LIST;
    case parquet::NA: return type_id::STRING;
    // return type_id::EMPTY; //TODO(kn): enable after Null/Empty column support
    default: break;
  }

  if (inferred_converted_type == parquet::UNKNOWN and physical == parquet::INT64 and
      logical_type.TIMESTAMP.unit.isset.NANOS) {
    return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                 : type_id::TIMESTAMP_NANOSECONDS;
  }

  if (inferred_converted_type == parquet::UNKNOWN and physical == parquet::INT64 and
      logical_type.TIME.unit.isset.NANOS) {
    return type_id::DURATION_NANOSECONDS;
  }

  // is it simply a struct?
  if (schema.is_struct()) { return type_id::STRUCT; }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical) {
    case parquet::BOOLEAN: return type_id::BOOL8;
    case parquet::INT32: return type_id::INT32;
    case parquet::INT64: return type_id::INT64;
    case parquet::FLOAT: return type_id::FLOAT32;
    case parquet::DOUBLE: return type_id::FLOAT64;
    case parquet::BYTE_ARRAY:
    case parquet::FIXED_LEN_BYTE_ARRAY:
      // Can be mapped to INT32 (32-bit hash) or STRING
      return strings_to_categorical ? type_id::INT32 : type_id::STRING;
    case parquet::INT96:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    default: break;
  }

  return type_id::EMPTY;
}

metadata::metadata(datasource* source)
{
  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);

  auto const len           = source->size();
  auto const header_buffer = source->host_read(0, header_len);
  auto const header        = reinterpret_cast<file_header_s const*>(header_buffer->data());
  auto const ender_buffer  = source->host_read(len - ender_len, ender_len);
  auto const ender         = reinterpret_cast<file_ender_s const*>(ender_buffer->data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  auto const buffer = source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
  CompactProtocolReader cp(buffer->data(), ender->footer_len);
  CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
  CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
}

std::vector<metadata> aggregate_reader_metadata::metadatas_from_sources(
  std::vector<std::unique_ptr<datasource>> const& sources)
{
  std::vector<metadata> metadatas;
  std::transform(
    sources.cbegin(), sources.cend(), std::back_inserter(metadatas), [](auto const& source) {
      return metadata(source.get());
    });
  return metadatas;
}

std::vector<std::unordered_map<std::string, std::string>>
aggregate_reader_metadata::collect_keyval_metadata() const
{
  std::vector<std::unordered_map<std::string, std::string>> kv_maps;
  std::transform(per_file_metadata.cbegin(),
                 per_file_metadata.cend(),
                 std::back_inserter(kv_maps),
                 [](auto const& pfm) {
                   std::unordered_map<std::string, std::string> kv_map;
                   std::transform(pfm.key_value_metadata.cbegin(),
                                  pfm.key_value_metadata.cend(),
                                  std::inserter(kv_map, kv_map.end()),
                                  [](auto const& kv) {
                                    return std::pair{kv.key, kv.value};
                                  });
                   return kv_map;
                 });

  return kv_maps;
}

int64_t aggregate_reader_metadata::calc_num_rows() const
{
  return std::accumulate(
    per_file_metadata.begin(), per_file_metadata.end(), 0l, [](auto& sum, auto& pfm) {
      auto const rowgroup_rows = std::accumulate(
        pfm.row_groups.begin(), pfm.row_groups.end(), 0l, [](auto& rg_sum, auto& rg) {
          return rg_sum + rg.num_rows;
        });
      CUDF_EXPECTS(pfm.num_rows == 0 || pfm.num_rows == rowgroup_rows,
                   "Header and row groups disagree about number of rows in file!");
      return sum + (pfm.num_rows == 0 && rowgroup_rows > 0 ? rowgroup_rows : pfm.num_rows);
    });
}

size_type aggregate_reader_metadata::calc_num_row_groups() const
{
  return std::accumulate(
    per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto& sum, auto& pfm) {
      return sum + pfm.row_groups.size();
    });
}

aggregate_reader_metadata::aggregate_reader_metadata(
  std::vector<std::unique_ptr<datasource>> const& sources)
  : per_file_metadata(metadatas_from_sources(sources)),
    keyval_maps(collect_keyval_metadata()),
    num_rows(calc_num_rows()),
    num_row_groups(calc_num_row_groups())
{
  if (per_file_metadata.size() > 0) {
    auto const& first_meta = per_file_metadata.front();
    auto const num_cols =
      first_meta.row_groups.size() > 0 ? first_meta.row_groups.front().columns.size() : 0;
    auto const& schema = first_meta.schema;

    // Verify that the input files have matching numbers of columns and schema.
    for (auto const& pfm : per_file_metadata) {
      if (pfm.row_groups.size() > 0) {
        CUDF_EXPECTS(num_cols == pfm.row_groups.front().columns.size(),
                     "All sources must have the same number of columns");
      }
      CUDF_EXPECTS(schema == pfm.schema, "All sources must have the same schema");
    }
  }
}

RowGroup const& aggregate_reader_metadata::get_row_group(size_type row_group_index,
                                                         size_type src_idx) const
{
  CUDF_EXPECTS(src_idx >= 0 && src_idx < static_cast<size_type>(per_file_metadata.size()),
               "invalid source index");
  return per_file_metadata[src_idx].row_groups[row_group_index];
}

ColumnChunkMetaData const& aggregate_reader_metadata::get_column_metadata(size_type row_group_index,
                                                                          size_type src_idx,
                                                                          int schema_idx) const
{
  auto col =
    std::find_if(per_file_metadata[src_idx].row_groups[row_group_index].columns.begin(),
                 per_file_metadata[src_idx].row_groups[row_group_index].columns.end(),
                 [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });
  CUDF_EXPECTS(col != std::end(per_file_metadata[src_idx].row_groups[row_group_index].columns),
               "Found no metadata for schema index");
  return col->meta_data;
}

std::string aggregate_reader_metadata::get_pandas_index() const
{
  // Assumes that all input files have the same metadata
  // TODO: verify this assumption
  auto it = keyval_maps[0].find("pandas");
  if (it != keyval_maps[0].end()) {
    // Captures a list of quoted strings found inside square brackets after `"index_columns":`
    // Inside quotes supports newlines, brackets, escaped quotes, etc.
    // One-liner regex:
    // "index_columns"\s*:\s*\[\s*((?:"(?:|(?:.*?(?![^\\]")).?)[^\\]?",?\s*)*)\]
    // Documented below.
    std::regex index_columns_expr{
      R"("index_columns"\s*:\s*\[\s*)"  // match preamble, opening square bracket, whitespace
      R"(()"                            // Open first capturing group
      R"((?:")"                         // Open non-capturing group match opening quote
      R"((?:|(?:.*?(?![^\\]")).?))"     // match empty string or anything between quotes
      R"([^\\]?")"                      // Match closing non-escaped quote
      R"(,?\s*)"                        // Match optional comma and whitespace
      R"()*)"                           // Close non-capturing group and repeat 0 or more times
      R"())"                            // Close first capturing group
      R"(\])"                           // Match closing square brackets
    };
    std::smatch sm;
    if (std::regex_search(it->second, sm, index_columns_expr)) { return sm[1].str(); }
  }
  return "";
}

std::vector<std::string> aggregate_reader_metadata::get_pandas_index_names() const
{
  std::vector<std::string> names;
  auto str = get_pandas_index();
  if (str.length() != 0) {
    std::regex index_name_expr{R"(\"((?:\\.|[^\"])*)\")"};
    std::smatch sm;
    while (std::regex_search(str, sm, index_name_expr)) {
      if (sm.size() == 2) {  // 2 = whole match, first item
        if (std::find(names.begin(), names.end(), sm[1].str()) == names.end()) {
          std::regex esc_quote{R"(\\")"};
          names.emplace_back(std::regex_replace(sm[1].str(), esc_quote, R"(")"));
        }
      }
      str = sm.suffix();
    }
  }
  return names;
}

std::tuple<int64_t, size_type, std::vector<row_group_info>>
aggregate_reader_metadata::select_row_groups(
  host_span<std::vector<size_type> const> row_group_indices,
  int64_t skip_rows_opt,
  std::optional<size_type> const& num_rows_opt) const
{
  std::vector<row_group_info> selection;
  auto [rows_to_skip, rows_to_read] = [&]() {
    if (not row_group_indices.empty()) { return std::pair<int64_t, size_type>{}; }
    auto const from_opts = cudf::io::detail::skip_rows_num_rows_from_options(
      skip_rows_opt, num_rows_opt, get_num_rows());
    return std::pair{static_cast<int64_t>(from_opts.first), from_opts.second};
  }();

  if (!row_group_indices.empty()) {
    CUDF_EXPECTS(row_group_indices.size() == per_file_metadata.size(),
                 "Must specify row groups for each source");

    for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
      for (auto const& rowgroup_idx : row_group_indices[src_idx]) {
        CUDF_EXPECTS(
          rowgroup_idx >= 0 &&
            rowgroup_idx < static_cast<size_type>(per_file_metadata[src_idx].row_groups.size()),
          "Invalid rowgroup index");
        selection.emplace_back(rowgroup_idx, rows_to_read, src_idx);
        rows_to_read += get_row_group(rowgroup_idx, src_idx).num_rows;
      }
    }
  } else {
    size_type count = 0;
    for (size_t src_idx = 0; src_idx < per_file_metadata.size(); ++src_idx) {
      for (size_t rg_idx = 0; rg_idx < per_file_metadata[src_idx].row_groups.size(); ++rg_idx) {
        auto const chunk_start_row = count;
        count += get_row_group(rg_idx, src_idx).num_rows;
        if (count > rows_to_skip || count == 0) {
          selection.emplace_back(rg_idx, chunk_start_row, src_idx);
        }
        if (count >= rows_to_skip + rows_to_read) { break; }
      }
    }
  }

  return {rows_to_skip, rows_to_read, std::move(selection)};
}

std::tuple<std::vector<input_column_info>,
           std::vector<inline_column_buffer>,
           std::vector<size_type>>
aggregate_reader_metadata::select_columns(std::optional<std::vector<std::string>> const& use_names,
                                          bool include_index,
                                          bool strings_to_categorical,
                                          type_id timestamp_type_id) const
{
  auto find_schema_child = [&](SchemaElement const& schema_elem, std::string const& name) {
    auto const& col_schema_idx =
      std::find_if(schema_elem.children_idx.cbegin(),
                   schema_elem.children_idx.cend(),
                   [&](size_t col_schema_idx) { return get_schema(col_schema_idx).name == name; });

    return (col_schema_idx != schema_elem.children_idx.end())
             ? static_cast<size_type>(*col_schema_idx)
             : -1;
  };

  std::vector<inline_column_buffer> output_columns;
  std::vector<input_column_info> input_columns;
  std::vector<int> nesting;

  // Return true if column path is valid. e.g. if the path is {"struct1", "child1"}, then it is
  // valid if "struct1.child1" exists in this file's schema. If "struct1" exists but "child1" is
  // not a child of "struct1" then the function will return false for "struct1"
  std::function<bool(column_name_info const*, int, std::vector<inline_column_buffer>&, bool)>
    build_column = [&](column_name_info const* col_name_info,
                       int schema_idx,
                       std::vector<inline_column_buffer>& out_col_array,
                       bool has_list_parent) {
      if (schema_idx < 0) { return false; }
      auto const& schema_elem = get_schema(schema_idx);

      // if schema_elem is a stub then it does not exist in the column_name_info and column_buffer
      // hierarchy. So continue on
      if (schema_elem.is_stub()) {
        // is this legit?
        CUDF_EXPECTS(schema_elem.num_children == 1, "Unexpected number of children for stub");
        auto child_col_name_info = (col_name_info) ? &col_name_info->children[0] : nullptr;
        return build_column(
          child_col_name_info, schema_elem.children_idx[0], out_col_array, has_list_parent);
      }

      // if we're at the root, this is a new output column
      auto const col_type = schema_elem.is_one_level_list(get_schema(schema_elem.parent_idx))
                              ? type_id::LIST
                              : to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
      auto const dtype    = to_data_type(col_type, schema_elem);

      inline_column_buffer output_col(dtype, schema_elem.repetition_type == OPTIONAL);
      if (has_list_parent) { output_col.user_data |= PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT; }
      // store the index of this element if inserted in out_col_array
      nesting.push_back(static_cast<int>(out_col_array.size()));
      output_col.name = schema_elem.name;

      // build each child
      bool path_is_valid = false;
      if (col_name_info == nullptr or col_name_info->children.empty()) {
        // add all children of schema_elem.
        // At this point, we can no longer pass a col_name_info to build_column
        for (int idx = 0; idx < schema_elem.num_children; idx++) {
          path_is_valid |= build_column(nullptr,
                                        schema_elem.children_idx[idx],
                                        output_col.children,
                                        has_list_parent || col_type == type_id::LIST);
        }
      } else {
        for (size_t idx = 0; idx < col_name_info->children.size(); idx++) {
          path_is_valid |=
            build_column(&col_name_info->children[idx],
                         find_schema_child(schema_elem, col_name_info->children[idx].name),
                         output_col.children,
                         has_list_parent || col_type == type_id::LIST);
        }
      }

      // if I have no children, we're at a leaf and I'm an input column (that is, one with actual
      // data stored) so add me to the list.
      if (schema_elem.num_children == 0) {
        input_column_info& input_col = input_columns.emplace_back(
          input_column_info{schema_idx, schema_elem.name, schema_elem.max_repetition_level > 0});

        // set up child output column for one-level encoding list
        if (schema_elem.is_one_level_list(get_schema(schema_elem.parent_idx))) {
          // determine the element data type
          auto const element_type =
            to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
          auto const element_dtype = to_data_type(element_type, schema_elem);

          inline_column_buffer element_col(element_dtype, schema_elem.repetition_type == OPTIONAL);
          if (has_list_parent || col_type == type_id::LIST) {
            element_col.user_data |= PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT;
          }
          // store the index of this element
          nesting.push_back(static_cast<int>(output_col.children.size()));
          // TODO: not sure if we should assign a name or leave it blank
          element_col.name = "element";

          output_col.children.push_back(std::move(element_col));
        }

        std::copy(nesting.cbegin(), nesting.cend(), std::back_inserter(input_col.nesting));

        // pop off the extra nesting element.
        if (schema_elem.is_one_level_list(get_schema(schema_elem.parent_idx))) {
          nesting.pop_back();
        }

        path_is_valid = true;  // If we're able to reach leaf then path is valid
      }

      if (path_is_valid) { out_col_array.push_back(std::move(output_col)); }

      nesting.pop_back();
      return path_is_valid;
    };

  std::vector<int> output_column_schemas;

  //
  // there is not necessarily a 1:1 mapping between input columns and output columns.
  // For example, parquet does not explicitly store a ColumnChunkDesc for struct columns.
  // The "structiness" is simply implied by the schema.  For example, this schema:
  //  required group field_id=1 name {
  //    required binary field_id=2 firstname (String);
  //    required binary field_id=3 middlename (String);
  //    required binary field_id=4 lastname (String);
  // }
  // will only contain 3 internal columns of data (firstname, middlename, lastname).  But of
  // course "name" is ultimately the struct column we want to return.
  //
  // "firstname", "middlename" and "lastname" represent the input columns in the file that we
  // process to produce the final cudf "name" column.
  //
  // A user can ask for a single field out of the struct e.g. firstname.
  // In this case they'll pass a fully qualified name to the schema element like
  // ["name", "firstname"]
  //
  auto const& root = get_schema(0);
  if (not use_names.has_value()) {
    for (auto const& schema_idx : root.children_idx) {
      build_column(nullptr, schema_idx, output_columns, false);
      output_column_schemas.push_back(schema_idx);
    }
  } else {
    struct path_info {
      std::string full_path;
      int schema_idx;
    };

    // Convert schema into a vector of every possible path
    std::vector<path_info> all_paths;
    std::function<void(std::string, int)> add_path = [&](std::string path_till_now,
                                                         int schema_idx) {
      auto const& schema_elem = get_schema(schema_idx);
      std::string curr_path   = path_till_now + schema_elem.name;
      all_paths.push_back({curr_path, schema_idx});
      for (auto const& child_idx : schema_elem.children_idx) {
        add_path(curr_path + ".", child_idx);
      }
    };
    for (auto const& child_idx : get_schema(0).children_idx) {
      add_path("", child_idx);
    }

    // Find which of the selected paths are valid and get their schema index
    std::vector<path_info> valid_selected_paths;
    for (auto const& selected_path : *use_names) {
      auto found_path =
        std::find_if(all_paths.begin(), all_paths.end(), [&](path_info& valid_path) {
          return valid_path.full_path == selected_path;
        });
      if (found_path != all_paths.end()) {
        valid_selected_paths.push_back({selected_path, found_path->schema_idx});
      }
    }

    // Now construct paths as vector of strings for further consumption
    std::vector<std::vector<std::string>> use_names3;
    std::transform(valid_selected_paths.begin(),
                   valid_selected_paths.end(),
                   std::back_inserter(use_names3),
                   [&](path_info const& valid_path) {
                     auto schema_idx = valid_path.schema_idx;
                     std::vector<std::string> result_path;
                     do {
                       SchemaElement const& elem = get_schema(schema_idx);
                       result_path.push_back(elem.name);
                       schema_idx = elem.parent_idx;
                     } while (schema_idx > 0);
                     return std::vector<std::string>(result_path.rbegin(), result_path.rend());
                   });

    std::vector<column_name_info> selected_columns;
    if (include_index) {
      std::vector<std::string> index_names = get_pandas_index_names();
      std::transform(index_names.cbegin(),
                     index_names.cend(),
                     std::back_inserter(selected_columns),
                     [](std::string const& name) { return column_name_info(name); });
    }
    // Merge the vector use_names into a set of hierarchical column_name_info objects
    /* This is because if we have columns like this:
     *     col1
     *      / \
     *    s3   f4
     *   / \
     * f5   f6
     *
     * there may be common paths in use_names like:
     * {"col1", "s3", "f5"}, {"col1", "f4"}
     * which means we want the output to contain
     *     col1
     *      / \
     *    s3   f4
     *   /
     * f5
     *
     * rather than
     *  col1   col1
     *   |      |
     *   s3     f4
     *   |
     *   f5
     */
    for (auto const& path : use_names3) {
      auto array_to_find_in = &selected_columns;
      for (size_t depth = 0; depth < path.size(); ++depth) {
        // Check if the path exists in our selected_columns and if not, add it.
        auto const& name_to_find = path[depth];
        auto found_col           = std::find_if(
          array_to_find_in->begin(),
          array_to_find_in->end(),
          [&name_to_find](column_name_info const& col) { return col.name == name_to_find; });
        if (found_col == array_to_find_in->end()) {
          auto& col        = array_to_find_in->emplace_back(name_to_find);
          array_to_find_in = &col.children;
        } else {
          // Path exists. go down further.
          array_to_find_in = &found_col->children;
        }
      }
    }
    for (auto& col : selected_columns) {
      auto const& top_level_col_schema_idx = find_schema_child(root, col.name);
      bool valid_column = build_column(&col, top_level_col_schema_idx, output_columns, false);
      if (valid_column) output_column_schemas.push_back(top_level_col_schema_idx);
    }
  }

  return std::make_tuple(
    std::move(input_columns), std::move(output_columns), std::move(output_column_schemas));
}

}  // namespace cudf::io::detail::parquet
