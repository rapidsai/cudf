/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/**
 * @file reader_impl.cu
 * @brief cuDF-IO Parquet reader class implementation
 */

#include "reader_impl.hpp"

#include "compact_protocol_reader.hpp"

#include <io/comp/gpuinflate.hpp>
#include <io/comp/nvcomp_adapter.hpp>
#include <io/utilities/config_utils.hpp>
#include <io/utilities/time_utils.cuh>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <regex>

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
// Import functionality that's independent of legacy code
using namespace cudf::io::parquet;
using namespace cudf::io;

// bit space we are reserving in column_buffer::user_data
constexpr uint32_t PARQUET_COLUMN_BUFFER_SCHEMA_MASK          = (0xffffff);
constexpr uint32_t PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED = (1 << 24);

namespace {

parquet::ConvertedType logical_type_to_converted_type(parquet::LogicalType const& logical)
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
  int32_t decimal_scale                   = schema.decimal_scale;

  // Logical type used for actual data interpretation; the legacy converted type
  // is superceded by 'logical' type whenever available.
  auto const inferred_converted_type = logical_type_to_converted_type(logical_type);
  if (inferred_converted_type != parquet::UNKNOWN) converted_type = inferred_converted_type;
  if (inferred_converted_type == parquet::DECIMAL && decimal_scale == 0)
    decimal_scale = schema.logical_type.DECIMAL.scale;

  switch (converted_type) {
    case parquet::UINT_8: return type_id::UINT8;
    case parquet::INT_8: return type_id::INT8;
    case parquet::UINT_16: return type_id::UINT16;
    case parquet::INT_16: return type_id::INT16;
    case parquet::UINT_32: return type_id::UINT32;
    case parquet::UINT_64: return type_id::UINT64;
    case parquet::DATE: return type_id::TIMESTAMP_DAYS;
    case parquet::TIME_MILLIS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::DURATION_MILLISECONDS;
    case parquet::TIME_MICROS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::DURATION_MICROSECONDS;
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

/**
 * @brief Converts cuDF type enum to column logical type
 */
data_type to_data_type(type_id t_id, SchemaElement const& schema)
{
  return t_id == type_id::DECIMAL32 || t_id == type_id::DECIMAL64 || t_id == type_id::DECIMAL128
           ? data_type{t_id, numeric::scale_type{-schema.decimal_scale}}
           : data_type{t_id};
}

/**
 * @brief Function that returns the required the number of bits to store a value
 */
template <typename T = uint8_t>
T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet type width, Parquet clock rate and Parquet decimal type.
 */
std::tuple<int32_t, int32_t, int8_t> conversion_info(type_id column_type_id,
                                                     type_id timestamp_type_id,
                                                     parquet::Type physical,
                                                     int8_t converted,
                                                     int32_t length)
{
  int32_t type_width = (physical == parquet::FIXED_LEN_BYTE_ARRAY) ? length : 0;
  int32_t clock_rate = 0;
  if (column_type_id == type_id::INT8 or column_type_id == type_id::UINT8) {
    type_width = 1;  // I32 -> I8
  } else if (column_type_id == type_id::INT16 or column_type_id == type_id::UINT16) {
    type_width = 2;  // I32 -> I16
  } else if (column_type_id == type_id::INT32) {
    type_width = 4;  // str -> hash32
  } else if (is_chrono(data_type{column_type_id})) {
    clock_rate = to_clockrate(timestamp_type_id);
  }

  int8_t converted_type = converted;
  if (converted_type == parquet::DECIMAL && column_type_id != type_id::FLOAT64 &&
      not cudf::is_fixed_point(column_type_id)) {
    converted_type = parquet::UNKNOWN;  // Not converting to float64 or decimal
  }
  return std::make_tuple(type_width, clock_rate, converted_type);
}

inline void decompress_check(device_span<decompress_status const> stats,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream),
                              stats.begin(),
                              stats.end(),
                              [] __device__(auto const& stat) { return stat.status == 0; }),
               "Error during decompression");
}
}  // namespace

std::string name_from_path(const std::vector<std::string>& path_in_schema)
{
  // For the case of lists, we will see a schema that looks like:
  // a.list.element.list.element
  // where each (list.item) pair represents a level of nesting.  According to the parquet spec,
  // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
  // the initial field must be named "list" and the inner element must be named "element".
  // If we are dealing with a list, we want to return the topmost name of the group ("a").
  //
  // For other nested schemas, like structs we just want to return the bottom-most name. For
  // example a struct with the schema
  // b.employee.id,  the column representing "id" should simply be named "id".
  //
  // In short, this means : return the highest level of the schema that does not have list
  // definitions underneath it.
  //
  std::string s = (path_in_schema.size() > 0) ? path_in_schema[0] : "";
  for (size_t i = 1; i < path_in_schema.size(); i++) {
    // The Parquet spec requires that the outer schema field is named "list". However it also
    // provides a list of backwards compatibility cases that are applicable as well.  Currently
    // we are only handling the formal spec.  This will get cleaned up and improved when we add
    // support for structs. The correct thing to do will probably be to examine the type of
    // the SchemaElement itself to concretely identify the start of a nested type of any kind rather
    // than trying to derive it from the path string.
    if (path_in_schema[i] == "list") {
      // Again, strictly speaking, the Parquet spec says the inner field should be named
      // "element", but there are some backwards compatibility issues that we have seen in the
      // wild. For example, Pandas calls the field "item".  We will allow any name for now.
      i++;
      continue;
    }
    // otherwise, we've got a real nested column. update the name
    s = path_in_schema[i];
  }
  return s;
}

/**
 * @brief Class for parsing dataset metadata
 */
struct metadata : public FileMetaData {
  explicit metadata(datasource* source)
  {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);

    const auto len           = source->size();
    const auto header_buffer = source->host_read(0, header_len);
    const auto header        = reinterpret_cast<const file_header_s*>(header_buffer->data());
    const auto ender_buffer  = source->host_read(len - ender_len, ender_len);
    const auto ender         = reinterpret_cast<const file_ender_s*>(ender_buffer->data());
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
    CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
                 "Corrupted header or footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    const auto buffer = source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
    CompactProtocolReader cp(buffer->data(), ender->footer_len);
    CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
    CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
  }
};

class aggregate_reader_metadata {
  std::vector<metadata> per_file_metadata;
  std::vector<std::unordered_map<std::string, std::string>> keyval_maps;
  size_type num_rows;
  size_type num_row_groups;
  /**
   * @brief Create a metadata object from each element in the source vector
   */
  auto metadatas_from_sources(std::vector<std::unique_ptr<datasource>> const& sources)
  {
    std::vector<metadata> metadatas;
    std::transform(
      sources.cbegin(), sources.cend(), std::back_inserter(metadatas), [](auto const& source) {
        return metadata(source.get());
      });
    return metadatas;
  }

  /**
   * @brief Collect the keyvalue maps from each per-file metadata object into a vector of maps.
   */
  [[nodiscard]] auto collect_keyval_metadata()
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

  /**
   * @brief Sums up the number of rows of each source
   */
  [[nodiscard]] size_type calc_num_rows() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto& sum, auto& pfm) {
        return sum + pfm.num_rows;
      });
  }

  /**
   * @brief Sums up the number of row groups of each source
   */
  [[nodiscard]] size_type calc_num_row_groups() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto& sum, auto& pfm) {
        return sum + pfm.row_groups.size();
      });
  }

 public:
  aggregate_reader_metadata(std::vector<std::unique_ptr<datasource>> const& sources)
    : per_file_metadata(metadatas_from_sources(sources)),
      keyval_maps(collect_keyval_metadata()),
      num_rows(calc_num_rows()),
      num_row_groups(calc_num_row_groups())
  {
    // Verify that the input files have matching numbers of columns
    size_type num_cols = -1;
    for (auto const& pfm : per_file_metadata) {
      if (pfm.row_groups.size() != 0) {
        if (num_cols == -1)
          num_cols = pfm.row_groups[0].columns.size();
        else
          CUDF_EXPECTS(num_cols == static_cast<size_type>(pfm.row_groups[0].columns.size()),
                       "All sources must have the same number of columns");
      }
    }
    // Verify that the input files have matching schemas
    for (auto const& pfm : per_file_metadata) {
      CUDF_EXPECTS(per_file_metadata[0].schema == pfm.schema,
                   "All sources must have the same schemas");
    }
  }

  [[nodiscard]] auto const& get_row_group(size_type row_group_index, size_type src_idx) const
  {
    CUDF_EXPECTS(src_idx >= 0 && src_idx < static_cast<size_type>(per_file_metadata.size()),
                 "invalid source index");
    return per_file_metadata[src_idx].row_groups[row_group_index];
  }

  [[nodiscard]] auto const& get_column_metadata(size_type row_group_index,
                                                size_type src_idx,
                                                int schema_idx) const
  {
    auto col = std::find_if(
      per_file_metadata[src_idx].row_groups[row_group_index].columns.begin(),
      per_file_metadata[src_idx].row_groups[row_group_index].columns.end(),
      [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx ? true : false; });
    CUDF_EXPECTS(col != std::end(per_file_metadata[src_idx].row_groups[row_group_index].columns),
                 "Found no metadata for schema index");
    return col->meta_data;
  }

  [[nodiscard]] auto get_num_rows() const { return num_rows; }

  [[nodiscard]] auto get_num_row_groups() const { return num_row_groups; }

  [[nodiscard]] auto const& get_schema(int schema_idx) const
  {
    return per_file_metadata[0].schema[schema_idx];
  }

  [[nodiscard]] auto const& get_key_value_metadata() const { return keyval_maps; }

  /**
   * @brief Gets the concrete nesting depth of output cudf columns
   *
   * @param schema_index Schema index of the input column
   *
   * @return comma-separated index column names in quotes
   */
  [[nodiscard]] inline int get_output_nesting_depth(int schema_index) const
  {
    auto& pfm = per_file_metadata[0];
    int depth = 0;

    // walk upwards, skipping repeated fields
    while (schema_index > 0) {
      if (!pfm.schema[schema_index].is_stub()) { depth++; }
      // schema of one-level encoding list doesn't contain nesting information, so we need to
      // manually add an extra nesting level
      if (pfm.schema[schema_index].is_one_level_list()) { depth++; }
      schema_index = pfm.schema[schema_index].parent_idx;
    }
    return depth;
  }

  /**
   * @brief Extracts the pandas "index_columns" section
   *
   * PANDAS adds its own metadata to the key_value section when writing out the
   * dataframe to a file to aid in exact reconstruction. The JSON-formatted
   * metadata contains the index column(s) and PANDA-specific datatypes.
   *
   * @return comma-separated index column names in quotes
   */
  [[nodiscard]] std::string get_pandas_index() const
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

  /**
   * @brief Extracts the column name(s) used for the row indexes in a dataframe
   *
   * @param names List of column names to load, where index column name(s) will be added
   */
  [[nodiscard]] std::vector<std::string> get_pandas_index_names() const
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

  struct row_group_info {
    size_type const index;
    size_t const start_row;  // TODO source index
    size_type const source_index;
    row_group_info(size_type index, size_t start_row, size_type source_index)
      : index(index), start_row(start_row), source_index(source_index)
    {
    }
  };

  /**
   * @brief Filters and reduces down to a selection of row groups
   *
   * @param row_groups Lists of row group to reads, one per source
   * @param row_start Starting row of the selection
   * @param row_count Total number of rows selected
   *
   * @return List of row group indexes and its starting row
   */
  [[nodiscard]] auto select_row_groups(std::vector<std::vector<size_type>> const& row_groups,
                                       size_type& row_start,
                                       size_type& row_count) const
  {
    if (!row_groups.empty()) {
      std::vector<row_group_info> selection;
      CUDF_EXPECTS(row_groups.size() == per_file_metadata.size(),
                   "Must specify row groups for each source");

      row_count = 0;
      for (size_t src_idx = 0; src_idx < row_groups.size(); ++src_idx) {
        for (auto const& rowgroup_idx : row_groups[src_idx]) {
          CUDF_EXPECTS(
            rowgroup_idx >= 0 &&
              rowgroup_idx < static_cast<size_type>(per_file_metadata[src_idx].row_groups.size()),
            "Invalid rowgroup index");
          selection.emplace_back(rowgroup_idx, row_count, src_idx);
          row_count += get_row_group(rowgroup_idx, src_idx).num_rows;
        }
      }
      return selection;
    }

    row_start = std::max(row_start, 0);
    if (row_count < 0) {
      row_count = static_cast<size_type>(
        std::min<int64_t>(get_num_rows(), std::numeric_limits<size_type>::max()));
    }
    row_count = min(row_count, get_num_rows() - row_start);
    CUDF_EXPECTS(row_count >= 0, "Invalid row count");
    CUDF_EXPECTS(row_start <= get_num_rows(), "Invalid row start");

    std::vector<row_group_info> selection;
    size_type count = 0;
    for (size_t src_idx = 0; src_idx < per_file_metadata.size(); ++src_idx) {
      for (size_t rg_idx = 0; rg_idx < per_file_metadata[src_idx].row_groups.size(); ++rg_idx) {
        auto const chunk_start_row = count;
        count += get_row_group(rg_idx, src_idx).num_rows;
        if (count > row_start || count == 0) {
          selection.emplace_back(rg_idx, chunk_start_row, src_idx);
        }
        if (count >= row_start + row_count) { break; }
      }
    }

    return selection;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of paths of column names to select; `nullopt` if user did not select
   * columns to read
   * @param include_index Whether to always include the PANDAS index column(s)
   * @param strings_to_categorical Type conversion parameter
   * @param timestamp_type_id Type conversion parameter
   *
   * @return input column information, output column information, list of output column schema
   * indices
   */
  [[nodiscard]] auto select_columns(std::optional<std::vector<std::string>> const& use_names,
                                    bool include_index,
                                    bool strings_to_categorical,
                                    type_id timestamp_type_id) const
  {
    auto find_schema_child = [&](SchemaElement const& schema_elem, std::string const& name) {
      auto const& col_schema_idx = std::find_if(
        schema_elem.children_idx.cbegin(),
        schema_elem.children_idx.cend(),
        [&](size_t col_schema_idx) { return get_schema(col_schema_idx).name == name; });

      return (col_schema_idx != schema_elem.children_idx.end()) ? static_cast<int>(*col_schema_idx)
                                                                : -1;
    };

    std::vector<column_buffer> output_columns;
    std::vector<input_column_info> input_columns;
    std::vector<int> nesting;

    // Return true if column path is valid. e.g. if the path is {"struct1", "child1"}, then it is
    // valid if "struct1.child1" exists in this file's schema. If "struct1" exists but "child1" is
    // not a child of "struct1" then the function will return false for "struct1"
    std::function<bool(column_name_info const*, int, std::vector<column_buffer>&)> build_column =
      [&](column_name_info const* col_name_info,
          int schema_idx,
          std::vector<column_buffer>& out_col_array) {
        if (schema_idx < 0) { return false; }
        auto const& schema_elem = get_schema(schema_idx);

        // if schema_elem is a stub then it does not exist in the column_name_info and column_buffer
        // hierarchy. So continue on
        if (schema_elem.is_stub()) {
          // is this legit?
          CUDF_EXPECTS(schema_elem.num_children == 1, "Unexpected number of children for stub");
          auto child_col_name_info = (col_name_info) ? &col_name_info->children[0] : nullptr;
          return build_column(child_col_name_info, schema_elem.children_idx[0], out_col_array);
        }

        // if we're at the root, this is a new output column
        auto const col_type =
          schema_elem.is_one_level_list()
            ? type_id::LIST
            : to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
        auto const dtype = to_data_type(col_type, schema_elem);

        column_buffer output_col(dtype, schema_elem.repetition_type == OPTIONAL);
        // store the index of this element if inserted in out_col_array
        nesting.push_back(static_cast<int>(out_col_array.size()));
        output_col.name = schema_elem.name;

        // build each child
        bool path_is_valid = false;
        if (col_name_info == nullptr or col_name_info->children.empty()) {
          // add all children of schema_elem.
          // At this point, we can no longer pass a col_name_info to build_column
          for (int idx = 0; idx < schema_elem.num_children; idx++) {
            path_is_valid |=
              build_column(nullptr, schema_elem.children_idx[idx], output_col.children);
          }
        } else {
          for (size_t idx = 0; idx < col_name_info->children.size(); idx++) {
            path_is_valid |=
              build_column(&col_name_info->children[idx],
                           find_schema_child(schema_elem, col_name_info->children[idx].name),
                           output_col.children);
          }
        }

        // if I have no children, we're at a leaf and I'm an input column (that is, one with actual
        // data stored) so add me to the list.
        if (schema_elem.num_children == 0) {
          input_column_info& input_col =
            input_columns.emplace_back(input_column_info{schema_idx, schema_elem.name});

          // set up child output column for one-level encoding list
          if (schema_elem.is_one_level_list()) {
            // determine the element data type
            auto const element_type =
              to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
            auto const element_dtype = to_data_type(element_type, schema_elem);

            column_buffer element_col(element_dtype, schema_elem.repetition_type == OPTIONAL);
            // store the index of this element
            nesting.push_back(static_cast<int>(output_col.children.size()));
            // TODO: not sure if we should assign a name or leave it blank
            element_col.name = "element";

            output_col.children.push_back(std::move(element_col));
          }

          std::copy(nesting.cbegin(), nesting.cend(), std::back_inserter(input_col.nesting));

          // pop off the extra nesting element.
          if (schema_elem.is_one_level_list()) { nesting.pop_back(); }

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
        build_column(nullptr, schema_idx, output_columns);
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
        bool valid_column = build_column(&col, top_level_col_schema_idx, output_columns);
        if (valid_column) output_column_schemas.push_back(top_level_col_schema_idx);
      }
    }

    return std::make_tuple(
      std::move(input_columns), std::move(output_columns), std::move(output_column_schemas));
  }
};

/**
 * @brief Generate depth remappings for repetition and definition levels.
 *
 * When dealing with columns that contain lists, we must examine incoming
 * repetition and definition level pairs to determine what range of output nesting
 * is indicated when adding new values.  This function generates the mappings of
 * the R/D levels to those start/end bounds
 *
 * @param remap Maps column schema index to the R/D remapping vectors for that column
 * @param src_col_schema The column schema to generate the new mapping for
 * @param md File metadata information
 */
void generate_depth_remappings(std::map<int, std::pair<std::vector<int>, std::vector<int>>>& remap,
                               int src_col_schema,
                               aggregate_reader_metadata const& md)
{
  // already generated for this level
  if (remap.find(src_col_schema) != remap.end()) { return; }
  auto schema   = md.get_schema(src_col_schema);
  int max_depth = md.get_output_nesting_depth(src_col_schema);

  CUDF_EXPECTS(remap.find(src_col_schema) == remap.end(),
               "Attempting to remap a schema more than once");
  auto inserted =
    remap.insert(std::pair<int, std::pair<std::vector<int>, std::vector<int>>>{src_col_schema, {}});
  auto& depth_remap = inserted.first->second;

  std::vector<int>& rep_depth_remap = (depth_remap.first);
  rep_depth_remap.resize(schema.max_repetition_level + 1);
  std::vector<int>& def_depth_remap = (depth_remap.second);
  def_depth_remap.resize(schema.max_definition_level + 1);

  // the key:
  // for incoming level values  R/D
  // add values starting at the shallowest nesting level X has repetition level R
  // until you reach the deepest nesting level Y that corresponds to the repetition level R1
  // held by the nesting level that has definition level D
  //
  // Example: a 3 level struct with a list at the bottom
  //
  //                     R / D   Depth
  // level0              0 / 1     0
  //   level1            0 / 2     1
  //     level2          0 / 3     2
  //       list          0 / 3     3
  //         element     1 / 4     4
  //
  // incoming R/D : 0, 0  -> add values from depth 0 to 3   (def level 0 always maps to depth 0)
  // incoming R/D : 0, 1  -> add values from depth 0 to 3
  // incoming R/D : 0, 2  -> add values from depth 0 to 3
  // incoming R/D : 1, 4  -> add values from depth 4 to 4
  //
  // Note : the -validity- of values is simply checked by comparing the incoming D value against the
  // D value of the given nesting level (incoming D >= the D for the nesting level == valid,
  // otherwise NULL).  The tricky part is determining what nesting levels to add values at.
  //
  // For schemas with no repetition level (no lists), X is always 0 and Y is always max nesting
  // depth.
  //

  // compute "X" from above
  for (int s_idx = schema.max_repetition_level; s_idx >= 0; s_idx--) {
    auto find_shallowest = [&](int r) {
      int shallowest = -1;
      int cur_depth  = max_depth - 1;
      int schema_idx = src_col_schema;
      while (schema_idx > 0) {
        auto cur_schema = md.get_schema(schema_idx);
        if (cur_schema.max_repetition_level == r) {
          // if this is a repeated field, map it one level deeper
          shallowest = cur_schema.is_stub() ? cur_depth + 1 : cur_depth;
        }
        // if it's one-level encoding list
        else if (cur_schema.is_one_level_list()) {
          shallowest = cur_depth - 1;
        }
        if (!cur_schema.is_stub()) { cur_depth--; }
        schema_idx = cur_schema.parent_idx;
      }
      return shallowest;
    };
    rep_depth_remap[s_idx] = find_shallowest(s_idx);
  }

  // compute "Y" from above
  for (int s_idx = schema.max_definition_level; s_idx >= 0; s_idx--) {
    auto find_deepest = [&](int d) {
      SchemaElement prev_schema;
      int schema_idx = src_col_schema;
      int r1         = 0;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx);
        if (cur_schema.max_definition_level == d) {
          // if this is a repeated field, map it one level deeper
          r1 = cur_schema.is_stub() ? prev_schema.max_repetition_level
                                    : cur_schema.max_repetition_level;
          break;
        }
        prev_schema = cur_schema;
        schema_idx  = cur_schema.parent_idx;
      }

      // we now know R1 from above. return the deepest nesting level that has the
      // same repetition level
      schema_idx = src_col_schema;
      int depth  = max_depth - 1;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx);
        if (cur_schema.max_repetition_level == r1) {
          // if this is a repeated field, map it one level deeper
          depth = cur_schema.is_stub() ? depth + 1 : depth;
          break;
        }
        if (!cur_schema.is_stub()) { depth--; }
        prev_schema = cur_schema;
        schema_idx  = cur_schema.parent_idx;
      }
      return depth;
    };
    def_depth_remap[s_idx] = find_deepest(s_idx);
  }
}

/**
 * @copydoc cudf::io::detail::parquet::read_column_chunks
 */
std::future<void> reader::impl::read_column_chunks(
  std::vector<std::unique_ptr<datasource::buffer>>& page_data,
  hostdevice_vector<gpu::ColumnChunkDesc>& chunks,  // TODO const?
  size_t begin_chunk,
  size_t end_chunk,
  const std::vector<size_t>& column_chunk_offsets,
  std::vector<size_type> const& chunk_source_map)
{
  // Transfer chunk data, coalescing adjacent chunks
  std::vector<std::future<size_t>> read_tasks;
  for (size_t chunk = begin_chunk; chunk < end_chunk;) {
    const size_t io_offset   = column_chunk_offsets[chunk];
    size_t io_size           = chunks[chunk].compressed_size;
    size_t next_chunk        = chunk + 1;
    const bool is_compressed = (chunks[chunk].codec != parquet::Compression::UNCOMPRESSED);
    while (next_chunk < end_chunk) {
      const size_t next_offset = column_chunk_offsets[next_chunk];
      const bool is_next_compressed =
        (chunks[next_chunk].codec != parquet::Compression::UNCOMPRESSED);
      if (next_offset != io_offset + io_size || is_next_compressed != is_compressed) {
        // Can't merge if not contiguous or mixing compressed and uncompressed
        // Not coalescing uncompressed with compressed chunks is so that compressed buffers can be
        // freed earlier (immediately after decompression stage) to limit peak memory requirements
        break;
      }
      io_size += chunks[next_chunk].compressed_size;
      next_chunk++;
    }
    if (io_size != 0) {
      auto& source = _sources[chunk_source_map[chunk]];
      if (source->is_device_read_preferred(io_size)) {
        auto buffer        = rmm::device_buffer(io_size, _stream);
        auto fut_read_size = source->device_read_async(
          io_offset, io_size, static_cast<uint8_t*>(buffer.data()), _stream);
        read_tasks.emplace_back(std::move(fut_read_size));
        page_data[chunk] = datasource::buffer::create(std::move(buffer));
      } else {
        auto const buffer = source->host_read(io_offset, io_size);
        page_data[chunk] =
          datasource::buffer::create(rmm::device_buffer(buffer->data(), buffer->size(), _stream));
      }
      auto d_compdata = page_data[chunk]->data();
      do {
        chunks[chunk].compressed_data = d_compdata;
        d_compdata += chunks[chunk].compressed_size;
      } while (++chunk != next_chunk);
    } else {
      chunk = next_chunk;
    }
  }
  auto sync_fn = [](decltype(read_tasks) read_tasks) {
    for (auto& task : read_tasks) {
      task.wait();
    }
  };
  return std::async(std::launch::deferred, sync_fn, std::move(read_tasks));
}

/**
 * @copydoc cudf::io::detail::parquet::count_page_headers
 */
size_t reader::impl::count_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks)
{
  size_t total_pages = 0;

  chunks.host_to_device(_stream);
  gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), _stream);
  chunks.device_to_host(_stream, true);

  for (size_t c = 0; c < chunks.size(); c++) {
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

/**
 * @copydoc cudf::io::detail::parquet::decode_page_headers
 */
void reader::impl::decode_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                       hostdevice_vector<gpu::PageInfo>& pages)
{
  // IMPORTANT : if you change how pages are stored within a chunk (dist pages, then data pages),
  // please update preprocess_nested_columns to reflect this.
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    chunks[c].max_num_pages = chunks[c].num_data_pages + chunks[c].num_dict_pages;
    chunks[c].page_info     = pages.device_ptr(page_count);
    page_count += chunks[c].max_num_pages;
  }

  chunks.host_to_device(_stream);
  gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), _stream);
  pages.device_to_host(_stream, true);
}

/**
 * @copydoc cudf::io::detail::parquet::decompress_page_data
 */
rmm::device_buffer reader::impl::decompress_page_data(
  hostdevice_vector<gpu::ColumnChunkDesc>& chunks, hostdevice_vector<gpu::PageInfo>& pages)
{
  auto for_each_codec_page = [&](parquet::Compression codec, const std::function<void(size_t)>& f) {
    for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
      const auto page_stride = chunks[c].max_num_pages;
      if (chunks[c].codec == codec) {
        for (int k = 0; k < page_stride; k++) {
          f(page_count + k);
        }
      }
      page_count += page_stride;
    }
  };

  // Brotli scratch memory for decompressing
  rmm::device_buffer debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;

  struct codec_stats {
    parquet::Compression compression_type = UNCOMPRESSED;
    size_t num_pages                      = 0;
    int32_t max_decompressed_size         = 0;
    size_t total_decomp_size              = 0;
  };

  std::array codecs{codec_stats{parquet::GZIP},
                    codec_stats{parquet::SNAPPY},
                    codec_stats{parquet::BROTLI},
                    codec_stats{parquet::ZSTD}};

  auto is_codec_supported = [&codecs](int8_t codec) {
    if (codec == parquet::UNCOMPRESSED) return true;
    return std::find_if(codecs.begin(), codecs.end(), [codec](auto& cstats) {
             return codec == cstats.compression_type;
           }) != codecs.end();
  };
  CUDF_EXPECTS(std::all_of(chunks.begin(),
                           chunks.end(),
                           [&is_codec_supported](auto const& chunk) {
                             return is_codec_supported(chunk.codec);
                           }),
               "Unsupported compression type");

  for (auto& codec : codecs) {
    for_each_codec_page(codec.compression_type, [&](size_t page) {
      auto page_uncomp_size = pages[page].uncompressed_page_size;
      total_decomp_size += page_uncomp_size;
      codec.total_decomp_size += page_uncomp_size;
      codec.max_decompressed_size = std::max(codec.max_decompressed_size, page_uncomp_size);
      codec.num_pages++;
      num_comp_pages++;
    });
    if (codec.compression_type == parquet::BROTLI && codec.num_pages > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.num_pages), _stream);
    }
  }

  // Dispatch batches of pages to decompress for each codec
  rmm::device_buffer decomp_pages(total_decomp_size, _stream);

  std::vector<device_span<uint8_t const>> comp_in;
  comp_in.reserve(num_comp_pages);
  std::vector<device_span<uint8_t>> comp_out;
  comp_out.reserve(num_comp_pages);

  rmm::device_uvector<decompress_status> comp_stats(num_comp_pages, _stream);
  thrust::fill(rmm::exec_policy(_stream),
               comp_stats.begin(),
               comp_stats.end(),
               decompress_status{0, static_cast<uint32_t>(-1000), 0});

  size_t decomp_offset = 0;
  int32_t start_pos    = 0;
  for (const auto& codec : codecs) {
    if (codec.num_pages == 0) { continue; }

    for_each_codec_page(codec.compression_type, [&](size_t page) {
      auto dst_base = static_cast<uint8_t*>(decomp_pages.data());
      comp_in.emplace_back(pages[page].page_data,
                           static_cast<size_t>(pages[page].compressed_page_size));
      comp_out.emplace_back(dst_base + decomp_offset,
                            static_cast<size_t>(pages[page].uncompressed_page_size));

      pages[page].page_data = static_cast<uint8_t*>(comp_out.back().data());
      decomp_offset += comp_out.back().size();
    });

    host_span<device_span<uint8_t const> const> comp_in_view{comp_in.data() + start_pos,
                                                             codec.num_pages};
    auto const d_comp_in = cudf::detail::make_device_uvector_async(comp_in_view, _stream);
    host_span<device_span<uint8_t> const> comp_out_view(comp_out.data() + start_pos,
                                                        codec.num_pages);
    auto const d_comp_out = cudf::detail::make_device_uvector_async(comp_out_view, _stream);
    device_span<decompress_status> d_comp_stats_view(comp_stats.data() + start_pos,
                                                     codec.num_pages);

    switch (codec.compression_type) {
      case parquet::GZIP:
        gpuinflate(d_comp_in, d_comp_out, d_comp_stats_view, gzip_header_included::YES, _stream);
        break;
      case parquet::SNAPPY:
        if (nvcomp_integration::is_stable_enabled()) {
          nvcomp::batched_decompress(nvcomp::compression_type::SNAPPY,
                                     d_comp_in,
                                     d_comp_out,
                                     d_comp_stats_view,
                                     codec.max_decompressed_size,
                                     codec.total_decomp_size,
                                     _stream);
        } else {
          gpu_unsnap(d_comp_in, d_comp_out, d_comp_stats_view, _stream);
        }
        break;
      case parquet::ZSTD:
        nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                                   d_comp_in,
                                   d_comp_out,
                                   d_comp_stats_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   _stream);
        break;
      case parquet::BROTLI:
        gpu_debrotli(d_comp_in,
                     d_comp_out,
                     d_comp_stats_view,
                     debrotli_scratch.data(),
                     debrotli_scratch.size(),
                     _stream);
        break;
      default: CUDF_FAIL("Unexpected decompression dispatch"); break;
    }
    start_pos += codec.num_pages;
  }

  decompress_check(comp_stats, _stream);

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  pages.host_to_device(_stream);

  return decomp_pages;
}

/**
 * @copydoc cudf::io::detail::parquet::allocate_nesting_info
 */
void reader::impl::allocate_nesting_info(hostdevice_vector<gpu::ColumnChunkDesc> const& chunks,
                                         hostdevice_vector<gpu::PageInfo>& pages,
                                         hostdevice_vector<gpu::PageNestingInfo>& page_nesting_info)
{
  // compute total # of page_nesting infos needed and allocate space. doing this in one
  // buffer to keep it to a single gpu allocation
  size_t const total_page_nesting_infos = std::accumulate(
    chunks.host_ptr(), chunks.host_ptr() + chunks.size(), 0, [&](int total, auto& chunk) {
      // the schema of the input column
      auto const& schema                    = _metadata->get_schema(chunk.src_col_schema);
      auto const per_page_nesting_info_size = max(
        schema.max_definition_level + 1, _metadata->get_output_nesting_depth(chunk.src_col_schema));
      return total + (per_page_nesting_info_size * chunk.num_data_pages);
    });

  page_nesting_info = hostdevice_vector<gpu::PageNestingInfo>{total_page_nesting_infos, _stream};

  // retrieve from the gpu so we can update
  pages.device_to_host(_stream, true);

  // update pointers in the PageInfos
  int target_page_index = 0;
  int src_info_index    = 0;
  for (size_t idx = 0; idx < chunks.size(); idx++) {
    int src_col_schema                    = chunks[idx].src_col_schema;
    auto& schema                          = _metadata->get_schema(src_col_schema);
    auto const per_page_nesting_info_size = std::max(
      schema.max_definition_level + 1, _metadata->get_output_nesting_depth(src_col_schema));

    // skip my dict pages
    target_page_index += chunks[idx].num_dict_pages;
    for (int p_idx = 0; p_idx < chunks[idx].num_data_pages; p_idx++) {
      pages[target_page_index + p_idx].nesting = page_nesting_info.device_ptr() + src_info_index;
      pages[target_page_index + p_idx].num_nesting_levels = per_page_nesting_info_size;

      src_info_index += per_page_nesting_info_size;
    }
    target_page_index += chunks[idx].num_data_pages;
  }

  // copy back to the gpu
  pages.host_to_device(_stream);

  // fill in
  int nesting_info_index = 0;
  std::map<int, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
  for (size_t idx = 0; idx < chunks.size(); idx++) {
    int src_col_schema = chunks[idx].src_col_schema;

    // schema of the input column
    auto& schema = _metadata->get_schema(src_col_schema);
    // real depth of the output cudf column hierarchy (1 == no nesting, 2 == 1 level, etc)
    int max_depth = _metadata->get_output_nesting_depth(src_col_schema);

    // # of nesting infos stored per page for this column
    auto const per_page_nesting_info_size = std::max(schema.max_definition_level + 1, max_depth);

    // if this column has lists, generate depth remapping
    std::map<int, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
    if (schema.max_repetition_level > 0) {
      generate_depth_remappings(depth_remapping, src_col_schema, *_metadata);
    }

    // fill in host-side nesting info
    int schema_idx  = src_col_schema;
    auto cur_schema = _metadata->get_schema(schema_idx);
    int cur_depth   = max_depth - 1;
    while (schema_idx > 0) {
      // stub columns (basically the inner field of a list scheme element) are not real columns.
      // we can ignore them for the purposes of output nesting info
      if (!cur_schema.is_stub()) {
        // initialize each page within the chunk
        for (int p_idx = 0; p_idx < chunks[idx].num_data_pages; p_idx++) {
          gpu::PageNestingInfo* pni =
            &page_nesting_info[nesting_info_index + (p_idx * per_page_nesting_info_size)];

          // if we have lists, set our start and end depth remappings
          if (schema.max_repetition_level > 0) {
            auto remap = depth_remapping.find(src_col_schema);
            CUDF_EXPECTS(remap != depth_remapping.end(),
                         "Could not find depth remapping for schema");
            std::vector<int> const& rep_depth_remap = (remap->second.first);
            std::vector<int> const& def_depth_remap = (remap->second.second);

            for (size_t m = 0; m < rep_depth_remap.size(); m++) {
              pni[m].start_depth = rep_depth_remap[m];
            }
            for (size_t m = 0; m < def_depth_remap.size(); m++) {
              pni[m].end_depth = def_depth_remap[m];
            }
          }

          // values indexed by output column index
          pni[cur_depth].max_def_level = cur_schema.max_definition_level;
          pni[cur_depth].max_rep_level = cur_schema.max_repetition_level;
          pni[cur_depth].size          = 0;
        }

        // move up the hierarchy
        cur_depth--;
      }

      // next schema
      schema_idx = cur_schema.parent_idx;
      cur_schema = _metadata->get_schema(schema_idx);
    }

    nesting_info_index += (per_page_nesting_info_size * chunks[idx].num_data_pages);
  }

  // copy nesting info to the device
  page_nesting_info.host_to_device(_stream);
}

/**
 * @copydoc cudf::io::detail::parquet::preprocess_columns
 */
void reader::impl::preprocess_columns(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                      hostdevice_vector<gpu::PageInfo>& pages,
                                      size_t min_row,
                                      size_t total_rows,
                                      bool has_lists)
{
  // TODO : we should be selectively preprocessing only columns that have
  // lists in them instead of doing them all if even one contains lists.

  // if there are no lists, simply allocate every allocate every output
  // column to be of size num_rows
  if (!has_lists) {
    std::function<void(std::vector<column_buffer>&)> create_columns =
      [&](std::vector<column_buffer>& cols) {
        for (size_t idx = 0; idx < cols.size(); idx++) {
          auto& col = cols[idx];
          col.create(total_rows, _stream, _mr);
          create_columns(col.children);
        }
      };
    create_columns(_output_columns);
  } else {
    // preprocess per-nesting level sizes by page
    gpu::PreprocessColumnData(
      pages, chunks, _input_columns, _output_columns, total_rows, min_row, _stream, _mr);
    _stream.synchronize();
  }
}

/**
 * @copydoc cudf::io::detail::parquet::decode_page_data
 */
void reader::impl::decode_page_data(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                    hostdevice_vector<gpu::PageInfo>& pages,
                                    hostdevice_vector<gpu::PageNestingInfo>& page_nesting,
                                    size_t min_row,
                                    size_t total_rows)
{
  auto is_dict_chunk = [](const gpu::ColumnChunkDesc& chunk) {
    return (chunk.data_type & 0x7) == BYTE_ARRAY && chunk.num_dict_pages > 0;
  };

  // Count the number of string dictionary entries
  // NOTE: Assumes first page in the chunk is always the dictionary page
  size_t total_str_dict_indexes = 0;
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) { total_str_dict_indexes += pages[page_count].num_input_values; }
    page_count += chunks[c].max_num_pages;
  }

  // Build index for string dictionaries since they can't be indexed
  // directly due to variable-sized elements
  auto str_dict_index = cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(
    total_str_dict_indexes, _stream);

  // TODO (dm): hd_vec should have begin and end iterator members
  size_t sum_max_depths =
    std::accumulate(chunks.host_ptr(),
                    chunks.host_ptr(chunks.size()),
                    0,
                    [&](size_t cursum, gpu::ColumnChunkDesc const& chunk) {
                      return cursum + _metadata->get_output_nesting_depth(chunk.src_col_schema);
                    });

  // In order to reduce the number of allocations of hostdevice_vector, we allocate a single vector
  // to store all per-chunk pointers to nested data/nullmask. `chunk_offsets[i]` will store the
  // offset into `chunk_nested_data`/`chunk_nested_valids` for the array of pointers for chunk `i`
  auto chunk_nested_valids = hostdevice_vector<uint32_t*>(sum_max_depths, _stream);
  auto chunk_nested_data   = hostdevice_vector<void*>(sum_max_depths, _stream);
  auto chunk_offsets       = std::vector<size_t>();

  // Update chunks with pointers to column data.
  for (size_t c = 0, page_count = 0, str_ofs = 0, chunk_off = 0; c < chunks.size(); c++) {
    input_column_info const& input_col = _input_columns[chunks[c].src_col_index];
    CUDF_EXPECTS(input_col.schema_idx == chunks[c].src_col_schema,
                 "Column/page schema index mismatch");

    if (is_dict_chunk(chunks[c])) {
      chunks[c].str_dict_index = str_dict_index.data() + str_ofs;
      str_ofs += pages[page_count].num_input_values;
    }

    size_t max_depth = _metadata->get_output_nesting_depth(chunks[c].src_col_schema);
    chunk_offsets.push_back(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_valids` to store an array of pointers
    // to validity data
    auto valids              = chunk_nested_valids.host_ptr(chunk_off);
    chunks[c].valid_map_base = chunk_nested_valids.device_ptr(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_data` to store an array of pointers to
    // out data
    auto data                  = chunk_nested_data.host_ptr(chunk_off);
    chunks[c].column_data_base = chunk_nested_data.device_ptr(chunk_off);

    chunk_off += max_depth;

    // fill in the arrays on the host.  there are some important considerations to
    // take into account here for nested columns.  specifically, with structs
    // there is sharing of output buffers between input columns.  consider this schema
    //
    //  required group field_id=1 name {
    //    required binary field_id=2 firstname (String);
    //    required binary field_id=3 middlename (String);
    //    required binary field_id=4 lastname (String);
    // }
    //
    // there are 3 input columns of data here (firstname, middlename, lastname), but
    // only 1 output column (name).  The structure of the output column buffers looks like
    // the schema itself
    //
    // struct      (name)
    //     string  (firstname)
    //     string  (middlename)
    //     string  (lastname)
    //
    // The struct column can contain validity information. the problem is, the decode
    // step for the input columns will all attempt to decode this validity information
    // because each one has it's own copy of the repetition/definition levels. but
    // since this is all happening in parallel it would mean multiple blocks would
    // be stomping all over the same memory randomly.  to work around this, we set
    // things up so that only 1 child of any given nesting level fills in the
    // data (offsets in the case of lists) or validity information for the higher
    // levels of the hierarchy that are shared.  In this case, it would mean we
    // would just choose firstname to be the one that decodes the validity for name.
    //
    // we do this by only handing out the pointers to the first child we come across.
    //
    auto* cols = &_output_columns;
    for (size_t idx = 0; idx < max_depth; idx++) {
      auto& out_buf = (*cols)[input_col.nesting[idx]];
      cols          = &out_buf.children;

      int owning_schema = out_buf.user_data & PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      if (owning_schema == 0 || owning_schema == input_col.schema_idx) {
        valids[idx] = out_buf.null_mask();
        data[idx]   = out_buf.data();
        out_buf.user_data |=
          static_cast<uint32_t>(input_col.schema_idx) & PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      } else {
        valids[idx] = nullptr;
        data[idx]   = nullptr;
      }
    }

    // column_data_base will always point to leaf data, even for nested types.
    page_count += chunks[c].max_num_pages;
  }

  chunks.host_to_device(_stream);
  chunk_nested_valids.host_to_device(_stream);
  chunk_nested_data.host_to_device(_stream);

  if (total_str_dict_indexes > 0) {
    gpu::BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size(), _stream);
  }

  gpu::DecodePageData(pages, chunks, total_rows, min_row, _stream);
  pages.device_to_host(_stream);
  page_nesting.device_to_host(_stream);
  _stream.synchronize();

  // for list columns, add the final offset to every offset buffer.
  // TODO : make this happen in more efficiently. Maybe use thrust::for_each
  // on each buffer.  Or potentially do it in PreprocessColumnData
  // Note : the reason we are doing this here instead of in the decode kernel is
  // that it is difficult/impossible for a given page to know that it is writing the very
  // last value that should then be followed by a terminator (because rows can span
  // page boundaries).
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    input_column_info const& input_col = _input_columns[idx];

    auto* cols = &_output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      if (out_buf.type.id() != type_id::LIST ||
          (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED)) {
        continue;
      }
      CUDF_EXPECTS(l_idx < input_col.nesting_depth() - 1, "Encountered a leaf list column");
      auto& child = (*cols)[input_col.nesting[l_idx + 1]];

      // the final offset for a list at level N is the size of it's child
      int offset = child.type.id() == type_id::LIST ? child.size - 1 : child.size;
      cudaMemcpyAsync(static_cast<int32_t*>(out_buf.data()) + (out_buf.size - 1),
                      &offset,
                      sizeof(offset),
                      cudaMemcpyHostToDevice,
                      _stream.value());
      out_buf.user_data |= PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED;
    }
  }

  // update null counts in the final column buffers
  for (size_t idx = 0; idx < pages.size(); idx++) {
    gpu::PageInfo* pi = &pages[idx];
    if (pi->flags & gpu::PAGEINFO_FLAGS_DICTIONARY) { continue; }
    gpu::ColumnChunkDesc* col          = &chunks[pi->chunk_idx];
    input_column_info const& input_col = _input_columns[col->src_col_index];

    int index                 = pi->nesting - page_nesting.device_ptr();
    gpu::PageNestingInfo* pni = &page_nesting[index];

    auto* cols = &_output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if I wasn't the one who wrote out the validity bits, skip it
      if (chunk_nested_valids.host_ptr(chunk_offsets[pi->chunk_idx])[l_idx] == nullptr) {
        continue;
      }
      out_buf.null_count() += pni[l_idx].null_count;
    }
  }

  _stream.synchronize();
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   parquet_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _stream(stream), _mr(mr), _sources(std::move(sources))
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_reader_metadata>(_sources);

  // Override output timestamp resolution if requested
  if (options.get_timestamp_type().id() != type_id::EMPTY) {
    _timestamp_type = options.get_timestamp_type();
  }

  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  // Select only columns required by the options
  std::tie(_input_columns, _output_columns, _output_column_schemas) =
    _metadata->select_columns(options.get_columns(),
                              options.is_enabled_use_pandas_metadata(),
                              _strings_to_categorical,
                              _timestamp_type.id());
}

table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       std::vector<std::vector<size_type>> const& row_group_list)
{
  // Select only row groups required
  const auto selected_row_groups =
    _metadata->select_row_groups(row_group_list, skip_rows, num_rows);

  table_metadata out_metadata;

  // output cudf columns as determined by the top level schema
  std::vector<std::unique_ptr<column>> out_columns;
  out_columns.reserve(_output_columns.size());

  if (selected_row_groups.size() != 0 && _input_columns.size() != 0) {
    // Descriptors for all the chunks that make up the selected columns
    const auto num_input_columns = _input_columns.size();
    const auto num_chunks        = selected_row_groups.size() * num_input_columns;
    hostdevice_vector<gpu::ColumnChunkDesc> chunks(0, num_chunks, _stream);

    // Association between each column chunk and its source
    std::vector<size_type> chunk_source_map(num_chunks);

    // Tracker for eventually deallocating compressed and uncompressed data
    std::vector<std::unique_ptr<datasource::buffer>> page_data(num_chunks);

    // Keep track of column chunk file offsets
    std::vector<size_t> column_chunk_offsets(num_chunks);

    // if there are lists present, we need to preprocess
    bool has_lists = false;

    // Initialize column chunk information
    size_t total_decompressed_size = 0;
    auto remaining_rows            = num_rows;
    std::vector<std::future<void>> read_rowgroup_tasks;
    for (const auto& rg : selected_row_groups) {
      const auto& row_group       = _metadata->get_row_group(rg.index, rg.source_index);
      auto const row_group_start  = rg.start_row;
      auto const row_group_source = rg.source_index;
      auto const row_group_rows   = std::min<int>(remaining_rows, row_group.num_rows);
      auto const io_chunk_idx     = chunks.size();

      // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
      for (size_t i = 0; i < num_input_columns; ++i) {
        auto col = _input_columns[i];
        // look up metadata
        auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
        auto& schema   = _metadata->get_schema(col.schema_idx);

        // this column contains repetition levels and will require a preprocess
        if (schema.max_repetition_level > 0) { has_lists = true; }

        // Spec requires each row group to contain exactly one chunk for every
        // column. If there are too many or too few, continue with best effort
        if (chunks.size() >= chunks.max_size()) {
          std::cerr << "Detected too many column chunks" << std::endl;
          continue;
        }

        auto [type_width, clock_rate, converted_type] =
          conversion_info(to_type_id(schema, _strings_to_categorical, _timestamp_type.id()),
                          _timestamp_type.id(),
                          schema.type,
                          schema.converted_type,
                          schema.type_length);

        column_chunk_offsets[chunks.size()] =
          (col_meta.dictionary_page_offset != 0)
            ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
            : col_meta.data_page_offset;

        chunks.insert(gpu::ColumnChunkDesc(col_meta.total_compressed_size,
                                           nullptr,
                                           col_meta.num_values,
                                           schema.type,
                                           type_width,
                                           row_group_start,
                                           row_group_rows,
                                           schema.max_definition_level,
                                           schema.max_repetition_level,
                                           _metadata->get_output_nesting_depth(col.schema_idx),
                                           required_bits(schema.max_definition_level),
                                           required_bits(schema.max_repetition_level),
                                           col_meta.codec,
                                           converted_type,
                                           schema.logical_type,
                                           schema.decimal_scale,
                                           clock_rate,
                                           i,
                                           col.schema_idx));

        // Map each column chunk to its column index and its source index
        chunk_source_map[chunks.size() - 1] = row_group_source;

        if (col_meta.codec != Compression::UNCOMPRESSED) {
          total_decompressed_size += col_meta.total_uncompressed_size;
        }
      }
      // Read compressed chunk data to device memory
      read_rowgroup_tasks.push_back(read_column_chunks(
        page_data, chunks, io_chunk_idx, chunks.size(), column_chunk_offsets, chunk_source_map));

      remaining_rows -= row_group.num_rows;
    }
    for (auto& task : read_rowgroup_tasks) {
      task.wait();
    }
    assert(remaining_rows <= 0);

    // Process dataset chunk pages into output columns
    const auto total_pages = count_page_headers(chunks);
    if (total_pages > 0) {
      hostdevice_vector<gpu::PageInfo> pages(total_pages, total_pages, _stream);
      rmm::device_buffer decomp_page_data;

      // decoding of column/page information
      decode_page_headers(chunks, pages);
      if (total_decompressed_size > 0) {
        decomp_page_data = decompress_page_data(chunks, pages);
        // Free compressed data
        for (size_t c = 0; c < chunks.size(); c++) {
          if (chunks[c].codec != parquet::Compression::UNCOMPRESSED) { page_data[c].reset(); }
        }
      }

      // build output column info
      // walk the schema, building out_buffers that mirror what our final cudf columns will look
      // like. important : there is not necessarily a 1:1 mapping between input columns and output
      // columns. For example, parquet does not explicitly store a ColumnChunkDesc for struct
      // columns. The "structiness" is simply implied by the schema.  For example, this schema:
      //  required group field_id=1 name {
      //    required binary field_id=2 firstname (String);
      //    required binary field_id=3 middlename (String);
      //    required binary field_id=4 lastname (String);
      // }
      // will only contain 3 columns of data (firstname, middlename, lastname).  But of course
      // "name" is a struct column that we want to return, so we have to make sure that we
      // create it ourselves.
      // std::vector<output_column_info> output_info = build_output_column_info();

      // nesting information (sizes, etc) stored -per page-
      // note : even for flat schemas, we allocate 1 level of "nesting" info
      hostdevice_vector<gpu::PageNestingInfo> page_nesting_info;
      allocate_nesting_info(chunks, pages, page_nesting_info);

      // - compute column sizes and allocate output buffers.
      //   important:
      //   for nested schemas, we have to do some further preprocessing to determine:
      //    - real column output sizes per level of nesting (in a flat schema, there's only 1 level
      //    of
      //      nesting and it's size is the row count)
      //
      // - for nested schemas, output buffer offset values per-page, per nesting-level for the
      // purposes of decoding.
      preprocess_columns(chunks, pages, skip_rows, num_rows, has_lists);

      // decoding of column data itself
      decode_page_data(chunks, pages, page_nesting_info, skip_rows, num_rows);

      // create the final output cudf columns
      for (size_t i = 0; i < _output_columns.size(); ++i) {
        column_name_info& col_name = out_metadata.schema_info.emplace_back("");
        out_columns.emplace_back(make_column(_output_columns[i], &col_name, _stream, _mr));
      }
    }
  }

  // Create empty columns as needed (this can happen if we've ended up with no actual data to read)
  for (size_t i = out_columns.size(); i < _output_columns.size(); ++i) {
    column_name_info& col_name = out_metadata.schema_info.emplace_back("");
    out_columns.emplace_back(io::detail::empty_like(_output_columns[i], &col_name, _stream, _mr));
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.resize(_output_columns.size());
  for (size_t i = 0; i < _output_column_schemas.size(); i++) {
    auto const& schema           = _metadata->get_schema(_output_column_schemas[i]);
    out_metadata.column_names[i] = schema.name;
  }

  // Return user metadata
  out_metadata.per_file_user_data = _metadata->get_key_value_metadata();
  out_metadata.user_data          = {out_metadata.per_file_user_data[0].begin(),
                            out_metadata.per_file_user_data[0].end()};

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
               parquet_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl(std::make_unique<impl>(std::move(sources), options, stream, mr))
{
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(parquet_reader_options const& options)
{
  return _impl->read(options.get_skip_rows(), options.get_num_rows(), options.get_row_groups());
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
