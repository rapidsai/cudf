/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <io/comp/gpuinflate.h>

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

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
/**
 * @brief Function that translates Parquet datatype to cuDF type enum
 */
type_id to_type_id(SchemaElement const &schema,
                   bool strings_to_categorical,
                   type_id timestamp_type_id)
{
  parquet::Type physical         = schema.type;
  parquet::ConvertedType logical = schema.converted_type;
  int32_t decimal_scale          = schema.decimal_scale;

  // Logical type used for actual data interpretation; the legacy converted type
  // is superceded by 'logical' type whenever available.
  switch (logical) {
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
    case parquet::TIMESTAMP_MICROS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MICROSECONDS;
    case parquet::TIMESTAMP_MILLIS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MILLISECONDS;
    case parquet::DECIMAL:
      if (decimal_scale != 0 || (physical != parquet::INT32 && physical != parquet::INT64)) {
        return type_id::FLOAT64;
      }
      break;

    // maps are just List<Struct<>>.
    case parquet::MAP:
    case parquet::LIST: return type_id::LIST;

    default: break;
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
 * @brief Function that translates cuDF time unit to Parquet clock frequency
 */
constexpr int32_t to_clockrate(type_id timestamp_type_id)
{
  switch (timestamp_type_id) {
    case type_id::DURATION_SECONDS: return 1;
    case type_id::DURATION_MILLISECONDS: return 1000;
    case type_id::DURATION_MICROSECONDS: return 1000000;
    case type_id::DURATION_NANOSECONDS: return 1000000000;
    case type_id::TIMESTAMP_SECONDS: return 1;
    case type_id::TIMESTAMP_MILLISECONDS: return 1000;
    case type_id::TIMESTAMP_MICROSECONDS: return 1000000;
    case type_id::TIMESTAMP_NANOSECONDS: return 1000000000;
    default: return 0;
  }
}

/**
 * @brief Function that returns the required the number of bits to store a value
 */
template <typename T = uint8_t>
T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

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
  if (converted_type == parquet::DECIMAL && column_type_id != type_id::FLOAT64) {
    converted_type = parquet::UNKNOWN;  // Not converting to float64
  }
  return std::make_tuple(type_width, clock_rate, converted_type);
}

}  // namespace

std::string name_from_path(const std::vector<std::string> &path_in_schema)
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
  explicit metadata(datasource *source)
  {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);

    const auto len           = source->size();
    const auto header_buffer = source->host_read(0, header_len);
    const auto header        = reinterpret_cast<const file_header_s *>(header_buffer->data());
    const auto ender_buffer  = source->host_read(len - ender_len, ender_len);
    const auto ender         = reinterpret_cast<const file_ender_s *>(ender_buffer->data());
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
    CUDF_EXPECTS(header->magic == PARQUET_MAGIC && ender->magic == PARQUET_MAGIC,
                 "Corrupted header or footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    const auto buffer = source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
    CompactProtocolReader cp(buffer->data(), ender->footer_len);
    CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
    CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
  }
};

class aggregate_metadata {
  std::vector<metadata> const per_file_metadata;
  std::map<std::string, std::string> const agg_keyval_map;
  size_type const num_rows;
  size_type const num_row_groups;
  /**
   * @brief Create a metadata object from each element in the source vector
   */
  auto metadatas_from_sources(std::vector<std::unique_ptr<datasource>> const &sources)
  {
    std::vector<metadata> metadatas;
    std::transform(
      sources.cbegin(), sources.cend(), std::back_inserter(metadatas), [](auto const &source) {
        return metadata(source.get());
      });
    return metadatas;
  }

  /**
   * @brief Merge the keyvalue maps from each per-file metadata object into a single map.
   */
  auto merge_keyval_metadata()
  {
    std::map<std::string, std::string> merged;
    // merge key/value maps TODO: warn/throw if there are mismatches?
    for (auto const &pfm : per_file_metadata) {
      for (auto const &kv : pfm.key_value_metadata) { merged[kv.key] = kv.value; }
    }
    return merged;
  }

  /**
   * @brief Sums up the number of rows of each source
   */
  size_type calc_num_rows() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto &sum, auto &pfm) {
        return sum + pfm.num_rows;
      });
  }

  /**
   * @brief Sums up the number of row groups of each source
   */
  size_type calc_num_row_groups() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto &sum, auto &pfm) {
        return sum + pfm.row_groups.size();
      });
  }

 public:
  aggregate_metadata(std::vector<std::unique_ptr<datasource>> const &sources)
    : per_file_metadata(metadatas_from_sources(sources)),
      agg_keyval_map(merge_keyval_metadata()),
      num_rows(calc_num_rows()),
      num_row_groups(calc_num_row_groups())
  {
    // Verify that the input files have matching numbers of columns
    size_type num_cols = -1;
    for (auto const &pfm : per_file_metadata) {
      if (pfm.row_groups.size() != 0) {
        if (num_cols == -1)
          num_cols = pfm.row_groups[0].columns.size();
        else
          CUDF_EXPECTS(num_cols == static_cast<size_type>(pfm.row_groups[0].columns.size()),
                       "All sources must have the same number of columns");
      }
    }
    // Verify that the input files have matching schemas
    for (auto const &pfm : per_file_metadata) {
      CUDF_EXPECTS(per_file_metadata[0].schema == pfm.schema,
                   "All sources must have the same schemas");
    }
  }

  auto const &get_row_group(size_type row_group_index, size_type src_idx) const
  {
    CUDF_EXPECTS(src_idx >= 0 && src_idx < static_cast<size_type>(per_file_metadata.size()),
                 "invalid source index");
    return per_file_metadata[src_idx].row_groups[row_group_index];
  }

  auto const &get_column_metadata(size_type row_group_index,
                                  size_type src_idx,
                                  int schema_idx) const
  {
    auto col = std::find_if(
      per_file_metadata[src_idx].row_groups[row_group_index].columns.begin(),
      per_file_metadata[src_idx].row_groups[row_group_index].columns.end(),
      [schema_idx](ColumnChunk const &col) { return col.schema_idx == schema_idx ? true : false; });
    CUDF_EXPECTS(col != std::end(per_file_metadata[src_idx].row_groups[row_group_index].columns),
                 "Found no metadata for schema index");
    return col->meta_data;
  }

  auto get_num_rows() const { return num_rows; }

  auto get_num_row_groups() const { return num_row_groups; }

  auto const &get_schema(int schema_idx) const { return per_file_metadata[0].schema[schema_idx]; }

  auto const &get_key_value_metadata() const { return agg_keyval_map; }

  /**
   * @brief Gets the concrete nesting depth of output cudf columns
   *
   * @param schema_index Schema index of the input column
   *
   * @return comma-separated index column names in quotes
   */
  inline int get_output_nesting_depth(int schema_index) const
  {
    auto &pfm = per_file_metadata[0];
    int depth = 0;

    // walk upwards, skipping repeated fields
    while (schema_index > 0) {
      if (!pfm.schema[schema_index].is_stub()) { depth++; }
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
  std::string get_pandas_index() const
  {
    auto it = agg_keyval_map.find("pandas");
    if (it != agg_keyval_map.end()) {
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
      if (std::regex_search(it->second, sm, index_columns_expr)) { return std::move(sm[1].str()); }
    }
    return "";
  }

  /**
   * @brief Extracts the column name(s) used for the row indexes in a dataframe
   *
   * @param names List of column names to load, where index column name(s) will be added
   */
  void add_pandas_index_names(std::vector<std::string> &names) const
  {
    auto str = get_pandas_index();
    if (str.length() != 0) {
      std::regex index_name_expr{R"(\"((?:\\.|[^\"])*)\")"};
      std::smatch sm;
      while (std::regex_search(str, sm, index_name_expr)) {
        if (sm.size() == 2) {  // 2 = whole match, first item
          if (std::find(names.begin(), names.end(), sm[1].str()) == names.end()) {
            std::regex esc_quote{R"(\\")"};
            names.emplace_back(std::move(std::regex_replace(sm[1].str(), esc_quote, R"(")")));
          }
        }
        str = sm.suffix();
      }
    }
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
  auto select_row_groups(std::vector<std::vector<size_type>> const &row_groups,
                         size_type &row_start,
                         size_type &row_count) const
  {
    if (!row_groups.empty()) {
      std::vector<row_group_info> selection;
      CUDF_EXPECTS(row_groups.size() == per_file_metadata.size(),
                   "Must specify row groups for each source");

      row_count = 0;
      for (size_t src_idx = 0; src_idx < row_groups.size(); ++src_idx) {
        for (auto const &rowgroup_idx : row_groups[src_idx]) {
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
   * @brief Build input and output column structures based on schema input. Recursive.
   *
   * @param[in,out] schema_idx Schema index to build information for. This value gets
   * incremented as the function recurses.
   * @param[out] input_columns Input column information (source data in the file)
   * @param[out] output_columns Output column structure (resulting cudf columns)
   * @param[in,out] nesting A stack keeping track of child column indices so we can
   * reproduce the linear list of output columns that correspond to an input column.
   * @param[in] strings_to_categorical Type conversion parameter
   * @param[in] timestamp_type_id Type conversion parameter
   *
   */
  void build_column_info(int &schema_idx,
                         std::vector<input_column_info> &input_columns,
                         std::vector<column_buffer> &output_columns,
                         std::deque<int> &nesting,
                         bool strings_to_categorical,
                         type_id timestamp_type_id) const
  {
    int start_schema_idx = schema_idx;
    auto const &schema   = get_schema(schema_idx);
    schema_idx++;

    // if I am a stub, continue on
    if (schema.is_stub()) {
      // is this legit?
      CUDF_EXPECTS(schema.num_children == 1, "Unexpected number of children for stub");
      build_column_info(schema_idx,
                        input_columns,
                        output_columns,
                        nesting,
                        strings_to_categorical,
                        timestamp_type_id);
      return;
    }

    // if we're at the root, this is a new output column
    int index = (int)output_columns.size();
    nesting.push_back(static_cast<int>(output_columns.size()));
    output_columns.emplace_back(
      data_type{to_type_id(schema, strings_to_categorical, timestamp_type_id)},
      schema.repetition_type == OPTIONAL ? true : false);
    column_buffer &output_col = output_columns.back();
    output_col.name           = schema.name;

    // build each child
    for (int idx = 0; idx < schema.num_children; idx++) {
      build_column_info(schema_idx,
                        input_columns,
                        output_col.children,
                        nesting,
                        strings_to_categorical,
                        timestamp_type_id);
    }

    // if I have no children, we're at a leaf and I'm an input column (that is, one with actual
    // data stored) so add me to the list.
    if (schema.num_children == 0) {
      input_columns.emplace_back(input_column_info{start_schema_idx, schema.name});
      input_column_info &input_col = input_columns.back();
      std::copy(nesting.begin(), nesting.end(), std::back_inserter(input_col.nesting));
    }

    nesting.pop_back();
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of column names to select
   * @param include_index Whether to always include the PANDAS index column(s)
   * @param strings_to_categorical Type conversion parameter
   * @param timestamp_type_id Type conversion parameter
   *
   * @return input column information, output column information, list of output column schema
   * indices
   */
  auto select_columns(std::vector<std::string> const &use_names,
                      bool include_index,
                      bool strings_to_categorical,
                      type_id timestamp_type_id) const
  {
    auto const &pfm = per_file_metadata[0];

    // determine the list of output columns
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
    std::vector<int> output_column_schemas;
    if (use_names.empty()) {
      // walk the schema and choose all top level columns
      for (size_t schema_idx = 1; schema_idx < pfm.schema.size(); schema_idx++) {
        auto const &schema = pfm.schema[schema_idx];
        if (schema.parent_idx == 0) { output_column_schemas.push_back(schema_idx); }
      }
    } else {
      // Load subset of columns; include PANDAS index unless excluded
      std::vector<std::string> local_use_names = use_names;
      if (include_index) { add_pandas_index_names(local_use_names); }
      for (const auto &use_name : local_use_names) {
        for (size_t schema_idx = 1; schema_idx < pfm.schema.size(); schema_idx++) {
          auto const &schema = pfm.schema[schema_idx];
          if (use_name == schema.name) { output_column_schemas.push_back(schema_idx); }
        }
      }
    }

    // construct input and output output column info
    std::vector<column_buffer> output_columns;
    output_columns.reserve(output_column_schemas.size());
    std::vector<input_column_info> input_columns;
    std::deque<int> nesting;
    for (size_t idx = 0; idx < output_column_schemas.size(); idx++) {
      int schema_index = output_column_schemas[idx];
      build_column_info(schema_index,
                        input_columns,
                        output_columns,
                        nesting,
                        strings_to_categorical,
                        timestamp_type_id);
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
 *
 */
void generate_depth_remappings(std::map<int, std::pair<std::vector<int>, std::vector<int>>> &remap,
                               int src_col_schema,
                               aggregate_metadata const &md)
{
  // already generated for this level
  if (remap.find(src_col_schema) != remap.end()) { return; }
  auto schema   = md.get_schema(src_col_schema);
  int max_depth = md.get_output_nesting_depth(src_col_schema);

  CUDF_EXPECTS(remap.find(src_col_schema) == remap.end(),
               "Attempting to remap a schema more than once");
  auto inserted =
    remap.insert(std::pair<int, std::pair<std::vector<int>, std::vector<int>>>{src_col_schema, {}});
  auto &depth_remap = inserted.first->second;

  std::vector<int> &rep_depth_remap = (depth_remap.first);
  rep_depth_remap.resize(schema.max_repetition_level + 1);
  std::vector<int> &def_depth_remap = (depth_remap.second);
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
void reader::impl::read_column_chunks(
  std::vector<rmm::device_buffer> &page_data,
  hostdevice_vector<gpu::ColumnChunkDesc> &chunks,  // TODO const?
  size_t begin_chunk,
  size_t end_chunk,
  const std::vector<size_t> &column_chunk_offsets,
  std::vector<size_type> const &chunk_source_map,
  cudaStream_t stream)
{
  // Transfer chunk data, coalescing adjacent chunks
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
      auto buffer         = _sources[chunk_source_map[chunk]]->host_read(io_offset, io_size);
      page_data[chunk]    = rmm::device_buffer(buffer->data(), buffer->size(), stream);
      uint8_t *d_compdata = static_cast<uint8_t *>(page_data[chunk].data());
      do {
        chunks[chunk].compressed_data = d_compdata;
        d_compdata += chunks[chunk].compressed_size;
      } while (++chunk != next_chunk);
    } else {
      chunk = next_chunk;
    }
  }
}

/**
 * @copydoc cudf::io::detail::parquet::count_page_headers
 */
size_t reader::impl::count_page_headers(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                        cudaStream_t stream)
{
  size_t total_pages = 0;

  chunks.host_to_device(stream);
  CUDA_TRY(gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream));
  chunks.device_to_host(stream, true);

  for (size_t c = 0; c < chunks.size(); c++) {
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

/**
 * @copydoc cudf::io::detail::parquet::decode_page_headers
 */
void reader::impl::decode_page_headers(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                       hostdevice_vector<gpu::PageInfo> &pages,
                                       cudaStream_t stream)
{
  // IMPORTANT : if you change how pages are stored within a chunk (dist pages, then data pages),
  // please update preprocess_nested_columns to reflect this.
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    chunks[c].max_num_pages = chunks[c].num_data_pages + chunks[c].num_dict_pages;
    chunks[c].page_info     = pages.device_ptr(page_count);
    page_count += chunks[c].max_num_pages;
  }

  chunks.host_to_device(stream);
  CUDA_TRY(gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream));
  pages.device_to_host(stream, true);
}

/**
 * @copydoc cudf::io::detail::parquet::decompress_page_data
 */
rmm::device_buffer reader::impl::decompress_page_data(
  hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
  hostdevice_vector<gpu::PageInfo> &pages,
  cudaStream_t stream)
{
  auto for_each_codec_page = [&](parquet::Compression codec, const std::function<void(size_t)> &f) {
    for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
      const auto page_stride = chunks[c].max_num_pages;
      if (chunks[c].codec == codec) {
        for (int k = 0; k < page_stride; k++) { f(page_count + k); }
      }
      page_count += page_stride;
    }
  };

  // Brotli scratch memory for decompressing
  rmm::device_vector<uint8_t> debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;
  std::array<std::pair<parquet::Compression, size_t>, 3> codecs{std::make_pair(parquet::GZIP, 0),
                                                                std::make_pair(parquet::SNAPPY, 0),
                                                                std::make_pair(parquet::BROTLI, 0)};

  for (auto &codec : codecs) {
    for_each_codec_page(codec.first, [&](size_t page) {
      total_decomp_size += pages[page].uncompressed_page_size;
      codec.second++;
      num_comp_pages++;
    });
    if (codec.first == parquet::BROTLI && codec.second > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.second));
    }
  }

  // Dispatch batches of pages to decompress for each codec
  rmm::device_buffer decomp_pages(total_decomp_size, stream);
  hostdevice_vector<gpu_inflate_input_s> inflate_in(0, num_comp_pages, stream);
  hostdevice_vector<gpu_inflate_status_s> inflate_out(0, num_comp_pages, stream);

  size_t decomp_offset = 0;
  int32_t argc         = 0;
  for (const auto &codec : codecs) {
    if (codec.second > 0) {
      int32_t start_pos = argc;

      for_each_codec_page(codec.first, [&](size_t page) {
        auto dst_base              = static_cast<uint8_t *>(decomp_pages.data());
        inflate_in[argc].srcDevice = pages[page].page_data;
        inflate_in[argc].srcSize   = pages[page].compressed_page_size;
        inflate_in[argc].dstDevice = dst_base + decomp_offset;
        inflate_in[argc].dstSize   = pages[page].uncompressed_page_size;

        inflate_out[argc].bytes_written = 0;
        inflate_out[argc].status        = static_cast<uint32_t>(-1000);
        inflate_out[argc].reserved      = 0;

        pages[page].page_data = static_cast<uint8_t *>(inflate_in[argc].dstDevice);
        decomp_offset += inflate_in[argc].dstSize;
        argc++;
      });

      CUDA_TRY(cudaMemcpyAsync(inflate_in.device_ptr(start_pos),
                               inflate_in.host_ptr(start_pos),
                               sizeof(decltype(inflate_in)::value_type) * (argc - start_pos),
                               cudaMemcpyHostToDevice,
                               stream));
      CUDA_TRY(cudaMemcpyAsync(inflate_out.device_ptr(start_pos),
                               inflate_out.host_ptr(start_pos),
                               sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
                               cudaMemcpyHostToDevice,
                               stream));
      switch (codec.first) {
        case parquet::GZIP:
          CUDA_TRY(gpuinflate(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos,
                              1,
                              stream))
          break;
        case parquet::SNAPPY:
          CUDA_TRY(gpu_unsnap(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos,
                              stream));
          break;
        case parquet::BROTLI:
          CUDA_TRY(gpu_debrotli(inflate_in.device_ptr(start_pos),
                                inflate_out.device_ptr(start_pos),
                                debrotli_scratch.data().get(),
                                debrotli_scratch.size(),
                                argc - start_pos,
                                stream));
          break;
        default: CUDF_EXPECTS(false, "Unexpected decompression dispatch"); break;
      }
      CUDA_TRY(cudaMemcpyAsync(inflate_out.host_ptr(start_pos),
                               inflate_out.device_ptr(start_pos),
                               sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
                               cudaMemcpyDeviceToHost,
                               stream));
    }
  }
  CUDA_TRY(cudaStreamSynchronize(stream));

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  CUDA_TRY(cudaMemcpyAsync(
    pages.device_ptr(), pages.host_ptr(), pages.memory_size(), cudaMemcpyHostToDevice, stream));

  return decomp_pages;
}

/**
 * @copydoc cudf::io::detail::parquet::allocate_nesting_info
 */
void reader::impl::allocate_nesting_info(hostdevice_vector<gpu::ColumnChunkDesc> const &chunks,
                                         hostdevice_vector<gpu::PageInfo> &pages,
                                         hostdevice_vector<gpu::PageNestingInfo> &page_nesting_info,
                                         cudaStream_t stream)
{
  // compute total # of page_nesting infos needed and allocate space. doing this in one
  // buffer to keep it to a single gpu allocation
  size_t const total_page_nesting_infos = std::accumulate(
    chunks.host_ptr(), chunks.host_ptr() + chunks.size(), 0, [&](int total, auto &chunk) {
      // the schema of the input column
      auto const &schema                    = _metadata->get_schema(chunk.src_col_schema);
      auto const per_page_nesting_info_size = max(
        schema.max_definition_level + 1, _metadata->get_output_nesting_depth(chunk.src_col_schema));
      return total + (per_page_nesting_info_size * chunk.num_data_pages);
    });

  page_nesting_info = hostdevice_vector<gpu::PageNestingInfo>{total_page_nesting_infos, stream};

  // retrieve from the gpu so we can update
  pages.device_to_host(stream, true);

  // update pointers in the PageInfos
  int target_page_index = 0;
  int src_info_index    = 0;
  for (size_t idx = 0; idx < chunks.size(); idx++) {
    int src_col_schema = chunks[idx].src_col_schema;
    auto &schema       = _metadata->get_schema(src_col_schema);
    auto const per_page_nesting_info_size =
      max(schema.max_definition_level + 1, _metadata->get_output_nesting_depth(src_col_schema));

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
  pages.host_to_device(stream);

  // fill in
  int nesting_info_index = 0;
  std::map<int, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
  for (size_t idx = 0; idx < chunks.size(); idx++) {
    int src_col_schema = chunks[idx].src_col_schema;

    // schema of the input column
    auto &schema = _metadata->get_schema(src_col_schema);
    // real depth of the output cudf column hierarchy (1 == no nesting, 2 == 1 level, etc)
    int max_depth = _metadata->get_output_nesting_depth(src_col_schema);

    // # of nesting infos stored per page for this column
    auto const per_page_nesting_info_size = max(schema.max_definition_level + 1, max_depth);

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
          gpu::PageNestingInfo *pni =
            &page_nesting_info[nesting_info_index + (p_idx * per_page_nesting_info_size)];

          // if we have lists, set our start and end depth remappings
          if (schema.max_repetition_level > 0) {
            auto remap = depth_remapping.find(src_col_schema);
            CUDF_EXPECTS(remap != depth_remapping.end(),
                         "Could not find depth remapping for schema");
            std::vector<int> const &rep_depth_remap = (remap->second.first);
            std::vector<int> const &def_depth_remap = (remap->second.second);

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
  page_nesting_info.host_to_device(stream);
}

/**
 * @copydoc cudf::io::detail::parquet::preprocess_columns
 */
void reader::impl::preprocess_columns(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                      hostdevice_vector<gpu::PageInfo> &pages,
                                      size_t min_row,
                                      size_t total_rows,
                                      bool has_lists,
                                      cudaStream_t stream)
{
  // TODO : we should be selectively preprocessing only columns that have
  // lists in them instead of doing them all if even one contains lists.

  // if there are no lists, simply allocate every allocate every output
  // column to be of size num_rows
  if (!has_lists) {
    std::function<void(std::vector<column_buffer> &)> create_columns =
      [&](std::vector<column_buffer> &cols) {
        for (size_t idx = 0; idx < cols.size(); idx++) {
          auto &col = cols[idx];
          col.create(total_rows, stream, _mr);
          create_columns(col.children);
        }
      };
    create_columns(_output_columns);
  } else {
    // preprocess per-nesting level sizes by page
    CUDA_TRY(gpu::PreprocessColumnData(
      pages, chunks, _input_columns, _output_columns, total_rows, min_row, stream, _mr));
    CUDA_TRY(cudaStreamSynchronize(stream));
  }
}

/**
 * @copydoc cudf::io::detail::parquet::decode_page_data
 */
void reader::impl::decode_page_data(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                    hostdevice_vector<gpu::PageInfo> &pages,
                                    hostdevice_vector<gpu::PageNestingInfo> &page_nesting,
                                    size_t min_row,
                                    size_t total_rows,
                                    cudaStream_t stream)
{
  auto is_dict_chunk = [](const gpu::ColumnChunkDesc &chunk) {
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
  rmm::device_vector<gpu::nvstrdesc_s> str_dict_index;
  if (total_str_dict_indexes > 0) { str_dict_index.resize(total_str_dict_indexes); }

  std::vector<hostdevice_vector<uint32_t *>> chunk_nested_valids;
  std::vector<hostdevice_vector<void *>> chunk_nested_data;

  // Update chunks with pointers to column data.
  for (size_t c = 0, page_count = 0, str_ofs = 0; c < chunks.size(); c++) {
    input_column_info const &input_col = _input_columns[chunks[c].src_col_index];
    CUDF_EXPECTS(input_col.schema_idx == chunks[c].src_col_schema,
                 "Column/page schema index mismatch");

    if (is_dict_chunk(chunks[c])) {
      chunks[c].str_dict_index = str_dict_index.data().get() + str_ofs;
      str_ofs += pages[page_count].num_input_values;
    }

    size_t max_depth = _metadata->get_output_nesting_depth(chunks[c].src_col_schema);

    // allocate (gpu) an array of pointers to validity data of size : nesting depth
    chunk_nested_valids.emplace_back(hostdevice_vector<uint32_t *>{max_depth});
    hostdevice_vector<uint32_t *> &valids = chunk_nested_valids.back();
    chunks[c].valid_map_base              = valids.device_ptr();

    // allocate (gpu) an array of pointers to out data of size : nesting depth
    chunk_nested_data.emplace_back(hostdevice_vector<void *>{max_depth});
    hostdevice_vector<void *> &data = chunk_nested_data.back();
    chunks[c].column_data_base      = data.device_ptr();

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
    auto *cols = &_output_columns;
    for (size_t idx = 0; idx < max_depth; idx++) {
      auto &out_buf = (*cols)[input_col.nesting[idx]];
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

    // copy to the gpu
    valids.host_to_device(stream);
    data.host_to_device(stream);

    // column_data_base will always point to leaf data, even for nested types.
    page_count += chunks[c].max_num_pages;
  }

  chunks.host_to_device(stream);

  if (total_str_dict_indexes > 0) {
    CUDA_TRY(gpu::BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size(), stream));
  }

  CUDA_TRY(gpu::DecodePageData(pages, chunks, total_rows, min_row, stream));
  pages.device_to_host(stream);
  page_nesting.device_to_host(stream);
  cudaStreamSynchronize(stream);

  // for list columns, add the final offset to every offset buffer.
  // TODO : make this happen in more efficiently. Maybe use thrust::for_each
  // on each buffer.  Or potentially do it in PreprocessColumnData
  // Note : the reason we are doing this here instead of in the decode kernel is
  // that it is difficult/impossible for a given page to know that it is writing the very
  // last value that should then be followed by a terminator (because rows can span
  // page boundaries).
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    input_column_info const &input_col = _input_columns[idx];

    auto *cols = &_output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto &out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      if (out_buf.type.id() != type_id::LIST ||
          (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED)) {
        continue;
      }
      CUDF_EXPECTS(l_idx < input_col.nesting_depth() - 1, "Encountered a leaf list column");
      auto &child = (*cols)[input_col.nesting[l_idx + 1]];

      // the final offset for a list at level N is the size of it's child
      int offset = child.type.id() == type_id::LIST ? child.size - 1 : child.size;
      cudaMemcpyAsync(static_cast<int32_t *>(out_buf.data()) + (out_buf.size - 1),
                      &offset,
                      sizeof(offset),
                      cudaMemcpyHostToDevice,
                      stream);
      out_buf.user_data |= PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED;
    }
  }

  // update null counts in the final column buffers
  for (size_t idx = 0; idx < pages.size(); idx++) {
    gpu::PageInfo *pi = &pages[idx];
    if (pi->flags & gpu::PAGEINFO_FLAGS_DICTIONARY) { continue; }
    gpu::ColumnChunkDesc *col          = &chunks[pi->chunk_idx];
    input_column_info const &input_col = _input_columns[col->src_col_index];

    int index                 = pi->nesting - page_nesting.device_ptr();
    gpu::PageNestingInfo *pni = &page_nesting[index];

    auto *cols = &_output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto &out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if I wasn't the one who wrote out the validity bits, skip it
      if (chunk_nested_valids[pi->chunk_idx][l_idx] == nullptr) { continue; }
      out_buf.null_count() += pni[l_idx].value_count - pni[l_idx].valid_count;
    }
  }

  cudaStreamSynchronize(stream);
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>> &&sources,
                   parquet_reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : _sources(std::move(sources)), _mr(mr)
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_metadata>(_sources);

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
                                       std::vector<std::vector<size_type>> const &row_group_list,
                                       cudaStream_t stream)
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
    hostdevice_vector<gpu::ColumnChunkDesc> chunks(0, num_chunks, stream);

    // Association between each column chunk and its source
    std::vector<size_type> chunk_source_map(num_chunks);

    // Tracker for eventually deallocating compressed and uncompressed data
    std::vector<rmm::device_buffer> page_data(num_chunks);

    // Keep track of column chunk file offsets
    std::vector<size_t> column_chunk_offsets(num_chunks);

    // if there are lists present, we need to preprocess
    bool has_lists = false;

    // Initialize column chunk information
    size_t total_decompressed_size = 0;
    auto remaining_rows            = num_rows;
    for (const auto &rg : selected_row_groups) {
      const auto &row_group       = _metadata->get_row_group(rg.index, rg.source_index);
      auto const row_group_start  = rg.start_row;
      auto const row_group_source = rg.source_index;
      auto const row_group_rows   = std::min<int>(remaining_rows, row_group.num_rows);
      auto const io_chunk_idx     = chunks.size();

      // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
      for (size_t i = 0; i < num_input_columns; ++i) {
        auto col = _input_columns[i];
        // look up metadata
        auto &col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
        auto &schema   = _metadata->get_schema(col.schema_idx);

        // this column contains repetition levels and will require a preprocess
        if (schema.max_repetition_level > 0) { has_lists = true; }

        // Spec requires each row group to contain exactly one chunk for every
        // column. If there are too many or too few, continue with best effort
        if (chunks.size() >= chunks.max_size()) {
          std::cerr << "Detected too many column chunks" << std::endl;
          continue;
        }

        int32_t type_width;
        int32_t clock_rate;
        int8_t converted_type;

        std::tie(type_width, clock_rate, converted_type) =
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
      read_column_chunks(page_data,
                         chunks,
                         io_chunk_idx,
                         chunks.size(),
                         column_chunk_offsets,
                         chunk_source_map,
                         stream);

      remaining_rows -= row_group.num_rows;
    }
    assert(remaining_rows <= 0);

    // Process dataset chunk pages into output columns
    const auto total_pages = count_page_headers(chunks, stream);
    if (total_pages > 0) {
      hostdevice_vector<gpu::PageInfo> pages(total_pages, total_pages, stream);
      rmm::device_buffer decomp_page_data;

      // decoding of column/page information
      decode_page_headers(chunks, pages, stream);
      if (total_decompressed_size > 0) {
        decomp_page_data = decompress_page_data(chunks, pages, stream);
        // Free compressed data
        for (size_t c = 0; c < chunks.size(); c++) {
          if (chunks[c].codec != parquet::Compression::UNCOMPRESSED && page_data[c].size() != 0) {
            page_data[c].resize(0);
            page_data[c].shrink_to_fit();
          }
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
      allocate_nesting_info(chunks, pages, page_nesting_info, stream);

      // - compute column sizes and allocate output buffers.
      //   important:
      //   for nested schemas, we have to do some further preprocessing to determine:
      //    - real column output sizes per level of nesting (in a flat schema, there's only 1 level
      //    of
      //      nesting and it's size is the row count)
      //
      // - for nested schemas, output buffer offset values per-page, per nesting-level for the
      // purposes of decoding.
      preprocess_columns(chunks, pages, skip_rows, num_rows, has_lists, stream);

      // decoding of column data itself
      decode_page_data(chunks, pages, page_nesting_info, skip_rows, num_rows, stream);

      // create the final output cudf columns
      for (size_t i = 0; i < _output_columns.size(); ++i) {
        out_metadata.schema_info.push_back(column_name_info{""});
        out_columns.emplace_back(
          make_column(_output_columns[i], stream, _mr, &out_metadata.schema_info.back()));
      }
    }
  }

  // Create empty columns as needed (this can happen if we've ended up with no actual data to read)
  for (size_t i = out_columns.size(); i < _output_columns.size(); ++i) {
    out_metadata.schema_info.push_back(column_name_info{""});
    out_columns.emplace_back(make_empty_column(_output_columns[i].type));
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.resize(_output_columns.size());
  for (size_t i = 0; i < _output_column_schemas.size(); i++) {
    auto const &schema           = _metadata->get_schema(_output_column_schemas[i]);
    out_metadata.column_names[i] = schema.name;
  }

  // Return user metadata
  out_metadata.user_data = _metadata->get_key_value_metadata();

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::vector<std::string> const &filepaths,
               parquet_reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(datasource::create(filepaths), options, mr))
{
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>> &&sources,
               parquet_reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(std::move(sources), options, mr))
{
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(parquet_reader_options const &options, cudaStream_t stream)
{
  return _impl->read(
    options.get_skip_rows(), options.get_num_rows(), options.get_row_groups(), stream);
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
