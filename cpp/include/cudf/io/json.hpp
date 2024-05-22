/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#pragma once

#include "types.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

class json_reader_options_builder;

/**
 * @brief Allows specifying the target types for nested JSON data via json_reader_options'
 * `set_dtypes` method.
 */
struct schema_element {
  /**
   * @brief The type that this column should be converted to
   */
  data_type type;

  /**
   * @brief Allows specifying this column's child columns target type
   */
  std::map<std::string, schema_element> child_types;
};

/**
 * @brief Control the error recovery behavior of the json parser
 */
enum class json_recovery_mode_t {
  FAIL,              ///< Does not recover from an error when encountering an invalid format
  RECOVER_WITH_NULL  ///< Recovers from an error, replacing invalid records with null
};

/**
 * @brief Input arguments to the `read_json` interface.
 *
 * Available parameters are closely patterned after PANDAS' `read_json` API.
 * Not all parameters are supported. If the matching PANDAS' parameter
 * has a default value of `None`, then a default value of `-1` or `0` may be
 * used as the equivalent.
 *
 * Parameters in PANDAS that are unavailable or in cudf:
 *
 * | Name                 | Description                                      |
 * | -------------------- | ------------------------------------------------ |
 * | `orient`             | currently fixed-format                           |
 * | `typ`                | data is always returned as a cudf::table         |
 * | `convert_axes`       | use column functions for axes operations instead |
 * | `convert_dates`      | dates are detected automatically                 |
 * | `keep_default_dates` | dates are detected automatically                 |
 * | `numpy`              | data is always returned as a cudf::table         |
 * | `precise_float`      | there is only one converter                      |
 * | `date_unit`          | only millisecond units are supported             |
 * | `encoding`           | only ASCII-encoded data is supported             |
 * | `chunksize`          | use `byte_range_xxx` for chunking instead        |
 */
class json_reader_options {
  source_info _source;

  // Data types of the column; empty to infer dtypes
  std::variant<std::vector<data_type>,
               std::map<std::string, data_type>,
               std::map<std::string, schema_element>>
    _dtypes;
  // Specify the compression format of the source or infer from file extension
  compression_type _compression = compression_type::AUTO;

  // Read the file as a json object per line
  bool _lines = false;
  // Parse mixed types as a string column
  bool _mixed_types_as_string = false;
  // Delimiter separating records in JSON lines
  char _delimiter = '\n';
  // Prune columns on read, selected based on the _dtypes option
  bool _prune_columns = false;

  // Bytes to skip from the start
  size_t _byte_range_offset = 0;
  // Bytes to read; always reads complete rows
  size_t _byte_range_size = 0;

  // Whether to parse dates as DD/MM versus MM/DD
  bool _dayfirst = false;

  // Whether to use the legacy reader
  bool _legacy = false;

  // Whether to keep the quote characters of string values
  bool _keep_quotes = false;

  // Normalize single quotes
  bool _normalize_single_quotes = false;

  // Normalize unquoted spaces and tabs
  bool _normalize_whitespace = false;

  // Whether to recover after an invalid JSON line
  json_recovery_mode_t _recovery_mode = json_recovery_mode_t::FAIL;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read parquet file
   */
  explicit json_reader_options(source_info src) : _source{std::move(src)} {}

  friend json_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  json_reader_options() = default;

  /**
   * @brief create json_reader_options_builder which will build json_reader_options.
   *
   * @param src source information used to read json file
   * @returns builder to build the options
   */
  static json_reader_options_builder builder(source_info src);

  /**
   * @brief Returns source info.
   *
   * @returns Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns data types of the columns.
   *
   * @returns Data types of the columns
   */
  std::variant<std::vector<data_type>,
               std::map<std::string, data_type>,
               std::map<std::string, schema_element>> const&
  get_dtypes() const
  {
    return _dtypes;
  }

  /**
   * @brief Returns compression format of the source.
   *
   * @return Compression format of the source
   */
  compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns number of bytes to skip from source start.
   *
   * @return Number of bytes to skip from source start
   */
  size_t get_byte_range_offset() const { return _byte_range_offset; }

  /**
   * @brief Returns number of bytes to read.
   *
   * @return Number of bytes to read
   */
  size_t get_byte_range_size() const { return _byte_range_size; }

  /**
   * @brief Returns number of bytes to read with padding.
   *
   * @return Number of bytes to read with padding
   */
  size_t get_byte_range_size_with_padding() const
  {
    if (_byte_range_size == 0) {
      return 0;
    } else {
      return _byte_range_size + get_byte_range_padding();
    }
  }

  /**
   * @brief Returns number of bytes to pad when reading.
   *
   * @return Number of bytes to pad
   */
  size_t get_byte_range_padding() const
  {
    auto const num_columns = std::visit([](auto const& dtypes) { return dtypes.size(); }, _dtypes);

    auto const max_row_bytes = 16 * 1024;  // 16KB
    auto const column_bytes  = 64;
    auto const base_padding  = 1024;  // 1KB

    if (num_columns == 0) {
      // Use flat size if the number of columns is not known
      return max_row_bytes;
    }

    // Expand the size based on the number of columns, if available
    return base_padding + num_columns * column_bytes;
  }

  /**
   * @brief Returns delimiter separating records in JSON lines
   *
   * @return Delimiter separating records in JSON lines
   */
  char get_delimiter() const { return _delimiter; }

  /**
   * @brief Whether to read the file as a json object per line.
   *
   * @return `true` if reading the file as a json object per line
   */
  bool is_enabled_lines() const { return _lines; }

  /**
   * @brief Whether to parse mixed types as a string column.
   *
   * @return `true` if mixed types are parsed as a string column
   */
  bool is_enabled_mixed_types_as_string() const { return _mixed_types_as_string; }

  /**
   * @brief Whether to prune columns on read, selected based on the @ref set_dtypes option.
   *
   * When set as true, if the reader options include @ref set_dtypes, then
   * the reader will only return those columns which are mentioned in @ref set_dtypes.
   * If false, then all columns are returned, independent of the @ref set_dtypes setting.
   *
   * @return True if column pruning is enabled
   */
  bool is_enabled_prune_columns() const { return _prune_columns; }

  /**
   * @brief Whether to parse dates as DD/MM versus MM/DD.
   *
   * @returns true if dates are parsed as DD/MM, false if MM/DD
   */
  bool is_enabled_dayfirst() const { return _dayfirst; }

  /**
   * @brief Whether the legacy reader should be used.
   *
   * @deprecated Since 24.06
   *
   * @returns true if the legacy reader will be used, false otherwise
   */
  [[deprecated]] bool is_enabled_legacy() const { return _legacy; }

  /**
   * @brief Whether the reader should keep quotes of string values.
   *
   * @returns true if the reader should keep quotes, false otherwise
   */
  bool is_enabled_keep_quotes() const { return _keep_quotes; }

  /**
   * @brief Whether the reader should normalize single quotes around strings
   *
   * @returns true if the reader should normalize single quotes, false otherwise
   */
  bool is_enabled_normalize_single_quotes() const { return _normalize_single_quotes; }

  /**
   * @brief Whether the reader should normalize unquoted whitespace characters
   *
   * @returns true if the reader should normalize whitespace, false otherwise
   */
  bool is_enabled_normalize_whitespace() const { return _normalize_whitespace; }

  /**
   * @brief Queries the JSON reader's behavior on invalid JSON lines.
   *
   * @returns An enum that specifies the JSON reader's behavior on invalid JSON lines.
   */
  json_recovery_mode_t recovery_mode() const { return _recovery_mode; }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector of dtypes
   */
  void set_dtypes(std::vector<data_type> types) { _dtypes = std::move(types); }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector dtypes in string format
   */
  void set_dtypes(std::map<std::string, data_type> types) { _dtypes = std::move(types); }

  /**
   * @brief Set data types for a potentially nested column hierarchy.
   *
   * @param types Map of column names to schema_element to support arbitrary nesting of data types
   */
  void set_dtypes(std::map<std::string, schema_element> types) { _dtypes = std::move(types); }

  /**
   * @brief Set the compression type.
   *
   * @param comp_type The compression type used
   */
  void set_compression(compression_type comp_type) { _compression = comp_type; }

  /**
   * @brief Set number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset
   */
  void set_byte_range_offset(size_type offset) { _byte_range_offset = offset; }

  /**
   * @brief Set number of bytes to read.
   *
   * @param size Number of bytes to read
   */
  void set_byte_range_size(size_type size) { _byte_range_size = size; }

  /**
   * @brief Set delimiter separating records in JSON lines
   *
   * @param delimiter Delimiter separating records in JSON lines
   */
  void set_delimiter(char delimiter)
  {
    switch (delimiter) {
      case '{':
      case '[':
      case '}':
      case ']':
      case ',':
      case ':':
      case '"':
      case '\'':
      case '\\':
      case ' ':
      case '\t':
      case '\r': CUDF_FAIL("Unsupported delimiter character.", std::invalid_argument); break;
    }
    _delimiter = delimiter;
  }

  /**
   * @brief Set whether to read the file as a json object per line.
   *
   * @param val Boolean value to enable/disable the option to read each line as a json object
   */
  void enable_lines(bool val) { _lines = val; }

  /**
   * @brief Set whether to parse mixed types as a string column.
   * Also enables forcing to read a struct as string column using schema.
   *
   * @param val Boolean value to enable/disable parsing mixed types as a string column
   */
  void enable_mixed_types_as_string(bool val) { _mixed_types_as_string = val; }

  /**
   * @brief Set whether to prune columns on read, selected based on the @ref set_dtypes option.
   *
   * When set as true, if the reader options include @ref set_dtypes, then
   * the reader will only return those columns which are mentioned in @ref set_dtypes.
   * If false, then all columns are returned, independent of the @ref set_dtypes setting.
   *
   * @param val Boolean value to enable/disable column pruning
   */
  void enable_prune_columns(bool val) { _prune_columns = val; }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable day first parsing format
   */
  void enable_dayfirst(bool val) { _dayfirst = val; }

  /**
   * @brief Set whether to use the legacy reader.
   *
   * @deprecated Since 24.06
   *
   * @param val Boolean value to enable/disable the legacy reader
   */
  [[deprecated]] void enable_legacy(bool val) { _legacy = val; }

  /**
   * @brief Set whether the reader should keep quotes of string values.
   *
   * @param val Boolean value to indicate whether the reader should keep quotes
   * of string values
   */
  void enable_keep_quotes(bool val) { _keep_quotes = val; }

  /**
   * @brief Set whether the reader should enable normalization of single quotes around strings.
   *
   * @param val Boolean value to indicate whether the reader should normalize single quotes around
   * strings
   */
  void enable_normalize_single_quotes(bool val) { _normalize_single_quotes = val; }

  /**
   * @brief Set whether the reader should enable normalization of unquoted whitespace
   *
   * @param val Boolean value to indicate whether the reader should normalize unquoted whitespace
   * characters i.e. tabs and spaces
   */
  void enable_normalize_whitespace(bool val) { _normalize_whitespace = val; }

  /**
   * @brief Specifies the JSON reader's behavior on invalid JSON lines.
   *
   * @param val An enum value to indicate the JSON reader's behavior on invalid JSON lines.
   */
  void set_recovery_mode(json_recovery_mode_t val) { _recovery_mode = val; }
};

/**
 * @brief Builds settings to use for `read_json()`.
 */
class json_reader_options_builder {
  json_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit json_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read avro file
   */
  explicit json_reader_options_builder(source_info src) : options{std::move(src)} {}

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector of dtypes
   * @return this for chaining
   */
  json_reader_options_builder& dtypes(std::vector<data_type> types)
  {
    options._dtypes = std::move(types);
    return *this;
  }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Column name -> dtype map
   * @return this for chaining
   */
  json_reader_options_builder& dtypes(std::map<std::string, data_type> types)
  {
    options._dtypes = std::move(types);
    return *this;
  }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Column name -> schema_element map
   * @return this for chaining
   */
  json_reader_options_builder& dtypes(std::map<std::string, schema_element> types)
  {
    options._dtypes = std::move(types);
    return *this;
  }

  /**
   * @brief Set the compression type.
   *
   * @param comp_type The compression type used
   * @return this for chaining
   */
  json_reader_options_builder& compression(compression_type comp_type)
  {
    options._compression = comp_type;
    return *this;
  }

  /**
   * @brief Set number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset
   * @return this for chaining
   */
  json_reader_options_builder& byte_range_offset(size_type offset)
  {
    options._byte_range_offset = offset;
    return *this;
  }

  /**
   * @brief Set number of bytes to read.
   *
   * @param size Number of bytes to read
   * @return this for chaining
   */
  json_reader_options_builder& byte_range_size(size_type size)
  {
    options._byte_range_size = size;
    return *this;
  }

  /**
   * @brief Set delimiter separating records in JSON lines
   *
   * @param delimiter Delimiter separating records in JSON lines
   * @return this for chaining
   */
  json_reader_options_builder& delimiter(char delimiter)
  {
    options.set_delimiter(delimiter);
    return *this;
  }

  /**
   * @brief Set whether to read the file as a json object per line.
   *
   * @param val Boolean value to enable/disable the option to read each line as a json object
   * @return this for chaining
   */
  json_reader_options_builder& lines(bool val)
  {
    options._lines = val;
    return *this;
  }

  /**
   * @brief Set whether to parse mixed types as a string column.
   * Also enables forcing to read a struct as string column using schema.
   *
   * @param val Boolean value to enable/disable parsing mixed types as a string column
   * @return this for chaining
   */
  json_reader_options_builder& mixed_types_as_string(bool val)
  {
    options._mixed_types_as_string = val;
    return *this;
  }

  /**
   * @brief Set whether to prune columns on read, selected based on the @ref dtypes option.
   *
   * When set as true, if the reader options include @ref dtypes, then
   * the reader will only return those columns which are mentioned in @ref dtypes.
   * If false, then all columns are returned, independent of the @ref dtypes setting.
   *
   * @param val Boolean value to enable/disable column pruning
   * @return this for chaining
   */
  json_reader_options_builder& prune_columns(bool val)
  {
    options._prune_columns = val;
    return *this;
  }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable day first parsing format
   * @return this for chaining
   */
  json_reader_options_builder& dayfirst(bool val)
  {
    options._dayfirst = val;
    return *this;
  }

  /**
   * @brief Set whether to use the legacy reader.
   *
   * @deprecated Since 24.06
   *
   * @param val Boolean value to enable/disable legacy parsing
   * @return this for chaining
   */
  [[deprecated]] json_reader_options_builder& legacy(bool val)
  {
    options._legacy = val;
    return *this;
  }

  /**
   * @brief Set whether the reader should keep quotes of string values.
   *
   * @param val Boolean value to indicate whether the reader should keep quotes
   * of string values
   * @return this for chaining
   */
  json_reader_options_builder& keep_quotes(bool val)
  {
    options._keep_quotes = val;
    return *this;
  }

  /**
   * @brief Set whether the reader should normalize single quotes around strings
   *
   * @param val Boolean value to indicate whether the reader should normalize single quotes
   * of strings
   * @return this for chaining
   */
  json_reader_options_builder& normalize_single_quotes(bool val)
  {
    options._normalize_single_quotes = val;
    return *this;
  }

  /**
   * @brief Set whether the reader should normalize unquoted whitespace
   *
   * @param val Boolean value to indicate whether the reader should normalize unquoted
   * whitespace
   * @return this for chaining
   */
  json_reader_options_builder& normalize_whitespace(bool val)
  {
    options._normalize_whitespace = val;
    return *this;
  }

  /**
   * @brief Specifies the JSON reader's behavior on invalid JSON lines.
   *
   * @param val An enum value to indicate the JSON reader's behavior on invalid JSON lines.
   * @return this for chaining
   */
  json_reader_options_builder& recovery_mode(json_recovery_mode_t val)
  {
    options._recovery_mode = val;
    return *this;
  }

  /**
   * @brief move json_reader_options member once it's built.
   */
  operator json_reader_options&&() { return std::move(options); }

  /**
   * @brief move json_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `json_reader_options` object r-value reference
   */
  json_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a JSON dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.json");
 *  auto options = cudf::io::read_json_options::builder(source);
 *  auto result  = cudf::io::read_json(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata.
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_json(
  json_reader_options options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group

/**
 * @addtogroup io_writers
 * @{
 * @file
 */

/**
 *@brief Builder to build options for `writer_json()`.
 */
class json_writer_options_builder;

/**
 * @brief Settings to use for `write_json()`.
 */
class json_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Set of columns to output
  table_view _table;
  // string to use for null entries
  std::string _na_rep = "";
  // Indicates whether to output nulls as 'null' or exclude the field
  bool _include_nulls = false;
  // Indicates whether to use JSON lines for records format
  bool _lines = false;
  // maximum number of rows to write in each chunk (limits memory use)
  size_type _rows_per_chunk = std::numeric_limits<size_type>::max();
  // string to use for values != 0 in INT8 types (default 'true')
  std::string _true_value = std::string{"true"};
  // string to use for values == 0 in INT8 types (default 'false')
  std::string _false_value = std::string{"false"};
  // Names of all columns; if empty, writer will generate column names
  std::optional<table_metadata> _metadata;  // Optional column names

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit json_writer_options(sink_info const& sink, table_view const& table)
    : _sink(sink), _table(table), _rows_per_chunk(table.num_rows())
  {
  }

  friend json_writer_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit json_writer_options() = default;

  /**
   * @brief Create builder to create `json_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build json_writer_options
   */
  static json_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink used for writer output.
   *
   * @return sink used for writer output
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns table that would be written to output.
   *
   * @return Table that would be written to output
   */
  [[nodiscard]] table_view const& get_table() const { return _table; }

  /**
   * @brief Returns metadata information.
   *
   * @return Metadata information
   */
  [[nodiscard]] std::optional<table_metadata> const& get_metadata() const { return _metadata; }

  /**
   * @brief Returns string to used for null entries.
   *
   * @return string to used for null entries
   */
  [[nodiscard]] std::string const& get_na_rep() const { return _na_rep; }

  /**
   * @brief Whether to output nulls as 'null'.
   *
   * @return `true` if nulls are output as 'null'
   */
  [[nodiscard]] bool is_enabled_include_nulls() const { return _include_nulls; }

  /**
   * @brief Whether to use JSON lines for records format.
   *
   * @return `true` if JSON lines is used for records format
   */
  [[nodiscard]] bool is_enabled_lines() const { return _lines; }

  /**
   * @brief Returns maximum number of rows to process for each file write.
   *
   * @return Maximum number of rows to process for each file write
   */
  [[nodiscard]] size_type get_rows_per_chunk() const { return _rows_per_chunk; }

  /**
   * @brief Returns string used for values != 0 in INT8 types.
   *
   * @return string used for values != 0 in INT8 types
   */
  [[nodiscard]] std::string const& get_true_value() const { return _true_value; }

  /**
   * @brief Returns string used for values == 0 in INT8 types.
   *
   * @return string used for values == 0 in INT8 types
   */
  [[nodiscard]] std::string const& get_false_value() const { return _false_value; }

  // Setter

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   */
  void set_table(table_view tbl) { _table = tbl; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata
   */
  void set_metadata(table_metadata metadata) { _metadata = std::move(metadata); }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value
   */
  void set_na_rep(std::string val) { _na_rep = std::move(val); }

  /**
   * @brief Enables/Disables output of nulls as 'null'.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_include_nulls(bool val) { _include_nulls = val; }

  /**
   * @brief Enables/Disables JSON lines for records format.
   *
   * @param val Boolean value to enable/disable JSON lines
   */
  void enable_lines(bool val) { _lines = val; }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk
   */
  void set_rows_per_chunk(size_type val) { _rows_per_chunk = val; }

  /**
   * @brief Sets string used for values != 0 in INT8 types.
   *
   * @param val String to represent values != 0 in INT8 types
   */
  void set_true_value(std::string val) { _true_value = std::move(val); }

  /**
   * @brief Sets string used for values == 0 in INT8 types.
   *
   * @param val String to represent values == 0 in INT8 types
   */
  void set_false_value(std::string val) { _false_value = std::move(val); }
};

/**
 * @brief Builder to build options for `writer_json()`
 */
class json_writer_options_builder {
  json_writer_options options;  ///< Options to be built.

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit json_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit json_writer_options_builder(sink_info const& sink, table_view const& table)
    : options{sink, table}
  {
  }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   * @return this for chaining
   */
  json_writer_options_builder& table(table_view tbl)
  {
    options._table = tbl;
    return *this;
  }

  /**
   * @brief Sets optional metadata (with column names).
   *
   * @param metadata metadata (with column names)
   * @return this for chaining
   */
  json_writer_options_builder& metadata(table_metadata metadata)
  {
    options._metadata = std::move(metadata);
    return *this;
  }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value
   * @return this for chaining
   */
  json_writer_options_builder& na_rep(std::string val)
  {
    options._na_rep = std::move(val);
    return *this;
  };

  /**
   * @brief Enables/Disables output of nulls as 'null'.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  json_writer_options_builder& include_nulls(bool val)
  {
    options._include_nulls = val;
    return *this;
  }

  /**
   * @brief Enables/Disables JSON lines for records format.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  json_writer_options_builder& lines(bool val)
  {
    options._lines = val;
    return *this;
  }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk
   * @return this for chaining
   */
  json_writer_options_builder& rows_per_chunk(int val)
  {
    options._rows_per_chunk = val;
    return *this;
  }

  /**
   * @brief Sets string used for values != 0 in INT8 types.
   *
   * @param val String to represent values != 0 in INT8 types
   * @return this for chaining
   */
  json_writer_options_builder& true_value(std::string val)
  {
    options._true_value = std::move(val);
    return *this;
  }

  /**
   * @brief Sets string used for values == 0 in INT8 types.
   *
   * @param val String to represent values == 0 in INT8 types
   * @return this for chaining
   */
  json_writer_options_builder& false_value(std::string val)
  {
    options._false_value = std::move(val);
    return *this;
  }

  /**
   * @brief move `json_writer_options` member once it's built.
   */
  operator json_writer_options&&() { return std::move(options); }

  /**
   * @brief move `json_writer_options` member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `json_writer_options` object's r-value reference
   */
  json_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to JSON format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  auto destination = cudf::io::sink_info("dataset.json");
 *  auto options     = cudf::io::json_writer_options(destination, table->view())
 *    .na_rep(na)
 *    .lines(lines)
 *    .rows_per_chunk(rows_per_chunk);
 *
 *  cudf::io::write_json(options);
 * @endcode
 *
 * @param options Settings for controlling writing behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 */
void write_json(json_writer_options const& options,
                rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace io
}  // namespace cudf
