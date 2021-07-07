/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/optional.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief Builds parquet_reader_options to use for `read_parquet()`.
 */
class parquet_reader_options_builder;

/**
 * @brief Settings or `read_parquet()`.
 */
class parquet_reader_options {
  source_info _source;

  // Names of column to read; empty is all
  std::vector<std::string> _columns;

  // List of individual row groups to read (ignored if empty)
  std::vector<std::vector<size_type>> _row_groups;
  // Number of rows to skip from the start
  size_type _skip_rows = 0;
  // Number of rows to read; -1 is all
  size_type _num_rows = -1;

  // Whether to store string data as categorical type
  bool _convert_strings_to_categories = false;
  // Whether to use PANDAS metadata to load columns
  bool _use_pandas_metadata = true;
  // Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  // force decimal reading to error if resorting to
  // doubles for storage of types unsupported by cudf
  bool _strict_decimal_types = false;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read parquet file.
   */
  explicit parquet_reader_options(source_info const& src) : _source(src) {}

  friend parquet_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_reader_options() = default;

  /**
   * @brief Creates a parquet_reader_options_builder which will build parquet_reader_options.
   *
   * @param src Source information to read parquet file.
   * @return Builder to build reader options.
   */
  static parquet_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info.
   */
  source_info const& get_source() const { return _source; }

  /**
   * @brief Returns true/false depending on whether strings should be converted to categories or
   * not.
   */
  bool is_enabled_convert_strings_to_categories() const { return _convert_strings_to_categories; }

  /**
   * @brief Returns true/false depending whether to use pandas metadata or not while reading.
   */
  bool is_enabled_use_pandas_metadata() const { return _use_pandas_metadata; }

  /**
   * @brief Returns number of rows to skip from the start.
   */
  size_type get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of rows to read.
   */
  size_type get_num_rows() const { return _num_rows; }

  /**
   * @brief Returns names of column to be read.
   */
  std::vector<std::string> const& get_columns() const { return _columns; }

  /**
   * @brief Returns list of individual row groups to be read.
   */
  std::vector<std::vector<size_type>> const& get_row_groups() const { return _row_groups; }

  /**
   * @brief Returns timestamp type used to cast timestamp columns.
   */
  data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Returns true if strict decimal types is set, which errors if reading
   * a decimal type that is unsupported.
   */
  bool is_enabled_strict_decimal_types() const { return _strict_decimal_types; }

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names.
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }

  /**
   * @brief Sets vector of individual row groups to read.
   *
   * @param row_groups Vector of row groups to read.
   */
  void set_row_groups(std::vector<std::vector<size_type>> row_groups)
  {
    if ((!row_groups.empty()) and ((_skip_rows != 0) or (_num_rows != -1))) {
      CUDF_FAIL("row_groups can't be set along with skip_rows and num_rows");
    }

    _row_groups = std::move(row_groups);
  }

  /**
   * @brief Sets to enable/disable conversion of strings to categories.
   *
   * @param val Boolean value to enable/disable conversion of string columns to categories.
   */
  void enable_convert_strings_to_categories(bool val) { _convert_strings_to_categories = val; }

  /**
   * @brief Sets to enable/disable use of pandas metadata to read.
   *
   * @param val Boolean value whether to use pandas metadata.
   */
  void enable_use_pandas_metadata(bool val) { _use_pandas_metadata = val; }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start.
   */
  void set_skip_rows(size_type val)
  {
    if ((val != 0) and (!_row_groups.empty())) {
      CUDF_FAIL("skip_rows can't be set along with a non-empty row_groups");
    }

    _skip_rows = val;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip.
   */
  void set_num_rows(size_type val)
  {
    if ((val != -1) and (!_row_groups.empty())) {
      CUDF_FAIL("num_rows can't be set along with a non-empty row_groups");
    }

    _num_rows = val;
  }

  /**
   * @brief Sets timestamp_type used to cast timestamp columns.
   *
   * @param type The timestamp data_type to which all timestamp columns need to be cast.
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }

  /**
   * @brief Enables/disables strict decimal type checking.
   *
   * @param val If true, cudf will error if reading a decimal type that is unsupported. If false,
   * cudf will convert unsupported types to double.
   */
  void set_strict_decimal_types(bool val) { _strict_decimal_types = val; }
};

class parquet_reader_options_builder {
  parquet_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  parquet_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read parquet file.
   */
  explicit parquet_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names.
   * @return this for chaining.
   */
  parquet_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options._columns = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets vector of individual row groups to read.
   *
   * @param row_groups Vector of row groups to read.
   * @return this for chaining.
   */
  parquet_reader_options_builder& row_groups(std::vector<std::vector<size_type>> row_groups)
  {
    options.set_row_groups(std::move(row_groups));
    return *this;
  }

  /**
   * @brief Sets enable/disable conversion of strings to categories.
   *
   * @param val Boolean value to enable/disable conversion of string columns to categories.
   * @return this for chaining.
   */
  parquet_reader_options_builder& convert_strings_to_categories(bool val)
  {
    options._convert_strings_to_categories = val;
    return *this;
  }

  /**
   * @brief Sets to enable/disable use of pandas metadata to read.
   *
   * @param val Boolean value whether to use pandas metadata.
   * @return this for chaining.
   */
  parquet_reader_options_builder& use_pandas_metadata(bool val)
  {
    options._use_pandas_metadata = val;
    return *this;
  }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start.
   * @return this for chaining.
   */
  parquet_reader_options_builder& skip_rows(size_type val)
  {
    options.set_skip_rows(val);
    return *this;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip.
   * @return this for chaining.
   */
  parquet_reader_options_builder& num_rows(size_type val)
  {
    options.set_num_rows(val);
    return *this;
  }

  /**
   * @brief timestamp_type used to cast timestamp columns.
   *
   * @param type The timestamp data_type to which all timestamp columns need to be cast.
   * @return this for chaining.
   */
  parquet_reader_options_builder& timestamp_type(data_type type)
  {
    options._timestamp_type = type;
    return *this;
  }

  /**
   * @brief Sets to enable/disable error with unsupported decimal types.
   *
   * @param val Boolean value whether to error with unsupported decimal types.
   * @return this for chaining.
   */
  parquet_reader_options_builder& use_strict_decimal_types(bool val)
  {
    options._strict_decimal_types = val;
    return *this;
  }

  /**
   * @brief move parquet_reader_options member once it's built.
   */
  operator parquet_reader_options&&() { return std::move(options); }

  /**
   * @brief move parquet_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  parquet_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a Parquet dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::io::parquet_reader_options options =
 *  cudf::io::parquet_reader_options::builder(cudf::source_info(filepath));
 *  ...
 *  auto result = cudf::read_parquet(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_parquet(
  parquet_reader_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
/**
 * @addtogroup io_writers
 * @{
 * @file
 */
class table_input_metadata;

class column_in_metadata {
  friend table_input_metadata;
  std::string _name = "";
  thrust::optional<bool> _nullable;
  // TODO: This isn't implemented yet
  bool _list_column_is_map  = false;
  bool _use_int96_timestamp = false;
  // bool _output_as_binary = false;
  thrust::optional<uint8_t> _decimal_precision;
  std::vector<column_in_metadata> children;

 public:
  /**
   * @brief Get the children of this column metadata
   *
   * @return this for chaining
   */
  column_in_metadata& add_child(column_in_metadata const& child)
  {
    children.push_back(child);
    return *this;
  }

  /**
   * @brief Set the name of this column
   *
   * @return this for chaining
   */
  column_in_metadata& set_name(std::string const& name)
  {
    _name = name;
    return *this;
  }

  /**
   * @brief Set the nullability of this column
   *
   * Only valid in case of chunked writes. In single writes, this option is ignored.
   *
   * @return column_in_metadata&
   */
  column_in_metadata& set_nullability(bool nullable)
  {
    _nullable = nullable;
    return *this;
  }

  /**
   * @brief Specify that this list column should be encoded as a map in the written parquet file
   *
   * The column must have the structure list<struct<key, value>>. This option is invalid otherwise
   *
   * @return this for chaining
   */
  column_in_metadata& set_list_column_as_map()
  {
    _list_column_is_map = true;
    return *this;
  }

  /**
   * @brief Specifies whether this timestamp column should be encoded using the deprecated int96
   * physical type. Only valid for the following column types:
   * timestamp_s, timestamp_ms, timestamp_us, timestamp_ns
   *
   * @param req True = use int96 physical type. False = use int64 physical type
   * @return this for chaining
   */
  column_in_metadata& set_int96_timestamps(bool req)
  {
    _use_int96_timestamp = req;
    return *this;
  }

  /**
   * @brief Set the decimal precision of this column. Only valid if this column is a decimal
   * (fixed-point) type
   *
   * @param precision The integer precision to set for this decimal column
   * @return this for chaining
   */
  column_in_metadata& set_decimal_precision(uint8_t precision)
  {
    _decimal_precision = precision;
    return *this;
  }

  /**
   * @brief Get reference to a child of this column
   *
   * @param i Index of the child to get
   * @return this for chaining
   */
  column_in_metadata& child(size_type i) { return children[i]; }

  /**
   * @brief Get const reference to a child of this column
   *
   * @param i Index of the child to get
   * @return this for chaining
   */
  column_in_metadata const& child(size_type i) const { return children[i]; }

  /**
   * @brief Get the name of this column
   */
  std::string get_name() const { return _name; }

  /**
   * @brief Get whether nullability has been explicitly set for this column.
   */
  bool is_nullability_defined() const { return _nullable.has_value(); }

  /**
   * @brief Gets the explicitly set nullability for this column.
   * @throws If nullability is not explicitly defined for this column.
   *         Check using `is_nullability_defined()` first.
   */
  bool nullable() const { return _nullable.value(); }

  /**
   * @brief If this is the metadata of a list column, returns whether it is to be encoded as a map.
   */
  bool is_map() const { return _list_column_is_map; }

  /**
   * @brief Get whether to encode this timestamp column using deprecated int96 physical type
   */
  bool is_enabled_int96_timestamps() const { return _use_int96_timestamp; }

  /**
   * @brief Get whether precision has been set for this decimal column
   */
  bool is_decimal_precision_set() const { return _decimal_precision.has_value(); }

  /**
   * @brief Get the decimal precision that was set for this column.
   * @throws If decimal precision was not set for this column.
   *         Check using `is_decimal_precision_set()` first.
   */
  uint8_t get_decimal_precision() const { return _decimal_precision.value(); }

  /**
   * @brief Get the number of children of this column
   */
  size_type num_children() const { return children.size(); }
};

class table_input_metadata {
 public:
  table_input_metadata() = default;  // Required by cython

  /**
   * @brief Construct a new table_input_metadata from a table_view.
   *
   * The constructed table_input_metadata has the same structure as the passed table_view
   *
   * @param table The table_view to construct metadata for
   * @param user_data Optional Additional metadata to encode, as key-value pairs
   */
  table_input_metadata(table_view const& table, std::map<std::string, std::string> user_data = {});

  std::vector<column_in_metadata> column_metadata;
  std::map<std::string, std::string> user_data;  //!< Format-dependent metadata as key-values pairs
};

/**
 * @brief Class to build `parquet_writer_options`.
 */
class parquet_writer_options_builder;

/**
 * @brief Settings for `write_parquet()`.
 */
class parquet_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::SNAPPY;
  // Specify the level of statistics in the output file
  statistics_freq _stats_level = statistics_freq::STATISTICS_ROWGROUP;
  // Sets of columns to output
  table_view _table;
  // Optional associated metadata
  table_input_metadata const* _metadata = nullptr;
  // Parquet writer can write INT96 or TIMESTAMP_MICROS. Defaults to TIMESTAMP_MICROS.
  // If true then overrides any per-column setting in _metadata.
  bool _write_timestamps_as_int96 = false;
  // Column chunks file path to be set in the raw output metadata
  std::string _column_chunks_file_path;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   */
  explicit parquet_writer_options(sink_info const& sink, table_view const& table)
    : _sink(sink), _table(table)
  {
  }

  friend class parquet_writer_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  parquet_writer_options() = default;

  /**
   * @brief Create builder to create `parquet_writer_options`.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   *
   * @return Builder to build parquet_writer_options.
   */
  static parquet_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Create builder to create `parquet_writer_options`.
   *
   * @return parquet_writer_options_builder.
   */
  static parquet_writer_options_builder builder();

  /**
   * @brief Returns sink info.
   */
  sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression format used.
   */
  compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns level of statistics requested in output file.
   */
  statistics_freq get_stats_level() const { return _stats_level; }

  /**
   * @brief Returns table_view.
   */
  table_view get_table() const { return _table; }

  /**
   * @brief Returns associated metadata.
   */
  table_input_metadata const* get_metadata() const { return _metadata; }

  /**
   * @brief Returns `true` if timestamps will be written as INT96
   */
  bool is_enabled_int96_timestamps() const { return _write_timestamps_as_int96; }

  /**
   * @brief Returns Column chunks file path to be set in the raw output metadata.
   */
  std::string get_column_chunks_file_path() const { return _column_chunks_file_path; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata.
   */
  void set_metadata(table_input_metadata const* metadata) { _metadata = metadata; }

  /**
   * @brief Sets the level of statistics.
   *
   * @param sf Level of statistics requested in the output file.
   */
  void set_stats_level(statistics_freq sf) { _stats_level = sf; }

  /**
   * @brief Sets compression type.
   *
   * @param compression The compression type to use.
   */
  void set_compression(compression_type compression) { _compression = compression; }

  /**
   * @brief Sets timestamp writing preferences. INT96 timestamps will be written
   * if `true` and TIMESTAMP_MICROS will be written if `false`.
   *
   * @param req Boolean value to enable/disable writing of INT96 timestamps
   */
  void enable_int96_timestamps(bool req) { _write_timestamps_as_int96 = req; }

  /**
   * @brief Sets column chunks file path to be set in the raw output metadata.
   *
   * @param file_path String which indicates file path.
   */
  void set_column_chunks_file_path(std::string file_path)
  {
    _column_chunks_file_path.assign(file_path);
  }
};

class parquet_writer_options_builder {
  parquet_writer_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   */
  explicit parquet_writer_options_builder(sink_info const& sink, table_view const& table)
    : options(sink, table)
  {
  }

  /**
   * @brief Sets metadata in parquet_writer_options.
   *
   * @param metadata Associated metadata.
   * @return this for chaining.
   */
  parquet_writer_options_builder& metadata(table_input_metadata const* metadata)
  {
    options._metadata = metadata;
    return *this;
  }

  /**
   * @brief Sets the level of statistics in parquet_writer_options.
   *
   * @param sf Level of statistics requested in the output file.
   * @return this for chaining.
   */
  parquet_writer_options_builder& stats_level(statistics_freq sf)
  {
    options._stats_level = sf;
    return *this;
  }

  /**
   * @brief Sets compression type in parquet_writer_options.
   *
   * @param compression The compression type to use.
   * @return this for chaining.
   */
  parquet_writer_options_builder& compression(compression_type compression)
  {
    options._compression = compression;
    return *this;
  }

  /**
   * @brief Sets column chunks file path to be set in the raw output metadata.
   *
   * @param file_path String which indicates file path.
   * @return this for chaining.
   */
  parquet_writer_options_builder& column_chunks_file_path(std::string file_path)
  {
    options._column_chunks_file_path.assign(file_path);
    return *this;
  }

  /**
   * @brief Sets whether int96 timestamps are written or not in parquet_writer_options.
   *
   * @param enabled Boolean value to enable/disable int96 timestamps.
   * @return this for chaining.
   */
  parquet_writer_options_builder& int96_timestamps(bool enabled)
  {
    options._write_timestamps_as_int96 = enabled;
    return *this;
  }

  /**
   * @brief move parquet_writer_options member once it's built.
   */
  operator parquet_writer_options&&() { return std::move(options); }

  /**
   * @brief move parquet_writer_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  parquet_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to parquet format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::io::parquet_writer_options options =
 *  cudf::io::parquet_writer_options::builder(cudf::sink_info(filepath), table->view());
 *  ...
 *  cudf::write_parquet(options);
 * @endcode
 *
 * @param options Settings for controlling writing behavior.
 * @param mr Device memory resource to use for device memory allocation.
 *
 * @return A blob that contains the file metadata (parquet FileMetadata thrift message) if
 *         requested in parquet_writer_options (empty blob otherwise).
 */

std::unique_ptr<std::vector<uint8_t>> write_parquet(
  parquet_writer_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Merges multiple raw metadata blobs that were previously created by write_parquet
 * into a single metadata blob.
 *
 * @param[in] metadata_list List of input file metadata.
 * @return A parquet-compatible blob that contains the data for all row groups in the list.
 */
std::unique_ptr<std::vector<uint8_t>> merge_rowgroup_metadata(
  const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list);

/**
 * @brief Builds options for chunked_parquet_writer_options.
 */
class chunked_parquet_writer_options_builder;

/**
 * @brief Settings for `write_parquet_chunked()`.
 */
class chunked_parquet_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  // Specify the level of statistics in the output file
  statistics_freq _stats_level = statistics_freq::STATISTICS_ROWGROUP;
  // Optional associated metadata.
  table_input_metadata const* _metadata = nullptr;
  // Parquet writer can write INT96 or TIMESTAMP_MICROS. Defaults to TIMESTAMP_MICROS.
  // If true then overrides any per-column setting in _metadata.
  bool _write_timestamps_as_int96 = false;

  /**
   * @brief Constructor from sink.
   *
   * @param sink Sink used for writer output.
   */
  explicit chunked_parquet_writer_options(sink_info const& sink) : _sink(sink) {}

  friend chunked_parquet_writer_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  chunked_parquet_writer_options() = default;

  /**
   * @brief Returns sink info.
   */
  sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression format used.
   */
  compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns level of statistics requested in output file.
   */
  statistics_freq get_stats_level() const { return _stats_level; }

  /**
   * @brief Returns metadata information.
   */
  table_input_metadata const* get_metadata() const { return _metadata; }

  /**
   * @brief Returns `true` if timestamps will be written as INT96
   */
  bool is_enabled_int96_timestamps() const { return _write_timestamps_as_int96; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata.
   */
  void set_metadata(table_input_metadata const* metadata) { _metadata = metadata; }

  /**
   * @brief Sets the level of statistics in parquet_writer_options.
   *
   * @param sf Level of statistics requested in the output file.
   */
  void set_stats_level(statistics_freq sf) { _stats_level = sf; }

  /**
   * @brief Sets compression type.
   *
   * @param compression The compression type to use.
   */
  void set_compression(compression_type compression) { _compression = compression; }

  /**
   * @brief Sets timestamp writing preferences. INT96 timestamps will be written
   * if `true` and TIMESTAMP_MICROS will be written if `false`.
   *
   * @param req Boolean value to enable/disable writing of INT96 timestamps
   */
  void enable_int96_timestamps(bool req) { _write_timestamps_as_int96 = req; }

  /**
   * @brief creates builder to build chunked_parquet_writer_options.
   *
   * @param sink sink to use for writer output.
   *
   * @return Builder to build `chunked_parquet_writer_options`.
   */
  static chunked_parquet_writer_options_builder builder(sink_info const& sink);
};

class chunked_parquet_writer_options_builder {
  chunked_parquet_writer_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  chunked_parquet_writer_options_builder() = default;

  /**
   * @brief Constructor from sink.
   *
   * @param sink The sink used for writer output.
   */
  chunked_parquet_writer_options_builder(sink_info const& sink) : options(sink){};

  /**
   * @brief Sets metadata to chunked_parquet_writer_options.
   *
   * @param metadata Associated metadata.
   * @return this for chaining.
   */
  chunked_parquet_writer_options_builder& metadata(table_input_metadata const* metadata)
  {
    options._metadata = metadata;
    return *this;
  }

  /**
   * @brief Sets Sets the level of statistics in chunked_parquet_writer_options.
   *
   * @param sf Level of statistics requested in the output file.
   * @return this for chaining.
   */
  chunked_parquet_writer_options_builder& stats_level(statistics_freq sf)
  {
    options._stats_level = sf;
    return *this;
  }

  /**
   * @brief Sets compression type to chunked_parquet_writer_options.
   *
   * compression The compression type to use.
   * @return this for chaining.
   */
  chunked_parquet_writer_options_builder& compression(compression_type compression)
  {
    options._compression = compression;
    return *this;
  }

  /**
   * @brief Set to true if timestamps should be written as
   * int96 types instead of int64 types. Even though int96 is deprecated and is
   * not an internal type for cudf, it needs to be written for backwards
   * compatability reasons.
   *
   * @param enabled Boolean value to enable/disable int96 timestamps.
   * @return this for chaining.
   */
  chunked_parquet_writer_options_builder& int96_timestamps(bool enabled)
  {
    options._write_timestamps_as_int96 = enabled;
    return *this;
  }

  /**
   * @brief move chunked_parquet_writer_options member once it's built.
   */
  operator chunked_parquet_writer_options&&() { return std::move(options); }

  /**
   * @brief move chunked_parquet_writer_options member once it's is built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  chunked_parquet_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Merges multiple raw metadata blobs that were previously created by write_parquet
 * into a single metadata blob
 *
 * @ingroup io_writers
 *
 * @param[in] metadata_list List of input file metadata
 * @return A parquet-compatible blob that contains the data for all rowgroups in the list
 */
std::unique_ptr<std::vector<uint8_t>> merge_rowgroup_metadata(
  const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list);

/**
 * @brief chunked parquet writer class to handle options and write tables in chunks.
 *
 * The intent of the parquet_chunked_writer is to allow writing of an
 * arbitrarily large / arbitrary number of rows to a parquet file in multiple passes.
 *
 * The following code snippet demonstrates how to write a single parquet file containing
 * one logical table by writing a series of individual cudf::tables.
 *
 * @code
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::io::chunked_parquet_writer_options options =
 *  cudf::io::chunked_parquet_writer_options::builder(cudf::sink_info(filepath), table->view());
 *  ...
 *  cudf::io::parquet_chunked_writer writer(options)
 *  writer.write(table0)
 *  writer.write(table1)
 *  ...
 *  writer.close()
 *  @endcode
 */
class parquet_chunked_writer {
 public:
  /**
   * @brief Default constructor, this should never be used.
   *        This is added just to satisfy cython.
   */
  parquet_chunked_writer() = default;

  /**
   * @brief Constructor with chunked writer options
   *
   * @param[in] op options used to write table
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  parquet_chunked_writer(
    chunked_parquet_writer_options const& op,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Writes table to output.
   *
   * @param[in] table Table that needs to be written
   * @return returns reference of the class object
   */
  parquet_chunked_writer& write(table_view const& table);

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output metadata
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list only if
   * `column_chunks_file_path` is provided, else null.
   */
  std::unique_ptr<std::vector<uint8_t>> close(std::string const& column_chunks_file_path = "");

  // Unique pointer to impl writer class
  std::unique_ptr<cudf::io::detail::parquet::writer> writer;
};

/** @} */  // end of group
}  // namespace io
}  // namespace cudf
