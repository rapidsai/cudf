/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudf::io {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

constexpr size_t default_row_group_size_bytes   = 128 * 1024 * 1024;  ///< 128MB per row group
constexpr size_type default_row_group_size_rows = 1000000;     ///< 1 million rows per row group
constexpr size_t default_max_page_size_bytes    = 512 * 1024;  ///< 512KB per page
constexpr size_type default_max_page_size_rows  = 20000;       ///< 20k rows per page
constexpr int32_t default_column_index_truncate_length = 64;   ///< truncate to 64 bytes

class parquet_reader_options_builder;

/**
 * @brief Settings for `read_parquet()`.
 */
class parquet_reader_options {
  source_info _source;

  // Path in schema of column to read; `nullopt` is all
  std::optional<std::vector<std::string>> _columns;

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

  std::optional<std::vector<reader_column_schema>> _reader_column_schema;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read parquet file
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
   * @param src Source information to read parquet file
   * @return Builder to build reader options
   */
  static parquet_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns true/false depending on whether strings should be converted to categories or
   * not.
   *
   * @return `true` if strings should be converted to categories
   */
  [[nodiscard]] bool is_enabled_convert_strings_to_categories() const
  {
    return _convert_strings_to_categories;
  }

  /**
   * @brief Returns true/false depending whether to use pandas metadata or not while reading.
   *
   * @return `true` if pandas metadata is used while reading
   */
  [[nodiscard]] bool is_enabled_use_pandas_metadata() const { return _use_pandas_metadata; }

  /**
   * @brief Returns optional tree of metadata.
   *
   * @return vector of reader_column_schema objects.
   */
  [[nodiscard]] std::optional<std::vector<reader_column_schema>> get_column_schema() const
  {
    return _reader_column_schema;
  }

  /**
   * @brief Returns number of rows to skip from the start.
   *
   * @return Number of rows to skip from the start
   */
  [[nodiscard]] size_type get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of rows to read.
   *
   * @return Number of rows to read
   */
  [[nodiscard]] size_type get_num_rows() const { return _num_rows; }

  /**
   * @brief Returns names of column to be read, if set.
   *
   * @return Names of column to be read; `nullopt` if the option is not set
   */
  [[nodiscard]] auto const& get_columns() const { return _columns; }

  /**
   * @brief Returns list of individual row groups to be read.
   *
   * @return List of individual row groups to be read
   */
  [[nodiscard]] auto const& get_row_groups() const { return _row_groups; }

  /**
   * @brief Returns timestamp type used to cast timestamp columns.
   *
   * @return Timestamp type used to cast timestamp columns
   */
  data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }

  /**
   * @brief Sets vector of individual row groups to read.
   *
   * @param row_groups Vector of row groups to read
   */
  void set_row_groups(std::vector<std::vector<size_type>> row_groups);

  /**
   * @brief Sets to enable/disable conversion of strings to categories.
   *
   * @param val Boolean value to enable/disable conversion of string columns to categories
   */
  void enable_convert_strings_to_categories(bool val) { _convert_strings_to_categories = val; }

  /**
   * @brief Sets to enable/disable use of pandas metadata to read.
   *
   * @param val Boolean value whether to use pandas metadata
   */
  void enable_use_pandas_metadata(bool val) { _use_pandas_metadata = val; }

  /**
   * @brief Sets reader column schema.
   *
   * @param val Tree of schema nodes to enable/disable conversion of binary to string columns.
   * Note default is to convert to string columns.
   */
  void set_column_schema(std::vector<reader_column_schema> val)
  {
    _reader_column_schema = std::move(val);
  }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start
   */
  void set_skip_rows(size_type val);

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip
   */
  void set_num_rows(size_type val);

  /**
   * @brief Sets timestamp_type used to cast timestamp columns.
   *
   * @param type The timestamp data_type to which all timestamp columns need to be cast
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }
};

/**
 * @brief Builds parquet_reader_options to use for `read_parquet()`.
 */
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
   * @param src The source information used to read parquet file
   */
  explicit parquet_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names
   * @return this for chaining
   */
  parquet_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options._columns = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets vector of individual row groups to read.
   *
   * @param row_groups Vector of row groups to read
   * @return this for chaining
   */
  parquet_reader_options_builder& row_groups(std::vector<std::vector<size_type>> row_groups)
  {
    options.set_row_groups(std::move(row_groups));
    return *this;
  }

  /**
   * @brief Sets enable/disable conversion of strings to categories.
   *
   * @param val Boolean value to enable/disable conversion of string columns to categories
   * @return this for chaining
   */
  parquet_reader_options_builder& convert_strings_to_categories(bool val)
  {
    options._convert_strings_to_categories = val;
    return *this;
  }

  /**
   * @brief Sets to enable/disable use of pandas metadata to read.
   *
   * @param val Boolean value whether to use pandas metadata
   * @return this for chaining
   */
  parquet_reader_options_builder& use_pandas_metadata(bool val)
  {
    options._use_pandas_metadata = val;
    return *this;
  }

  /**
   * @brief Sets reader metadata.
   *
   * @param val Tree of metadata information.
   * @return this for chaining
   */
  parquet_reader_options_builder& set_column_schema(std::vector<reader_column_schema> val)
  {
    options._reader_column_schema = std::move(val);
    return *this;
  }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start
   * @return this for chaining
   */
  parquet_reader_options_builder& skip_rows(size_type val)
  {
    options.set_skip_rows(val);
    return *this;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip
   * @return this for chaining
   */
  parquet_reader_options_builder& num_rows(size_type val)
  {
    options.set_num_rows(val);
    return *this;
  }

  /**
   * @brief timestamp_type used to cast timestamp columns.
   *
   * @param type The timestamp data_type to which all timestamp columns need to be cast
   * @return this for chaining
   */
  parquet_reader_options_builder& timestamp_type(data_type type)
  {
    options._timestamp_type = type;
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
   *
   * @return Built `parquet_reader_options` object's r-value reference
   */
  parquet_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a Parquet dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.parquet");
 *  auto options = cudf::io::parquet_reader_options::builder(source);
 *  auto result  = cudf::io::read_parquet(options);
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

/**
 * @brief The chunked parquet reader class to read Parquet file iteratively in to a series of
 * tables, chunk by chunk.
 *
 * This class is designed to address the reading issue when reading very large Parquet files such
 * that the sizes of their column exceed the limit that can be stored in cudf column. By reading the
 * file content by chunks using this class, each chunk is guaranteed to have its sizes stay within
 * the given limit.
 */
class chunked_parquet_reader {
 public:
  /**
   * @brief Default constructor, this should never be used.
   *
   * This is added just to satisfy cython.
   */
  chunked_parquet_reader() = default;

  /**
   * @brief Constructor for chunked reader.
   *
   * This constructor requires the same `parquet_reader_option` parameter as in
   * `cudf::read_parquet()`, and an additional parameter to specify the size byte limit of the
   * output table for each reading.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   * @param options The options used to read Parquet file
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    parquet_reader_options const& options,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Destructor, destroying the internal reader instance.
   *
   * Since the declaration of the internal `reader` object does not exist in this header, this
   * destructor needs to be defined in a separate source file which can access to that object's
   * declaration.
   */
  ~chunked_parquet_reader();

  /**
   * @brief Check if there is any data in the given file has not yet read.
   *
   * @return A boolean value indicating if there is any data left to read
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @brief Read a chunk of rows in the given Parquet file.
   *
   * The sequence of returned tables, if concatenated by their order, guarantees to form a complete
   * dataset as reading the entire given file at once.
   *
   * An empty table will be returned if the given file is empty, or all the data in the file has
   * been read and returned by the previous calls.
   *
   * @return An output `cudf::table` along with its metadata
   */
  [[nodiscard]] table_with_metadata read_chunk() const;

 private:
  std::unique_ptr<cudf::io::detail::parquet::chunked_reader> reader;
};

/** @} */  // end of group
/**
 * @addtogroup io_writers
 * @{
 * @file
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
  // Partitions described as {start_row, num_rows} pairs
  std::vector<partition_info> _partitions;
  // Optional associated metadata
  table_input_metadata const* _metadata = nullptr;
  // Optional footer key_value_metadata
  std::vector<std::map<std::string, std::string>> _user_data;
  // Parquet writer can write INT96 or TIMESTAMP_MICROS. Defaults to TIMESTAMP_MICROS.
  // If true then overrides any per-column setting in _metadata.
  bool _write_timestamps_as_int96 = false;
  // Column chunks file paths to be set in the raw output metadata. One per output file
  std::vector<std::string> _column_chunks_file_paths;
  // Maximum size of each row group (unless smaller than a single page)
  size_t _row_group_size_bytes = default_row_group_size_bytes;
  // Maximum number of rows in row group (unless smaller than a single page)
  size_type _row_group_size_rows = default_row_group_size_rows;
  // Maximum size of each page (uncompressed)
  size_t _max_page_size_bytes = default_max_page_size_bytes;
  // Maximum number of rows in a page
  size_type _max_page_size_rows = default_max_page_size_rows;
  // Maximum size of min or max values in column index
  int32_t _column_index_truncate_length = default_column_index_truncate_length;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit parquet_writer_options(sink_info const& sink, table_view const& table)
    : _sink(sink), _table(table)
  {
  }

  friend parquet_writer_options_builder;

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
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build parquet_writer_options
   */
  static parquet_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Create builder to create `parquet_writer_options`.
   *
   * @return parquet_writer_options_builder
   */
  static parquet_writer_options_builder builder();

  /**
   * @brief Returns sink info.
   *
   * @return Sink info
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression format used.
   *
   * @return Compression format
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns level of statistics requested in output file.
   *
   * @return level of statistics requested in output file
   */
  [[nodiscard]] statistics_freq get_stats_level() const { return _stats_level; }

  /**
   * @brief Returns table_view.
   *
   * @return Table view
   */
  [[nodiscard]] table_view get_table() const { return _table; }

  /**
   * @brief Returns partitions.
   *
   * @return Partitions
   */
  [[nodiscard]] std::vector<partition_info> const& get_partitions() const { return _partitions; }

  /**
   * @brief Returns associated metadata.
   *
   * @return Associated metadata
   */
  [[nodiscard]] table_input_metadata const* get_metadata() const { return _metadata; }

  /**
   * @brief Returns Key-Value footer metadata information.
   *
   * @return Key-Value footer metadata information
   */
  std::vector<std::map<std::string, std::string>> const& get_key_value_metadata() const
  {
    return _user_data;
  }

  /**
   * @brief Returns `true` if timestamps will be written as INT96
   *
   * @return `true` if timestamps will be written as INT96
   */
  bool is_enabled_int96_timestamps() const { return _write_timestamps_as_int96; }

  /**
   * @brief Returns Column chunks file paths to be set in the raw output metadata.
   *
   * @return Column chunks file paths to be set in the raw output metadata
   */
  std::vector<std::string> const& get_column_chunks_file_paths() const
  {
    return _column_chunks_file_paths;
  }

  /**
   * @brief Returns maximum row group size, in bytes.
   *
   * @return Maximum row group size, in bytes
   */
  auto get_row_group_size_bytes() const { return _row_group_size_bytes; }

  /**
   * @brief Returns maximum row group size, in rows.
   *
   * @return Maximum row group size, in rows
   */
  auto get_row_group_size_rows() const { return _row_group_size_rows; }

  /**
   * @brief Returns the maximum uncompressed page size, in bytes.
   *
   * If set larger than the row group size, then this will return the row group size.
   *
   * @return Maximum uncompressed page size, in bytes
   */
  auto get_max_page_size_bytes() const
  {
    return std::min(_max_page_size_bytes, get_row_group_size_bytes());
  }

  /**
   * @brief Returns maximum page size, in rows.
   *
   * If set larger than the row group size, then this will return the row group size.
   *
   * @return Maximum page size, in rows
   */
  auto get_max_page_size_rows() const
  {
    return std::min(_max_page_size_rows, get_row_group_size_rows());
  }

  /**
   * @brief Returns maximum length of min or max values in column index, in bytes.
   *
   * @return length min/max will be truncated to
   */
  auto get_column_index_truncate_length() const { return _column_index_truncate_length; }

  /**
   * @brief Sets partitions.
   *
   * @param partitions Partitions of input table in {start_row, num_rows} pairs. If specified, must
   * be same size as number of sinks in sink_info
   */
  void set_partitions(std::vector<partition_info> partitions);

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata
   */
  void set_metadata(table_input_metadata const* metadata) { _metadata = metadata; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Key-Value footer metadata
   */
  void set_key_value_metadata(std::vector<std::map<std::string, std::string>> metadata);

  /**
   * @brief Sets the level of statistics.
   *
   * @param sf Level of statistics requested in the output file
   */
  void set_stats_level(statistics_freq sf) { _stats_level = sf; }

  /**
   * @brief Sets compression type.
   *
   * @param compression The compression type to use
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
   * @param file_paths Vector of Strings which indicates file path. Must be same size as number of
   * data sinks in sink info
   */
  void set_column_chunks_file_paths(std::vector<std::string> file_paths);

  /**
   * @brief Sets the maximum row group size, in bytes.
   *
   * @param size_bytes Maximum row group size, in bytes to set
   */
  void set_row_group_size_bytes(size_t size_bytes);

  /**
   * @brief Sets the maximum row group size, in rows.
   *
   * @param size_rows Maximum row group size, in rows to set
   */
  void set_row_group_size_rows(size_type size_rows);

  /**
   * @brief Sets the maximum uncompressed page size, in bytes.
   *
   * @param size_bytes Maximum uncompressed page size, in bytes to set
   */
  void set_max_page_size_bytes(size_t size_bytes);

  /**
   * @brief Sets the maximum page size, in rows.
   *
   * @param size_rows Maximum page size, in rows to set
   */
  void set_max_page_size_rows(size_type size_rows);

  /**
   * @brief Sets the maximum length of min or max values in column index, in bytes.
   *
   * @param size_bytes length min/max will be truncated to
   */
  void set_column_index_truncate_length(int32_t size_bytes);
};

/**
 * @brief Class to build `parquet_writer_options`.
 */
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
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit parquet_writer_options_builder(sink_info const& sink, table_view const& table)
    : options(sink, table)
  {
  }

  /**
   * @brief Sets partitions in parquet_writer_options.
   *
   * @param partitions Partitions of input table in {start_row, num_rows} pairs. If specified, must
   * be same size as number of sinks in sink_info
   * @return this for chaining
   */
  parquet_writer_options_builder& partitions(std::vector<partition_info> partitions);

  /**
   * @brief Sets metadata in parquet_writer_options.
   *
   * @param metadata Associated metadata
   * @return this for chaining
   */
  parquet_writer_options_builder& metadata(table_input_metadata const* metadata)
  {
    options._metadata = metadata;
    return *this;
  }

  /**
   * @brief Sets Key-Value footer metadata in parquet_writer_options.
   *
   * @param metadata Key-Value footer metadata
   * @return this for chaining
   */
  parquet_writer_options_builder& key_value_metadata(
    std::vector<std::map<std::string, std::string>> metadata);

  /**
   * @brief Sets the level of statistics in parquet_writer_options.
   *
   * @param sf Level of statistics requested in the output file
   * @return this for chaining
   */
  parquet_writer_options_builder& stats_level(statistics_freq sf)
  {
    options._stats_level = sf;
    return *this;
  }

  /**
   * @brief Sets compression type in parquet_writer_options.
   *
   * @param compression The compression type to use
   * @return this for chaining
   */
  parquet_writer_options_builder& compression(compression_type compression)
  {
    options._compression = compression;
    return *this;
  }

  /**
   * @brief Sets column chunks file path to be set in the raw output metadata.
   *
   * @param file_paths Vector of Strings which indicates file path. Must be same size as number of
   * data sinks
   * @return this for chaining
   */
  parquet_writer_options_builder& column_chunks_file_paths(std::vector<std::string> file_paths);

  /**
   * @brief Sets the maximum row group size, in bytes.
   *
   * @param val maximum row group size
   * @return this for chaining
   */
  parquet_writer_options_builder& row_group_size_bytes(size_t val)
  {
    options.set_row_group_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum number of rows in output row groups.
   *
   * @param val maximum number or rows
   * @return this for chaining
   */
  parquet_writer_options_builder& row_group_size_rows(size_type val)
  {
    options.set_row_group_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the maximum uncompressed page size, in bytes.
   *
   * Serves as a hint to the writer, * and can be exceeded under certain circumstances.
   * Cannot be larger than the row group size in bytes, and will be adjusted to
   * match if it is.
   *
   * @param val maximum page size
   * @return this for chaining
   */
  parquet_writer_options_builder& max_page_size_bytes(size_t val)
  {
    options.set_max_page_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum page size, in rows. Counts only top-level rows, ignoring any nesting.
   * Cannot be larger than the row group size in rows, and will be adjusted to match if it is.
   *
   * @param val maximum rows per page
   * @return this for chaining
   */
  parquet_writer_options_builder& max_page_size_rows(size_type val)
  {
    options.set_max_page_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the desired maximum size in bytes for min and max values in the column index.
   *
   * Values exceeding this limit will be truncated, but modified such that they will still
   * be valid lower and upper bounds. This only applies to variable length types, such as string.
   * Maximum values will not be truncated if there is no suitable truncation that results in
   * a valid upper bound.
   *
   * Default value is 64.
   *
   * @param val length min/max will be truncated to, with 0 indicating no truncation
   * @return this for chaining
   */
  parquet_writer_options_builder& column_index_truncate_length(int32_t val)
  {
    options.set_column_index_truncate_length(val);
    return *this;
  }

  /**
   * @brief Sets whether int96 timestamps are written or not in parquet_writer_options.
   *
   * @param enabled Boolean value to enable/disable int96 timestamps
   * @return this for chaining
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
   *
   * @return Built `parquet_writer_options` object's r-value reference
   */
  parquet_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to parquet format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  auto destination = cudf::io::sink_info("dataset.parquet");
 *  auto options     = cudf::io::parquet_writer_options::builder(destination, table->view());
 *  cudf::io::write_parquet(options);
 * @endcode
 *
 * @param options Settings for controlling writing behavior
 * @param mr Device memory resource to use for device memory allocation
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
 * @ingroup io_writers
 *
 * @param[in] metadata_list List of input file metadata
 * @return A parquet-compatible blob that contains the data for all row groups in the list
 */
std::unique_ptr<std::vector<uint8_t>> merge_row_group_metadata(
  const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list);

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
  // Optional footer key_value_metadata
  std::vector<std::map<std::string, std::string>> _user_data;
  // Parquet writer can write INT96 or TIMESTAMP_MICROS. Defaults to TIMESTAMP_MICROS.
  // If true then overrides any per-column setting in _metadata.
  bool _write_timestamps_as_int96 = false;
  // Maximum size of each row group (unless smaller than a single page)
  size_t _row_group_size_bytes = default_row_group_size_bytes;
  // Maximum number of rows in row group (unless smaller than a single page)
  size_type _row_group_size_rows = default_row_group_size_rows;
  // Maximum size of each page (uncompressed)
  size_t _max_page_size_bytes = default_max_page_size_bytes;
  // Maximum number of rows in a page
  size_type _max_page_size_rows = default_max_page_size_rows;
  // Maximum size of min or max values in column index
  int32_t _column_index_truncate_length = default_column_index_truncate_length;

  /**
   * @brief Constructor from sink.
   *
   * @param sink Sink used for writer output
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
   *
   * @return Sink info
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression format used.
   *
   * @return Compression format
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns level of statistics requested in output file.
   *
   * @return Level of statistics requested in output file
   */
  [[nodiscard]] statistics_freq get_stats_level() const { return _stats_level; }

  /**
   * @brief Returns metadata information.
   *
   * @return Metadata information
   */
  [[nodiscard]] table_input_metadata const* get_metadata() const { return _metadata; }

  /**
   * @brief Returns Key-Value footer metadata information.
   *
   * @return Key-Value footer metadata information
   */
  std::vector<std::map<std::string, std::string>> const& get_key_value_metadata() const
  {
    return _user_data;
  }

  /**
   * @brief Returns `true` if timestamps will be written as INT96
   *
   * @return `true` if timestamps will be written as INT96
   */
  bool is_enabled_int96_timestamps() const { return _write_timestamps_as_int96; }

  /**
   * @brief Returns maximum row group size, in bytes.
   *
   * @return Maximum row group size, in bytes
   */
  auto get_row_group_size_bytes() const { return _row_group_size_bytes; }

  /**
   * @brief Returns maximum row group size, in rows.
   *
   * @return Maximum row group size, in rows
   */
  auto get_row_group_size_rows() const { return _row_group_size_rows; }

  /**
   * @brief Returns maximum uncompressed page size, in bytes.
   *
   * If set larger than the row group size, then this will return the
   * row group size.
   *
   * @return Maximum uncompressed page size, in bytes
   */
  auto get_max_page_size_bytes() const
  {
    return std::min(_max_page_size_bytes, get_row_group_size_bytes());
  }

  /**
   * @brief Returns maximum page size, in rows.
   *
   * If set larger than the row group size, then this will return the row group size.
   *
   * @return Maximum page size, in rows
   */
  auto get_max_page_size_rows() const
  {
    return std::min(_max_page_size_rows, get_row_group_size_rows());
  }

  /**
   * @brief Returns maximum length of min or max values in column index, in bytes.
   *
   * @return length min/max will be truncated to
   */
  auto get_column_index_truncate_length() const { return _column_index_truncate_length; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata
   */
  void set_metadata(table_input_metadata const* metadata) { _metadata = metadata; }

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   */
  void set_key_value_metadata(std::vector<std::map<std::string, std::string>> metadata);

  /**
   * @brief Sets the level of statistics in parquet_writer_options.
   *
   * @param sf Level of statistics requested in the output file
   */
  void set_stats_level(statistics_freq sf) { _stats_level = sf; }

  /**
   * @brief Sets compression type.
   *
   * @param compression The compression type to use
   */
  void set_compression(compression_type compression) { _compression = compression; }

  /**
   * @brief Sets timestamp writing preferences.
   *
   * INT96 timestamps will be written if `true` and TIMESTAMP_MICROS will be written if `false`.
   *
   * @param req Boolean value to enable/disable writing of INT96 timestamps
   */
  void enable_int96_timestamps(bool req) { _write_timestamps_as_int96 = req; }

  /**
   * @brief Sets the maximum row group size, in bytes.
   *
   * @param size_bytes Maximum row group size, in bytes to set
   */
  void set_row_group_size_bytes(size_t size_bytes);

  /**
   * @brief Sets the maximum row group size, in rows.
   *
   * @param size_rows The maximum row group size, in rows to set
   */
  void set_row_group_size_rows(size_type size_rows);

  /**
   * @brief Sets the maximum uncompressed page size, in bytes.
   *
   * @param size_bytes Maximum uncompressed page size, in bytes to set
   */
  void set_max_page_size_bytes(size_t size_bytes);

  /**
   * @brief Sets the maximum page size, in rows.
   *
   * @param size_rows The maximum page size, in rows to set
   */
  void set_max_page_size_rows(size_type size_rows);

  /**
   * @brief Sets the maximum length of min or max values in column index, in bytes.
   *
   * @param size_bytes length min/max will be truncated to
   */
  void set_column_index_truncate_length(int32_t size_bytes);

  /**
   * @brief creates builder to build chunked_parquet_writer_options.
   *
   * @param sink sink to use for writer output
   *
   * @return Builder to build `chunked_parquet_writer_options`
   */
  static chunked_parquet_writer_options_builder builder(sink_info const& sink);
};

/**
 * @brief Builds options for chunked_parquet_writer_options.
 */
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
   * @param sink The sink used for writer output
   */
  chunked_parquet_writer_options_builder(sink_info const& sink) : options(sink){};

  /**
   * @brief Sets metadata to chunked_parquet_writer_options.
   *
   * @param metadata Associated metadata
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& metadata(table_input_metadata const* metadata)
  {
    options._metadata = metadata;
    return *this;
  }

  /**
   * @brief Sets Key-Value footer metadata in parquet_writer_options.
   *
   * @param metadata Key-Value footer metadata
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& key_value_metadata(
    std::vector<std::map<std::string, std::string>> metadata);

  /**
   * @brief Sets Sets the level of statistics in chunked_parquet_writer_options.
   *
   * @param sf Level of statistics requested in the output file
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& stats_level(statistics_freq sf)
  {
    options._stats_level = sf;
    return *this;
  }

  /**
   * @brief Sets compression type to chunked_parquet_writer_options.
   *
   * @param compression The compression type to use
   * @return this for chaining
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
   * compatibility reasons.
   *
   * @param enabled Boolean value to enable/disable int96 timestamps
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& int96_timestamps(bool enabled)
  {
    options._write_timestamps_as_int96 = enabled;
    return *this;
  }

  /**
   * @brief Sets the maximum row group size, in bytes.
   *
   * @param val maximum row group size
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& row_group_size_bytes(size_t val)
  {
    options.set_row_group_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum number of rows in output row groups.
   *
   * @param val maximum number or rows
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& row_group_size_rows(size_type val)
  {
    options.set_row_group_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the maximum uncompressed page size, in bytes.
   *
   * Serves as a hint to the writer, and can be exceeded under certain circumstances. Cannot be
   * larger than the row group size in bytes, and will be adjusted to match if it is.
   *
   * @param val maximum page size
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& max_page_size_bytes(size_t val)
  {
    options.set_max_page_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum page size, in rows. Counts only top-level rows, ignoring any nesting.
   * Cannot be larger than the row group size in rows, and will be adjusted to match if it is.
   *
   * @param val maximum rows per page
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& max_page_size_rows(size_type val)
  {
    options.set_max_page_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the desired maximum size in bytes for min and max values in the column index.
   *
   * Values exceeding this limit will be truncated, but modified such that they will still
   * be valid lower and upper bounds. This only applies to variable length types, such as string.
   * Maximum values will not be truncated if there is no suitable truncation that results in
   * a valid upper bound.
   *
   * Default value is 64.
   *
   * @param val length min/max will be truncated to, with 0 indicating no truncation
   * @return this for chaining
   */
  chunked_parquet_writer_options_builder& column_index_truncate_length(int32_t val)
  {
    options.set_column_index_truncate_length(val);
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
   *
   * @return Built `chunked_parquet_writer_options` object's r-value reference
   */
  chunked_parquet_writer_options&& build() { return std::move(options); }
};

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
 *  auto destination = cudf::io::sink_info("dataset.parquet");
 *  auto options = cudf::io::chunked_parquet_writer_options::builder(destination, table->view());
 *  auto writer  = cudf::io::parquet_chunked_writer(options);
 *
 *  writer.write(table0)
 *  writer.write(table1)
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
   * @param[in] options options used to write table
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  parquet_chunked_writer(
    chunked_parquet_writer_options const& options,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Writes table to output.
   *
   * @param[in] table Table that needs to be written
   * @param[in] partitions Optional partitions to divide the table into. If specified, must be same
   * size as number of sinks.
   *
   * @throws cudf::logic_error If the number of partitions is not the same as number of sinks
   * @return returns reference of the class object
   */
  parquet_chunked_writer& write(table_view const& table,
                                std::vector<partition_info> const& partitions = {});

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] column_chunks_file_paths Column chunks file path to be set in the raw output
   * metadata
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list only if
   * `column_chunks_file_paths` is provided, else null.
   */
  std::unique_ptr<std::vector<uint8_t>> close(
    std::vector<std::string> const& column_chunks_file_paths = {});

  /// Unique pointer to impl writer class
  std::unique_ptr<cudf::io::detail::parquet::writer> writer;
};

/** @} */  // end of group

}  // namespace cudf::io
