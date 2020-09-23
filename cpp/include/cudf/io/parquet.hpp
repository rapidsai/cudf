/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
 * @file parquet.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {

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
   * @brief move parquet_reader_options member once it's built.
   */
  operator parquet_reader_options &&() { return std::move(options); }

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
 * @ingroup io_readers
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

/**
 * @brief Class to build `parquet_writer_options`.
 *
 * @ingroup io_writers
 */
class parquet_writer_options_builder;

/**
 * @brief Settings for `write_parquet()`.
 *
 * @ingroup io_writers
 */
class parquet_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  // Specify the level of statistics in the output file
  statistics_freq _stats_level = statistics_freq::STATISTICS_ROWGROUP;
  // Sets of columns to output
  table_view _table;
  // Optional associated metadata
  const table_metadata* _metadata = nullptr;
  // Optionally return the raw parquet file metadata output
  bool _return_filemetadata = false;
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
  table_metadata const* get_metadata() const { return _metadata; }

  /**
   * @brief Returns `true` if metadata is required, `false` otherwise.
   */
  bool is_enabled_return_filemetadata() const { return _return_filemetadata; }

  /**
   * @brief Returns Column chunks file path to be set in the raw output metadata.
   */
  std::string get_column_chunks_file_path() const { return _column_chunks_file_path; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata.
   */
  void set_metadata(table_metadata const* metadata) { _metadata = metadata; }

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
   * @brief Sets whether filemetadata is required or not.
   *
   * @param req Boolean value to enable/disable return of file metadata.
   */
  void enable_return_filemetadata(bool req) { _return_filemetadata = req; }

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
  parquet_writer_options_builder& metadata(table_metadata const* metadata)
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
   * @brief Sets whether filemetadata is required or not in parquet_writer_options.
   *
   * @param req Boolean value to enable/disable return of file metadata.
   * @return this for chaining.
   */
  parquet_writer_options_builder& return_filemetadata(bool req)
  {
    options._return_filemetadata = req;
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
   * @brief move parquet_writer_options member once it's built.
   */
  operator parquet_writer_options &&() { return std::move(options); }

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
 * @ingroup io_writers
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
 * @ingroup io_writers
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
 *
 * @ingroup io_writers
 */
class chunked_parquet_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  // Specify the level of statistics in the output file
  statistics_freq _stats_level = statistics_freq::STATISTICS_ROWGROUP;
  // Optional associated metadata.
  const table_metadata_with_nullability* _nullable_metadata = nullptr;

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
   * @brief Returns nullable metadata information.
   */
  const table_metadata_with_nullability* get_nullable_metadata() const
  {
    return _nullable_metadata;
  }

  /**
   * @brief Sets nullable metadata.
   *
   * @param metadata Associated metadata.
   */
  void set_nullable_metadata(const table_metadata_with_nullability* metadata)
  {
    _nullable_metadata = metadata;
  }

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
   * @brief Sets nullable metadata to chunked_parquet_writer_options.
   *
   * @param metadata Associated metadata.
   * @return this for chaining.
   */
  chunked_parquet_writer_options_builder& nullable_metadata(
    const table_metadata_with_nullability* metadata)
  {
    options._nullable_metadata = metadata;
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
   * @brief move chunked_parquet_writer_options member once it's built.
   */
  operator chunked_parquet_writer_options &&() { return std::move(options); }

  /**
   * @brief move chunked_parquet_writer_options member once it's is built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  chunked_parquet_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Forward declaration of anonymous chunked-writer state struct.
 */
struct pq_chunked_state;

/**
 * @brief Begin the process of writing a parquet file in a chunked/stream form.
 *
 * @ingroup io_writers
 *
 * The intent of the write_parquet_chunked_ path is to allow writing of an
 * arbitrarily large / arbitrary number of rows to a parquet file in multiple passes.
 *
 * The following code snippet demonstrates how to write a single parquet file containing
 * one logical table by writing a series of individual cudf::tables.
 * @code
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::io::chunked_parquet_writer_options options =
 *  cudf::io::chunked_parquet_writer_options::builder(cudf::sink_info(filepath), table->view());
 *  ...
 *  auto state = cudf::write_parquet_chunked_begin(options);
 *    cudf::write_parquet_chunked(table0, state);
 *    cudf::write_parquet_chunked(table1, state);
 *    ...
 *  cudf_write_parquet_chunked_end(state);
 * @endcode
 *
 * @param[in] options Settings for controlling writing behavior.
 * @param[in] mr Device memory resource to use for device memory allocation.
 *
 * @return pointer to an anonymous state structure storing information about the chunked write.
 * this pointer must be passed to all subsequent write_parquet_chunked() and
 * write_parquet_chunked_end() calls.
 */
std::shared_ptr<pq_chunked_state> write_parquet_chunked_begin(
  chunked_parquet_writer_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Write a single table as a subtable of a larger logical parquet file/table.
 *
 * @ingroup io_writers
 *
 * All tables passed into multiple calls of this function must contain the same # of columns and
 * have columns of the same type.
 *
 * @param[in] table The table data to be written.
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_parquet_chunked_begin().
 */
void write_parquet_chunked(table_view const& table, std::shared_ptr<pq_chunked_state> state);

/**
 * @brief Finish writing a chunked/stream parquet file.
 *
 * @ingroup io_writers
 *
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_parquet_chunked_begin().
 * @param[in] return_filemetadata If true, return the raw file metadata.
 * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output metadata.
 *
 * @return A blob that contains the file metadata (parquet FileMetadata thrift message) if
 *         requested in parquet_writer_options (empty blob otherwise).
 */
std::unique_ptr<std::vector<uint8_t>> write_parquet_chunked_end(
  std::shared_ptr<pq_chunked_state>& state,
  bool return_filemetadata                   = false,
  const std::string& column_chunks_file_path = "");

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
}  // namespace io
}  // namespace cudf
