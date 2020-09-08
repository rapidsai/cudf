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
 * @file functions.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include "types.hpp"

#include <cudf/io/writers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/default_memory_resource.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace io {
/**
 * @brief Builds settings to use for `read_orc()`.
 *
 * @ingroup io_readers
 */
class orc_reader_options_builder;

/**
 * @brief Settings to use for `read_orc()`.
 *
 * @ingroup io_readers
 */
class orc_reader_options {
  source_info _source;

  // Names of column to read; empty is all
  std::vector<std::string> _columns;

  // List of individual stripes to read (ignored if empty)
  std::vector<size_type> _stripes;
  // Rows to skip from the start;
  size_type _skip_rows = 0;
  // Rows to read; -1 is all
  size_type _num_rows = -1;

  // Whether to use row index to speed-up reading
  bool _use_index = true;

  // Whether to use numpy-compatible dtypes
  bool _use_np_dtypes = true;
  // Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  // Whether to convert decimals to float64
  bool _decimals_as_float = true;
  // For decimals as int, optional forced decimal scale;
  // -1 is auto (column scale), >=0: number of fractional digits
  size_type _forced_decimals_scale = -1;

  friend orc_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read orc file.
   */
  explicit orc_reader_options(source_info const& src) : _source(src) {}

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  orc_reader_options() = default;

  /**
   * @brief Creates `orc_reader_options_builder` which will build `orc_reader_options`.
   *
   * @param src Source information to read orc file.
   * @return Builder to build reader options.
   */
  static orc_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info.
   */
  source_info const& get_source() const { return _source; }

  /**
   * @brief Returns names of the columns to read.
   */
  std::vector<std::string> const& get_columns() const { return _columns; }

  /**
   * @brief Returns list of individual stripes to read.
   */
  std::vector<size_type> const& get_stripes() const { return _stripes; }

  /**
   * @brief Returns number of rows to skip from the start.
   */
  size_type get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of row to read.
   */
  size_type get_num_rows() const { return _num_rows; }

  /**
   * @brief Whether to use row index to speed-up reading.
   */
  bool is_enabled_use_index() const { return _use_index; }

  /**
   * @brief Whether to use numpy-compatible dtypes.
   */
  bool is_enabled_use_np_dtypes() const { return _use_np_dtypes; }

  /**
   * @brief Returns timestamp type to which timestamp column will be cast.
   */
  data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Whether to convert decimals to float64.
   */
  bool is_enabled_decimals_as_float() const { return _decimals_as_float; }

  /**
   * @brief Returns whether decimal scale is inferred or forced to have limited fractional digits.
   */
  size_type get_forced_decimals_scale() const { return _forced_decimals_scale; }

  // Setters

  /**
   * @brief Sets names of the column to read.
   *
   * @param col_names Vector of column names.
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }

  /**
   * @brief Sets list of individual stripes to read.
   *
   * @param strps Vector of individual stripes.
   */
  void set_stripes(std::vector<size_type> strps)
  {
    CUDF_EXPECTS(strps.empty() or (_skip_rows == 0 and _num_rows == -1),
                 "Can't set both stripes along with skip_rows/num_rows");
    _stripes = std::move(strps);
  }

  /**
   * @brief Sets number of rows to skip from the start.
   *
   * @param rows Number of rows.
   */
  void set_skip_rows(size_type rows)
  {
    CUDF_EXPECTS(rows == 0 or _stripes.empty(), "Can't set both skip_rows along with stripes");
    _skip_rows = rows;
  }

  /**
   * @brief Sets number of row to read.
   *
   * @param nrows Number of rows.
   */
  void set_num_rows(size_type nrows)
  {
    CUDF_EXPECTS(nrows == -1 or _stripes.empty(), "Can't set both num_rows along with stripes");
    _num_rows = (nrows != 0) ? nrows : -1;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading.
   *
   * @param use Boolean value to enable/disable row index use.
   */
  void set_use_index(bool use) { _use_index = use; }

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes
   *
   * @param rows Boolean value to enable/disable.
   */
  void set_use_np_dtypes(bool use) { _use_np_dtypes = use; }

  /**
   * @brief Sets timestamp type to which timestamp column will be cast.
   *
   * @param type Type of timestamp.
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }

  /**
   * @brief Enable/Disable conversion of decimals to float64.
   *
   * @param val Boolean value to enable/disable.
   */
  void set_decimals_as_float(bool val) { _decimals_as_float = val; }

  /**
   * @brief Sets whether decimal scale is inferred or forced to have limited fractional digits.
   *
   * @param val Length of fractional digits.
   */
  void set_forced_decimals_scale(size_type val) { _forced_decimals_scale = val; }
};

class orc_reader_options_builder {
  orc_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit orc_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read orc file.
   */
  explicit orc_reader_options_builder(source_info const& src) : options{src} {};

  /**
   * @brief Sets names of the column to read.
   *
   * @param col_names Vector of column names.
   * @return this for chaining.
   */
  orc_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options._columns = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets list of individual stripes to read.
   *
   * @param strps Vector of individual stripes.
   * @return this for chaining.
   */
  orc_reader_options_builder& stripes(std::vector<size_type> strps)
  {
    options.set_stripes(std::move(strps));
    return *this;
  }

  /**
   * @brief Sets number of rows to skip from the start.
   *
   * @param rows Number of rows.
   * @return this for chaining.
   */
  orc_reader_options_builder& skip_rows(size_type rows)
  {
    options.set_skip_rows(rows);
    return *this;
  }

  /**
   * @brief Sets number of row to read.
   *
   * @param nrows Number of rows.
   * @return this for chaining.
   */
  orc_reader_options_builder& num_rows(size_type nrows)
  {
    options.set_num_rows(nrows);
    return *this;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading.
   *
   * @param use Boolean value to enable/disable row index use.
   * @return this for chaining.
   */
  orc_reader_options_builder& use_index(bool use)
  {
    options._use_index = use;
    return *this;
  }

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes.
   *
   * @param rows Boolean value to enable/disable.
   * @return this for chaining.
   */
  orc_reader_options_builder& use_np_dtypes(bool use)
  {
    options._use_np_dtypes = use;
    return *this;
  }

  /**
   * @brief Sets timestamp type to which timestamp column will be cast.
   *
   * @param type Type of timestamp.
   * @return this for chaining.
   */
  orc_reader_options_builder& timestamp_type(data_type type)
  {
    options._timestamp_type = type;
    return *this;
  }

  /**
   * @brief Enable/Disable conversion of decimals to float64.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  orc_reader_options_builder& decimals_as_float(bool val)
  {
    options._decimals_as_float = val;
    return *this;
  }

  /**
   * @brief Sets whether decimal scale is inferred or forced to have limited fractional digits.
   *
   * @param val Length of fractional digits.
   * @return this for chaining.
   */
  orc_reader_options_builder& forced_decimals_scale(size_type val)
  {
    options._forced_decimals_scale = val;
    return *this;
  }

  /**
   * @brief move orc_reader_options member once it's built.
   */
  operator orc_reader_options &&() { return std::move(options); }

  /**
   * @brief move orc_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  orc_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads an ORC dataset into a set of columns.
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::orc_reader_options options =
 * cudf::orc_reader_options::builder(cudf::source_info(filepath));
 *  ...
 *  auto result = cudf::read_orc(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior.
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata.
 *
 * @return The set of columns.
 */
table_with_metadata read_orc(orc_reader_options const& options,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Builds settings to use for `write_orc()`.
 *
 * @ingroup io_writers
 */
class orc_writer_options_builder;

/**
 * @brief Settings to use for `write_orc()`.
 *
 * @ingroup io_writers
 */
class orc_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  // Enable writing column statistics
  bool _enable_statistics = true;
  // Set of columns to output
  table_view _table;
  // Optional associated metadata
  const table_metadata* _metadata = nullptr;

  friend orc_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   */
  explicit orc_writer_options(sink_info const& sink, table_view const& table)
    : _sink(sink), _table(table)
  {
  }

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit orc_writer_options() = default;

  /**
   * @brief Create builder to create `orc_writer_options`.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   *
   * @return Builder to build `orc_writer_options`.
   */
  static orc_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink info.
   */
  sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression type.
   */
  compression_type get_compression() const { return _compression; }

  /**
   * @brief Whether writing column statistics is enabled/disabled.
   */
  bool enable_statistics() const { return _enable_statistics; }

  /**
   * @brief Returns table to be written to output.
   */
  table_view get_table() const { return _table; }

  /**
   * @brief Returns associated metadata.
   */
  table_metadata const* get_metadata() const { return _metadata; }

  // Setters

  /**
   * @brief Sets compression type.
   *
   * @param comp Compression type.
   */
  void set_compression(compression_type comp) { _compression = comp; }

  /**
   * @brief Enable/Disable writing column statistics.
   *
   * @param val Boolean value to enable/disable statistics.
   */
  void enable_statistics(bool val) { _enable_statistics = val; }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output.
   */
  void set_table(table_view tbl) { _table = tbl; }

  /**
   * @brief Sets associated metadata
   *
   * @param meta Associated metadata.
   */
  void set_metadata(table_metadata* meta) { _metadata = meta; }
};

class orc_writer_options_builder {
  orc_writer_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  orc_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   */
  orc_writer_options_builder(sink_info const& sink, table_view const& table) : options{sink, table}
  {
  }

  /**
   * @brief Sets compression type.
   *
   * @param compression The compression type to use.
   * @return this for chaining.
   */
  orc_writer_options_builder& compression(compression_type comp)
  {
    options._compression = comp;
    return *this;
  }

  /**
   * @brief Enable/Disable writing column statistics.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  orc_writer_options_builder& enable_statistics(bool val)
  {
    options._enable_statistics = val;
    return *this;
  }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output.
   * @return this for chaining.
   */
  orc_writer_options_builder& table(table_view tbl)
  {
    options._table = tbl;
    return *this;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata.
   * @return this for chaining.
   */
  orc_writer_options_builder& metadata(table_metadata* meta)
  {
    options._metadata = meta;
    return *this;
  }

  /**
   * @brief move orc_writer_options member once it's built.
   */
  operator orc_writer_options &&() { return std::move(options); }

  /**
   * @brief move orc_writer_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  orc_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to ORC format.
 *
 * @ingroup io_writers
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::orc_writer_options options = cudf::orc_writer_options::builder(cudf::sink_info(filepath),
 * table->view());
 *  ...
 *  cudf::write_orc(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior.
 * @param mr Device memory resource to use for device memory allocation.
 */
void write_orc(orc_writer_options const& options,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Builds settings to use for `write_orc_chunked()`.
 *
 * @ingroup io_writers
 */
class chunked_orc_writer_options_builder;

/**
 * @brief Settings to use for `write_orc_chunked()`.
 *
 * @ingroup io_writers
 */
class chunked_orc_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  // Enable writing column statistics
  bool _enable_statistics = true;
  // Optional associated metadata
  const table_metadata_with_nullability* _metadata = nullptr;

  friend chunked_orc_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   */
  chunked_orc_writer_options(sink_info const& sink) : _sink(sink) {}

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit chunked_orc_writer_options() = default;

  /**
   * @brief Create builder to create `chunked_orc_writer_options`.
   *
   * @param sink The sink used for writer output.
   *
   * @return Builder to build chunked_orc_writer_options.
   */
  static chunked_orc_writer_options_builder builder(sink_info const& sink);

  /**
   * @brief Returns sink info.
   */
  sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression type.
   */
  compression_type get_compression() const { return _compression; }

  /**
   * @brief Whether writing column statistics is enabled/disabled.
   */
  bool enable_statistics() const { return _enable_statistics; }

  /**
   * @brief Returns associated metadata.
   */
  table_metadata_with_nullability const* get_metadata() const { return _metadata; }

  // Setters

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use.
   */
  void set_compression(compression_type comp) { _compression = comp; }

  /**
   * @brief Enable/Disable writing column statistics.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_statistics(bool val) { _enable_statistics = val; }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata.
   */
  void metadata(table_metadata_with_nullability* meta) { _metadata = meta; }
};

class chunked_orc_writer_options_builder {
  chunked_orc_writer_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  chunked_orc_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   */
  explicit chunked_orc_writer_options_builder(sink_info const& sink) : options{sink} {}

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use.
   * @return this for chaining.
   */
  chunked_orc_writer_options_builder& compression(compression_type comp)
  {
    options._compression = comp;
    return *this;
  }

  /**
   * @brief Enable/Disable writing column statistics.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  chunked_orc_writer_options_builder& enable_statistics(bool val)
  {
    options._enable_statistics = val;
    return *this;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata.
   * @return this for chaining.
   */
  chunked_orc_writer_options_builder& metadata(table_metadata_with_nullability* meta)
  {
    options._metadata = meta;
    return *this;
  }

  /**
   * @brief move chunked_orc_writer_options member once it's built.
   */
  operator chunked_orc_writer_options &&() { return std::move(options); }

  /**
   * @brief move chunked_orc_writer_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  chunked_orc_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Forward declaration of anonymous chunked-writer state struct.
 */
struct orc_chunked_state;

/**
 * @brief Begin the process of writing an ORC file in a chunked/stream form.
 *
 * @ingroup io_writers
 *
 * The intent of the write_orc_chunked_ path is to allow writing of an
 * arbitrarily large / arbitrary number of rows to an ORC file in multiple passes.
 *
 * The following code snippet demonstrates how to write a single ORC file containing
 * one logical table by writing a series of individual cudf::tables.
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::io::chunked_orc_writer_options options = cudf::io::chunked_orc_writer_options
 * options::builder(cudf::sink_info(filepath));
 *  ...
 *  auto state = cudf::write_orc_chunked_begin(options);
 *    cudf::write_orc_chunked(table0, state);
 *    cudf::write_orc_chunked(table1, state);
 *    ...
 *  cudf_write_orc_chunked_end(state);
 * @endcode
 *
 * @param[in] options Settings for controlling writing behavior.
 * @param[in] mr Device memory resource to use for device memory allocation.
 *
 * @returns pointer to an anonymous state structure storing information about the chunked write.
 * this pointer must be passed to all subsequent write_orc_chunked() and write_orc_chunked_end()
 *          calls.
 */
std::shared_ptr<orc_chunked_state> write_orc_chunked_begin(
  chunked_orc_writer_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Write a single table as a subtable of a larger logical orc file/table.
 *
 * @ingroup io_writers
 *
 * All tables passed into multiple calls of this function must contain the same # of columns and
 * have columns of the same type.
 *
 * @param[in] table The table data to be written.
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_orc_chunked_begin().
 */
void write_orc_chunked(table_view const& table, std::shared_ptr<orc_chunked_state> state);

/**
 * @brief Finish writing a chunked/stream orc file.
 *
 * @ingroup io_writers
 *
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_orc_chunked_begin().
 */
void write_orc_chunked_end(std::shared_ptr<orc_chunked_state>& state);

}  // namespace io
}  // namespace cudf
