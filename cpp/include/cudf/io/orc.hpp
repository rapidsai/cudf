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

#include <cudf/io/detail/orc.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

constexpr size_t default_stripe_size_bytes   = 64 * 1024 * 1024;  ///< 64MB default orc stripe size
constexpr size_type default_stripe_size_rows = 1000000;  ///< 1M rows default orc stripe rows
constexpr size_type default_row_index_stride = 10000;    ///< 10K rows default orc row index stride

/**
 * @brief Builds settings to use for `read_orc()`.
 */
class orc_reader_options_builder;

/**
 * @brief Settings to use for `read_orc()`.
 */
class orc_reader_options {
  source_info _source;

  // Names of column to read; `nullopt` is all
  std::optional<std::vector<std::string>> _columns;

  // List of individual stripes to read (ignored if empty)
  std::vector<std::vector<size_type>> _stripes;
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

  // Columns that should be read as Decimal128
  std::vector<std::string> _decimal128_columns;

  friend orc_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read orc file
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
   * @param src Source information to read orc file
   * @return Builder to build reader options
   */
  static orc_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns names of the columns to read, if set.
   *
   * @return Names of the columns to read; `nullopt` if the option is not set
   */
  [[nodiscard]] auto const& get_columns() const { return _columns; }

  /**
   * @brief Returns vector of vectors, stripes to read for each input source
   *
   * @return Vector of vectors, stripes to read for each input source
   */
  [[nodiscard]] auto const& get_stripes() const { return _stripes; }

  /**
   * @brief Returns number of rows to skip from the start.
   *
   * @return Number of rows to skip from the start
   */
  size_type get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of row to read.
   *
   * @return Number of row to read
   */
  size_type get_num_rows() const { return _num_rows; }

  /**
   * @brief Whether to use row index to speed-up reading.
   *
   * @return `true` if row index is used to speed-up reading
   */
  bool is_enabled_use_index() const { return _use_index; }

  /**
   * @brief Whether to use numpy-compatible dtypes.
   *
   * @return `true` if numpy-compatible dtypes are used
   */
  bool is_enabled_use_np_dtypes() const { return _use_np_dtypes; }

  /**
   * @brief Returns timestamp type to which timestamp column will be cast.
   *
   * @return Timestamp type to which timestamp column will be cast
   */
  data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Returns fully qualified names of columns that should be read as 128-bit Decimal.
   *
   * @return Fully qualified names of columns that should be read as 128-bit Decimal
   */
  std::vector<std::string> const& get_decimal128_columns() const { return _decimal128_columns; }

  // Setters

  /**
   * @brief Sets names of the column to read.
   *
   * @param col_names Vector of column names
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }

  /**
   * @brief Sets list of stripes to read for each input source
   *
   * @param stripes Vector of vectors, mapping stripes to read to input sources
   */
  void set_stripes(std::vector<std::vector<size_type>> stripes)
  {
    CUDF_EXPECTS(stripes.empty() or (_skip_rows == 0), "Can't set stripes along with skip_rows");
    CUDF_EXPECTS(stripes.empty() or (_num_rows == -1), "Can't set stripes along with num_rows");
    _stripes = std::move(stripes);
  }

  /**
   * @brief Sets number of rows to skip from the start.
   *
   * @param rows Number of rows
   */
  void set_skip_rows(size_type rows)
  {
    CUDF_EXPECTS(rows == 0 or _stripes.empty(), "Can't set both skip_rows along with stripes");
    _skip_rows = rows;
  }

  /**
   * @brief Sets number of row to read.
   *
   * @param nrows Number of rows
   */
  void set_num_rows(size_type nrows)
  {
    CUDF_EXPECTS(nrows == -1 or _stripes.empty(), "Can't set both num_rows along with stripes");
    _num_rows = nrows;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading.
   *
   * @param use Boolean value to enable/disable row index use
   */
  void enable_use_index(bool use) { _use_index = use; }

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes
   *
   * @param use Boolean value to enable/disable
   */
  void enable_use_np_dtypes(bool use) { _use_np_dtypes = use; }

  /**
   * @brief Sets timestamp type to which timestamp column will be cast.
   *
   * @param type Type of timestamp
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }

  /**
   * @brief Set columns that should be read as 128-bit Decimal
   *
   * @param val Vector of fully qualified column names
   */
  void set_decimal128_columns(std::vector<std::string> val)
  {
    _decimal128_columns = std::move(val);
  }
};

/**
 * @brief Builds settings to use for `read_orc()`.
 */
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
   * @param src The source information used to read orc file
   */
  explicit orc_reader_options_builder(source_info const& src) : options{src} {};

  /**
   * @brief Sets names of the column to read.
   *
   * @param col_names Vector of column names
   * @return this for chaining
   */
  orc_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options._columns = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets list of individual stripes to read per source
   *
   * @param stripes Vector of vectors, mapping stripes to read to input sources
   * @return this for chaining
   */
  orc_reader_options_builder& stripes(std::vector<std::vector<size_type>> stripes)
  {
    options.set_stripes(std::move(stripes));
    return *this;
  }

  /**
   * @brief Sets number of rows to skip from the start.
   *
   * @param rows Number of rows
   * @return this for chaining
   */
  orc_reader_options_builder& skip_rows(size_type rows)
  {
    options.set_skip_rows(rows);
    return *this;
  }

  /**
   * @brief Sets number of row to read.
   *
   * @param nrows Number of rows
   * @return this for chaining
   */
  orc_reader_options_builder& num_rows(size_type nrows)
  {
    options.set_num_rows(nrows);
    return *this;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading.
   *
   * @param use Boolean value to enable/disable row index use
   * @return this for chaining
   */
  orc_reader_options_builder& use_index(bool use)
  {
    options._use_index = use;
    return *this;
  }

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes.
   *
   * @param use Boolean value to enable/disable
   * @return this for chaining
   */
  orc_reader_options_builder& use_np_dtypes(bool use)
  {
    options._use_np_dtypes = use;
    return *this;
  }

  /**
   * @brief Sets timestamp type to which timestamp column will be cast.
   *
   * @param type Type of timestamp
   * @return this for chaining
   */
  orc_reader_options_builder& timestamp_type(data_type type)
  {
    options._timestamp_type = type;
    return *this;
  }

  /**
   * @brief Columns that should be read as 128-bit Decimal
   *
   * @param val Vector of column names
   * @return this for chaining
   */
  orc_reader_options_builder& decimal128_columns(std::vector<std::string> val)
  {
    options._decimal128_columns = std::move(val);
    return *this;
  }

  /**
   * @brief move orc_reader_options member once it's built.
   */
  operator orc_reader_options&&() { return std::move(options); }

  /**
   * @brief move orc_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `orc_reader_options` object's r-value reference
   */
  orc_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads an ORC dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.orc");
 *  auto options = cudf::io::orc_reader_options::builder(source);
 *  auto result  = cudf::io::read_orc(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata.
 *
 * @return The set of columns
 */
table_with_metadata read_orc(
  orc_reader_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
/**
 * @addtogroup io_writers
 * @{
 * @file
 */

/**
 * @brief Builds settings to use for `write_orc()`.
 */
class orc_writer_options_builder;

/**
 * @brief Constants to disambiguate statistics terminology for ORC.
 *
 * ORC refers to its finest granularity of row-grouping as "row group",
 * which corresponds to Parquet "pages".
 * Similarly, ORC's "stripe" corresponds to a Parquet "row group".
 * The following constants disambiguate the terminology for the statistics
 * collected at each level.
 */
static constexpr statistics_freq ORC_STATISTICS_STRIPE    = statistics_freq::STATISTICS_ROWGROUP;
static constexpr statistics_freq ORC_STATISTICS_ROW_GROUP = statistics_freq::STATISTICS_PAGE;

/**
 * @brief Settings to use for `write_orc()`.
 */
class orc_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  // Specify frequency of statistics collection
  statistics_freq _stats_freq = ORC_STATISTICS_ROW_GROUP;
  // Maximum size of each stripe (unless smaller than a single row group)
  size_t _stripe_size_bytes = default_stripe_size_bytes;
  // Maximum number of rows in stripe (unless smaller than a single row group)
  size_type _stripe_size_rows = default_stripe_size_rows;
  // Row index stride (maximum number of rows in each row group)
  size_type _row_index_stride = default_row_index_stride;
  // Set of columns to output
  table_view _table;
  // Optional associated metadata
  const table_input_metadata* _metadata = nullptr;
  // Optional footer key_value_metadata
  std::map<std::string, std::string> _user_data;

  friend orc_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
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
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build `orc_writer_options`
   */
  static orc_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink info.
   *
   * @return Sink info
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression type.
   *
   * @return Compression type
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Whether writing column statistics is enabled/disabled.
   *
   * @return `true` if writing column statistics is enabled
   */
  [[nodiscard]] bool is_enabled_statistics() const
  {
    return _stats_freq != statistics_freq::STATISTICS_NONE;
  }

  /**
   * @brief Returns frequency of statistics collection.
   *
   * @return Frequency of statistics collection
   */
  [[nodiscard]] statistics_freq get_statistics_freq() const { return _stats_freq; }

  /**
   * @brief Returns maximum stripe size, in bytes.
   *
   * @return Maximum stripe size, in bytes
   */
  [[nodiscard]] auto get_stripe_size_bytes() const { return _stripe_size_bytes; }

  /**
   * @brief Returns maximum stripe size, in rows.
   *
   * @return Maximum stripe size, in rows
   */
  [[nodiscard]] auto get_stripe_size_rows() const { return _stripe_size_rows; }

  /**
   * @brief Returns the row index stride.
   *
   * @return Row index stride
   */
  auto get_row_index_stride() const
  {
    auto const unaligned_stride = std::min(_row_index_stride, get_stripe_size_rows());
    return unaligned_stride - unaligned_stride % 8;
  }

  /**
   * @brief Returns table to be written to output.
   *
   * @return Table to be written to output
   */
  [[nodiscard]] table_view get_table() const { return _table; }

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
  [[nodiscard]] std::map<std::string, std::string> const& get_key_value_metadata() const
  {
    return _user_data;
  }

  // Setters

  /**
   * @brief Sets compression type.
   *
   * @param comp Compression type
   */
  void set_compression(compression_type comp) { _compression = comp; }

  /**
   * @brief Choose granularity of statistics collection.
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Frequency of statistics collection
   */
  void enable_statistics(statistics_freq val) { _stats_freq = val; }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param size_bytes Maximum stripe size, in bytes to be set
   */
  void set_stripe_size_bytes(size_t size_bytes)
  {
    CUDF_EXPECTS(size_bytes >= 64 << 10, "64KB is the minimum stripe size");
    _stripe_size_bytes = size_bytes;
  }

  /**
   * @brief Sets the maximum stripe size, in rows.
   *
   * If the stripe size is smaller that the row group size, row group size will be reduced to math
   * the stripe size.
   *
   * @param size_rows Maximum stripe size, in rows to be set
   */
  void set_stripe_size_rows(size_type size_rows)
  {
    CUDF_EXPECTS(size_rows >= 512, "Maximum stripe size cannot be smaller than 512");
    _stripe_size_rows = size_rows;
  }

  /**
   * @brief Sets the row index stride.
   *
   * Rounded down to a multiple of 8.
   *
   * @param stride Row index stride to be set
   */
  void set_row_index_stride(size_type stride)
  {
    CUDF_EXPECTS(stride >= 512, "Row index stride cannot be smaller than 512");
    _row_index_stride = stride;
  }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   */
  void set_table(table_view tbl) { _table = tbl; }

  /**
   * @brief Sets associated metadata
   *
   * @param meta Associated metadata
   */
  void set_metadata(table_input_metadata const* meta) { _metadata = meta; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Key-Value footer metadata
   */
  void set_key_value_metadata(std::map<std::string, std::string> metadata)
  {
    _user_data = std::move(metadata);
  }
};

/**
 * @brief Builds settings to use for `write_orc()`.
 */
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
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  orc_writer_options_builder(sink_info const& sink, table_view const& table) : options{sink, table}
  {
  }

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use
   * @return this for chaining
   */
  orc_writer_options_builder& compression(compression_type comp)
  {
    options._compression = comp;
    return *this;
  }

  /**
   * @brief Choose granularity of column statistics to be written
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Level of statistics collection
   * @return this for chaining
   */
  orc_writer_options_builder& enable_statistics(statistics_freq val)
  {
    options._stats_freq = val;
    return *this;
  }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param val maximum stripe size
   * @return this for chaining
   */
  orc_writer_options_builder& stripe_size_bytes(size_t val)
  {
    options.set_stripe_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum number of rows in output stripes.
   *
   * @param val maximum number or rows
   * @return this for chaining
   */
  orc_writer_options_builder& stripe_size_rows(size_type val)
  {
    options.set_stripe_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the row index stride.
   *
   * @param val new row index stride
   * @return this for chaining
   */
  orc_writer_options_builder& row_index_stride(size_type val)
  {
    options.set_row_index_stride(val);
    return *this;
  }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   * @return this for chaining
   */
  orc_writer_options_builder& table(table_view tbl)
  {
    options._table = tbl;
    return *this;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata
   * @return this for chaining
   */
  orc_writer_options_builder& metadata(table_input_metadata const* meta)
  {
    options._metadata = meta;
    return *this;
  }

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   * @return this for chaining
   */
  orc_writer_options_builder& key_value_metadata(std::map<std::string, std::string> metadata)
  {
    options._user_data = std::move(metadata);
    return *this;
  }

  /**
   * @brief move orc_writer_options member once it's built.
   */
  operator orc_writer_options&&() { return std::move(options); }

  /**
   * @brief move orc_writer_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `orc_writer_options` object's r-value reference
   */
  orc_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to ORC format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  auto destination = cudf::io::sink_info("dataset.orc");
 *  auto options     = cudf::io::orc_writer_options::builder(destination, table->view());
 *  cudf::io::write_orc(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param mr Device memory resource to use for device memory allocation
 */
void write_orc(orc_writer_options const& options,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Builds settings to use for `write_orc_chunked()`.
 */
class chunked_orc_writer_options_builder;

/**
 * @brief Settings to use for `write_orc_chunked()`.
 */
class chunked_orc_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  // Specify granularity of statistics collection
  statistics_freq _stats_freq = ORC_STATISTICS_ROW_GROUP;
  // Maximum size of each stripe (unless smaller than a single row group)
  size_t _stripe_size_bytes = default_stripe_size_bytes;
  // Maximum number of rows in stripe (unless smaller than a single row group)
  size_type _stripe_size_rows = default_stripe_size_rows;
  // Row index stride (maximum number of rows in each row group)
  size_type _row_index_stride = default_row_index_stride;
  // Optional associated metadata
  const table_input_metadata* _metadata = nullptr;
  // Optional footer key_value_metadata
  std::map<std::string, std::string> _user_data;

  friend chunked_orc_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
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
   * @param sink The sink used for writer output
   *
   * @return Builder to build chunked_orc_writer_options
   */
  static chunked_orc_writer_options_builder builder(sink_info const& sink);

  /**
   * @brief Returns sink info.
   *
   * @return Sink info
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression type.
   *
   * @return Compression type
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns granularity of statistics collection.
   *
   * @return Granularity of statistics collection
   */
  [[nodiscard]] statistics_freq get_statistics_freq() const { return _stats_freq; }

  /**
   * @brief Returns maximum stripe size, in bytes.
   *
   * @return Maximum stripe size, in bytes
   */
  [[nodiscard]] auto get_stripe_size_bytes() const { return _stripe_size_bytes; }

  /**
   * @brief Returns maximum stripe size, in rows.
   *
   * @return Maximum stripe size, in rows
   */
  [[nodiscard]] auto get_stripe_size_rows() const { return _stripe_size_rows; }

  /**
   * @brief Returns the row index stride.
   *
   * @return Row index stride
   */
  auto get_row_index_stride() const
  {
    auto const unaligned_stride = std::min(_row_index_stride, get_stripe_size_rows());
    return unaligned_stride - unaligned_stride % 8;
  }

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
  [[nodiscard]] std::map<std::string, std::string> const& get_key_value_metadata() const
  {
    return _user_data;
  }

  // Setters

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use
   */
  void set_compression(compression_type comp) { _compression = comp; }

  /**
   * @brief Choose granularity of statistics collection
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Frequency of statistics collection
   */
  void enable_statistics(statistics_freq val) { _stats_freq = val; }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param size_bytes Maximum stripe size, in bytes to be set
   */
  void set_stripe_size_bytes(size_t size_bytes)
  {
    CUDF_EXPECTS(size_bytes >= 64 << 10, "64KB is the minimum stripe size");
    _stripe_size_bytes = size_bytes;
  }

  /**
   * @brief Sets the maximum stripe size, in rows.
   *
   * If the stripe size is smaller that the row group size, row group size will be reduced to math
   * the stripe size.
   *
   * @param size_rows Maximum stripe size, in rows to be set
   */
  void set_stripe_size_rows(size_type size_rows)
  {
    CUDF_EXPECTS(size_rows >= 512, "maximum stripe size cannot be smaller than 512");
    _stripe_size_rows = size_rows;
  }

  /**
   * @brief Sets the row index stride.
   *
   * Rounded down to a multiple of 8.
   *
   * @param stride Row index stride to be set
   */
  void set_row_index_stride(size_type stride)
  {
    CUDF_EXPECTS(stride >= 512, "Row index stride cannot be smaller than 512");
    _row_index_stride = stride;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata
   */
  void metadata(table_input_metadata const* meta) { _metadata = meta; }

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   */
  void set_key_value_metadata(std::map<std::string, std::string> metadata)
  {
    _user_data = std::move(metadata);
  }
};

/**
 * @brief Builds settings to use for `write_orc_chunked()`.
 */
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
   * @param sink The sink used for writer output
   */
  explicit chunked_orc_writer_options_builder(sink_info const& sink) : options{sink} {}

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& compression(compression_type comp)
  {
    options._compression = comp;
    return *this;
  }

  /**
   * @brief Choose granularity of statistics collection
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Frequency of statistics collection
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& enable_statistics(statistics_freq val)
  {
    options._stats_freq = val;
    return *this;
  }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param val maximum stripe size
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& stripe_size_bytes(size_t val)
  {
    options.set_stripe_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum number of rows in output stripes.
   *
   * @param val maximum number or rows
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& stripe_size_rows(size_type val)
  {
    options.set_stripe_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the row index stride.
   *
   * @param val new row index stride
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& row_index_stride(size_type val)
  {
    options.set_row_index_stride(val);
    return *this;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& metadata(table_input_metadata const* meta)
  {
    options._metadata = meta;
    return *this;
  }

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& key_value_metadata(
    std::map<std::string, std::string> metadata)
  {
    options._user_data = std::move(metadata);
    return *this;
  }

  /**
   * @brief move chunked_orc_writer_options member once it's built.
   */
  operator chunked_orc_writer_options&&() { return std::move(options); }

  /**
   * @brief move chunked_orc_writer_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `chunked_orc_writer_options` object's r-value reference
   */
  chunked_orc_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Chunked orc writer class writes an ORC file in a chunked/stream form.
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
 *  orc_chunked_writer writer(options)
 *  writer.write(table0)
 *  writer.write(table1)
 *    ...
 *  writer.close();
 * @endcode
 */
class orc_chunked_writer {
 public:
  /**
   * @brief Default constructor, this should never be used.
   *        This is added just to satisfy cython.
   */
  orc_chunked_writer() = default;

  /**
   * @brief Constructor with chunked writer options
   *
   * @param[in] options options used to write table
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  orc_chunked_writer(chunked_orc_writer_options const& options,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Writes table to output.
   *
   * @param[in] table Table that needs to be written
   * @return returns reference of the class object
   */
  orc_chunked_writer& write(table_view const& table);

  /**
   * @brief Finishes the chunked/streamed write process.
   */
  void close();

  /// Unique pointer to impl writer class
  std::unique_ptr<cudf::io::detail::orc::writer> writer;
};

/** @} */  // end of group
}  // namespace io
}  // namespace cudf
