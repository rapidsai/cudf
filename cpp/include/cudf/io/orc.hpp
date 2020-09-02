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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace io {
/**
 * @brief Builds settings to use for `read_orc()`
 *
 * @ingroup io_readers
 */
class orc_reader_options_builder;

/**
 * @brief Settings to use for `read_orc()`
 *
 * @ingroup io_readers
 */
class orc_reader_options {
  source_info _source;

  /// Names of column to read; empty is all
  std::vector<std::string> _columns;

  /// List of individual stripes to read (ignored if empty)
  std::vector<size_type> _stripes;
  /// Rows to skip from the start; -1 is none
  size_type _skip_rows = 0;
  /// Rows to read; -1 is all
  size_type _num_rows = -1;

  /// Whether to use row index to speed-up reading
  bool _use_index = true;

  /// Whether to use numpy-compatible dtypes
  bool _use_np_dtypes = true;
  /// Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  /// Whether to convert decimals to float64
  bool _decimals_as_float = true;
  /// For decimals as int, optional forced decimal scale;
  /// -1 is auto (column scale), >=0: number of fractional digits
  size_type _forced_decimals_scale = -1;

  friend orc_reader_options_builder;

  explicit orc_reader_options(source_info const& src) : _source(src) {}
  
  public:
  explicit orc_reader_options() = default;

  /**
   * @brief Returns source_info
   */
  source_info const& source() const {return _source;}

  /**
   * @brief Returns names of the column to read
   */
  std::vector<std::string> const& columns() const {return _columns;}

  /**
   * @brief Returns list of individual stripes to read
   */
  std::vector<size_type> const& stripes() const {return _stripes;}

  /**
   * @brief Returns number of rows to skip from the start
   */
  size_type skip_rows() const {return _skip_rows;}

  /**
   * @brief Returns number of row to read
   */
  size_type num_rows() const {return _num_rows;}

  /**
   * @brief Whether to use row index to speed-up reading
   */
  bool use_index() const {return _use_index;}

  /**
   * @brief Whether to use numpy-compatible dtypes
   */
  bool use_np_dtypes() const {return _use_np_dtypes;}
  
  /**
   * @brief Returns timestamp type to which timestamp column will be cast
   */
  data_type timestamp_type() const {return _timestamp_type;}

  /**
   * @brief Whether to convert decimals to float64
   */
  bool decimals_as_float() const {return _decimals_as_float;}

  /**
   * @brief Returns whether decimal scale is inferred or forced to have limited fractional digits
   */
  size_type forced_decimals_scale() const {return _forced_decimals_scale;}

  // Setters

  /**
   * @brief Sets names of the column to read
   */
  void columns(std::vector<std::string> col_names) {_columns = std::move(col_names);}

  /**
   * @brief Sets list of individual stripes to read
   */
  void stripes(std::vector<size_type> strps) {
      CUDF_EXPECTS(strps.empty() or (_skip_rows == 0 and _num_rows == -1), "Can't set both stripes along with skip_rows/num_rows");
      _stripes = std::move(strps);
  }

  /**
   * @brief Sets number of rows to skip from the start
   */
  void skip_rows(size_type rows) {
      CUDF_EXPECTS(rows == 0 or _stripes.empty(), "Can't set both skip_rows along with stripes");
      _skip_rows = rows;
  }

  /**
   * @brief Sets number of row to read
   */
  void num_rows(size_type nrows) {
      CUDF_EXPECTS(nrows == -1 or _stripes.empty(), "Can't set both num_rows along with stripes");
      _num_rows = (nrows != 0)? nrows : -1;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading
   */
  void use_index(bool use) {_use_index = use;}

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes
   */
  void use_np_dtypes(bool use) {_use_np_dtypes = use;}
  
  /**
   * @brief Returns timestamp type to which timestamp column will be cast
   */
  void timestamp_type(data_type type) {_timestamp_type = type;}

  /**
   * @brief Enable/Disable convertion of decimals to float64
   */
  void decimals_as_float(bool val) {_decimals_as_float = val;}

  /**
   * @brief Sets whether decimal scale is inferred or forced to have limited fractional digits
   */
  void forced_decimals_scale(size_type val) {_forced_decimals_scale = val;}

  /**
   * @brief Creates `orc_reader_options_builder` which is used to update options
   */
  static orc_reader_options_builder builder(source_info const& src);
};

class orc_reader_options_builder {
    orc_reader_options options;

    public:
    explicit orc_reader_options_builder() = default;
    
    explicit orc_reader_options_builder(source_info const& src):options{src}{};

  /**
   * @brief Sets names of the column to read
   */
  orc_reader_options_builder& columns(std::vector<std::string> col_names) {
     options._columns = std::move(col_names);
     return *this;
  }

  /**
   * @brief Sets list of individual stripes to read
   */
  orc_reader_options_builder& stripes(std::vector<size_type> strps) {
      options.stripes(std::move(strps));
     return *this;
  }

  /**
   * @brief Sets number of rows to skip from the start
   */
  orc_reader_options_builder& skip_rows(size_type rows) {
      options.skip_rows(rows);
     return *this;
  }

  /**
   * @brief Sets number of row to read
   */
  orc_reader_options_builder& num_rows(size_type nrows) {
      options.num_rows(nrows);
     return *this;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading
   */
  orc_reader_options_builder& use_index(bool use) {
      options._use_index = use;
     return *this;
  }

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes
   */
  orc_reader_options_builder& use_np_dtypes(bool use) {
      options._use_np_dtypes = use;
     return *this;
  }
  
  /**
   * @brief Returns timestamp type to which timestamp column will be cast
   */
  orc_reader_options_builder& timestamp_type(data_type type) {
      options._timestamp_type = type;
     return *this;
  }

  /**
   * @brief Enable/Disable convertion of decimals to float64
   */
  orc_reader_options_builder& decimals_as_float(bool val) {
      options._decimals_as_float = val;
     return *this;
  }

  /**
   * @brief Sets whether decimal scale is inferred or forced to have limited fractional digits
   */
  orc_reader_options_builder& forced_decimals_scale(size_type val) {
      options._forced_decimals_scale = val;
     return *this;
  }

  /**
   * @brief move orc_reader_options member once options is built
   */
  operator orc_reader_options &&() {return std::move(options);}
  
  /**
   * @brief move orc_reader_options member once options is built
   */
  orc_reader_options&& build() {return std::move(options);}
};

/**
 * @brief Reads an ORC dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::orc_reader_options options = cudf::orc_reader_options::builder(cudf::source_info(filepath));
 *  ...
 *  auto result = cudf::read_orc(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns
 */
table_with_metadata read_orc(orc_reader_options const& options,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());



/**
 * @brief Builds settings to use for `write_orc()`
 *
 * @ingroup io_writers
 */
class orc_writer_options_builder;

/**
 * @brief Settings to use for `write_orc()`
 *
 * @ingroup io_writers
 */
class orc_writer_options {
  /// Specify the sink to use for writer output
  sink_info _sink;
  /// Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  /// Enable writing column statistics
  bool _enable_statistics = true;
  /// Set of columns to output
  table_view _table;
  /// Optional associated metadata
  const table_metadata* _metadata = nullptr;

  friend orc_writer_options_builder;

  explicit orc_writer_options(sink_info const& sink, table_view const& table): _sink(sink), _table(table){}
  public:
  explicit orc_writer_options() = default;

  /**
   * @brief Returns sink info
   */
  sink_info const& sink() const {return _sink;}

  /**
   * @brief Returns compression type
   */
  compression_type compression() const {return _compression;}

  /**
   * @brief Whether writing column statistics is enabled/disabled
   */
  bool enable_statistics() const {return _enable_statistics;}

  /**
   * @brief Returns table to be written to output
   */
  table_view table() const {return _table;}

  /**
   * @brief Returns associated metadata
   */
  table_metadata const* metadata() const {return _metadata;}


  // Setters
  
  /**
   * @brief Sets compression type
   */
  void compression(compression_type comp) {_compression = comp;}

  /**
   * @brief Enable/Disable writing column statistics
   */
  void enable_statistics(bool val) {_enable_statistics = val;}

  /**
   * @brief Returns table to be written to output
   */
  void table(table_view tbl) {_table = tbl;}

  /**
   * @brief Returns associated metadata
   */
  void metadata(table_metadata* meta) {_metadata = meta;}

  /**
   * @brief Returns builder to update options
   */
  static orc_writer_options_builder builder(sink_info const& sink, table_view const& table);
};

class orc_writer_options_builder {
    orc_writer_options options;

    public:
    orc_writer_options_builder() = default;

    orc_writer_options_builder(sink_info const& sink, table_view const& table):options{sink, table}{}

    /**
   * @brief Sets compression type
   */
  orc_writer_options_builder& compression(compression_type comp) {
      options._compression = comp;
      return *this;
  }

  /**
   * @brief Enable/Disable writing column statistics
   */
  orc_writer_options_builder& enable_statistics(bool val) {
      options._enable_statistics = val;
      return *this;
  }

  /**
   * @brief Returns table to be written to output
   */
  orc_writer_options_builder& table(table_view tbl) {
      options._table = tbl;
      return *this;
  }

  /**
   * @brief Returns associated metadata
   */
  orc_writer_options_builder& metadata(table_metadata* meta) {
      options._metadata = meta;
      return *this;
  }

  /**
   * @brief move orc_writer_options member once options is built
   */
  operator orc_writer_options &&() {return std::move(options);}

  /**
   * @brief move orc_writer_options member once options is built
   */
  orc_writer_options&& build() {return std::move(options);}
};

/**
 * @brief Writes a set of columns to ORC format
 *
 * @ingroup io_writers
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::orc_writer_options options = cudf::orc_writer_options::builder(cudf::sink_info(filepath), table->view());
 *  ...
 *  cudf::write_orc(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param mr Device memory resource to use for device memory allocation
 */
void write_orc(orc_writer_options const& options,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Builds settings to use for `write_orc_chunked()`
 *
 * @ingroup io_writers
 */
class chunked_orc_writer_options_builder;

/**
 * @brief Settings to use for `write_orc_chunked()`
 *
 * @ingroup io_writers
 */
class chunked_orc_writer_options {
  /// Specify the sink to use for writer output
  sink_info _sink;
  /// Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  /// Enable writing column statistics
  bool _enable_statistics = true;
  /// Optional associated metadata
  const table_metadata_with_nullability* _metadata = nullptr;

  friend chunked_orc_writer_options_builder;

  chunked_orc_writer_options(sink_info const& sink): _sink(sink){}

  public:
  explicit chunked_orc_writer_options() = default;

  /**
   * @brief Returns sink info
   */
  sink_info const& sink() const {return _sink;}

  /**
   * @brief Returns compression type
   */
  compression_type compression() const {return _compression;}

  /**
   * @brief Whether writing column statistics is enabled/disabled
   */
  bool enable_statistics() const {return _enable_statistics;}

  /**
   * @brief Returns associated metadata
   */
  table_metadata_with_nullability const* metadata() const {return _metadata;}

  // Setters

  /**
   * @brief Sets compression type
   */
  void compression(compression_type comp) {_compression = comp;}

  /**
   * @brief Enable/Disable writing column statistics
   */
  void enable_statistics(bool val) {_enable_statistics = val;}

  /**
   * @brief Returns associated metadata
   */
  void metadata(table_metadata_with_nullability* meta) {_metadata = meta;}

  /**
   * @brief Returns builder to update options
   */
  static chunked_orc_writer_options_builder builder(sink_info const& sink);
};

class chunked_orc_writer_options_builder {
    chunked_orc_writer_options options;

    public:
    chunked_orc_writer_options_builder() = default;

    chunked_orc_writer_options_builder(sink_info const& sink):options{sink}{}

    /**
   * @brief Sets compression type
   */
  chunked_orc_writer_options_builder& compression(compression_type comp) {
      options._compression = comp;
      return *this;
  }

  /**
   * @brief Enable/Disable writing column statistics
   */
  chunked_orc_writer_options_builder& enable_statistics(bool val) {
      options._enable_statistics = val;
      return *this;
  }

  /**
   * @brief Returns associated metadata
   */
  chunked_orc_writer_options_builder& metadata(table_metadata_with_nullability* meta) {
      options._metadata = meta;
      return *this;
  }

  /**
   * @brief move chunked_rc_writer_options member once options is built
   */
  operator chunked_orc_writer_options &&() {return std::move(options);}

  /**
   * @brief move chunked_rc_writer_options member once options is built
   */
  chunked_orc_writer_options&& build() {return std::move(options);}
};

namespace detail {
namespace orc {
/**
 * @brief Forward declaration of anonymous chunked-writer state struct.
 */
struct orc_chunked_state;
}  // namespace orc
}  // namespace detail

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
 *  cudf::io::chunked_orc_writer_options options{cudf::sink_info(filepath), table->view()};
 *  ...
 *  auto state = cudf::write_orc_chunked_begin(options);
 *    cudf::write_orc_chunked(table0, state);
 *    cudf::write_orc_chunked(table1, state);
 *    ...
 *  cudf_write_orc_chunked_end(state);
 * @endcode
 *
 * @param[in] options Settings for controlling writing behavior
 * @param[in] mr Device memory resource to use for device memory allocation
 *
 * @returns pointer to an anonymous state structure storing information about the chunked write.
 * this pointer must be passed to all subsequent write_orc_chunked() and write_orc_chunked_end()
 *          calls.
 */
std::shared_ptr<detail::orc::orc_chunked_state> write_orc_chunked_begin(
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
 * returned from write_orc_chunked_begin()
 */
void write_orc_chunked(table_view const& table,
                       std::shared_ptr<detail::orc::orc_chunked_state> state);

/**
 * @brief Finish writing a chunked/stream orc file.
 *
 * @ingroup io_writers
 *
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_orc_chunked_begin()
 */
void write_orc_chunked_end(std::shared_ptr<detail::orc::orc_chunked_state>& state);

} // namespace io
} // namespace cudf
