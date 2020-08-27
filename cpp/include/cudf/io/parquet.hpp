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
 * @file parquet.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <cudf/io/types.hpp>
#include <iostream>

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
 * @brief Builds parquet_reader_options to use for `read_parquet()`
 */
class parquet_reader_options_builder;

/**
 * @brief Settings to use for `read_parquet()`
 */
class parquet_reader_options {
  source_info _source;

  /// Names of column to read; empty is all
  std::vector<std::string> _columns;

  /// List of individual row groups to read (ignored if empty)
  std::vector<std::vector<size_type>> _row_groups;
  /// Number of rows to skip from the start; -1 is none
  size_type _skip_rows = -1;
  /// NUmber of rows to read; -1 is all
  size_type _num_rows = -1;

  /// Whether to store string data as categorical type
  bool _convert_strings_to_categories = false;
  /// Whether to use PANDAS metadata to load columns
  bool _use_pandas_metadata = true;
  /// Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  explicit parquet_reader_options(source_info const& src) : _source(src) {}

  friend parquet_reader_options_builder;

 public:
  explicit parquet_reader_options() = default;

  /**
   * @brief create parquet_reader_options_builder which will build parquet_reader_options
   *
   * @param src source information used to read parquet file
   * @returns parquet_reader_options_builder
   */
  static parquet_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info
   */
  source_info const& source() const { return _source; }

  /**
   * @breif Returns true/false depending on whether strings should be converted to categories or not
   */
  bool convert_strings_to_categories() const { return _convert_strings_to_categories; }

  /**
   * @breif Returns true/false depending whether to use pandas metadata or not while reading
   */
  bool use_pandas_metadata() const { return _use_pandas_metadata; }

  /**
   * @breif Returns number of rows to skip from the start
   */
  size_type skip_rows() const { return _skip_rows; }

  /**
   * @breif Returns number of rows to read
   */
  size_type num_rows() const { return _num_rows; }

  /**
   * @brief Returns names of column to be read
   */
  std::vector<std::string>& columns() { return _columns; }

  /**
   * @brief Returns names of column to be read
   */
  std::vector<std::string> const& columns() const { return _columns; }

  /**
   * @brief Returns list of individual row groups to be read
   */
  std::vector<std::vector<size_type>>& row_groups() { return _row_groups; }

  /**
   * @brief Returns list of individual row groups to be read
   */
  std::vector<std::vector<size_type>> const& row_groups() const { return _row_groups; }

  /**
   * @brief Returns timestamp_type used to cast timestamp columns
   */
  data_type timestamp_type() const { return _timestamp_type; }
};

class parquet_reader_options_builder {
  parquet_reader_options options;

 public:
  explicit parquet_reader_options_builder() = default;

  explicit parquet_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Set column names which needs to be read
   */
  parquet_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options._columns = std::move(col_names);
    return *this;
  }

  /**
   * @brief Set vector of individual row groups to read
   */
  parquet_reader_options_builder& row_groups(std::vector<std::vector<size_type>> row_grp)
  {
    options._row_groups = std::move(row_grp);
    return *this;
  }

  /*
   * @brief Set to convert strings to categories
   */
  parquet_reader_options_builder& convert_strings_to_categories(bool val)
  {
    options._convert_strings_to_categories = val;
    return *this;
  }

  /*
   * @brief Set to use pandas metadata to read
   */
  parquet_reader_options_builder& use_pandas_metadata(bool val)
  {
    options._use_pandas_metadata = val;
    return *this;
  }

  /*
   * @brief Set number of rows to skip
   */
  parquet_reader_options_builder& skip_rows(size_type val)
  {
    options._skip_rows = val;
    return *this;
  }

  /*
   * @brief Set number of rows to read
   */
  parquet_reader_options_builder& num_rows(size_type val)
  {
    options._num_rows = val;
    return *this;
  }

  /*
   * @brief timestamp_type used to cast timestamp columns
   */
  parquet_reader_options_builder& timestamp_type(data_type type)
  {
    options._timestamp_type = type;
    return *this;
  }

  /**
   * @brief move parquet_reader_options member once options is built
   */
  operator parquet_reader_options &&() { return std::move(options); }

  /**
   * @brief move parquet_reader_options member once options is built
   */
  parquet_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a Parquet dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::parquet_reader_options options{cudf::source_info(filepath)};
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
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Class to build `parquet_writer_options`
 *
 * @ingroup io_writers
 */
class parquet_writer_options_builder;

/**
 * @brief Settings to use for `write_parquet()`
 *
 * @ingroup io_writers
 */
class parquet_writer_options {
  /// Specify the sink to use for writer output
  sink_info _sink;
  /// Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  /// Specify the level of statistics in the output file
  statistics_freq _stats_level = statistics_freq::STATISTICS_ROWGROUP;
  /// Set of columns to output
  table_view _table;
  /// Optional associated metadata
  const table_metadata* _metadata = nullptr;
  /// Optionally return the raw parquet file metadata output
  bool _return_filemetadata = false;
  /// Column chunks file path to be set in the raw output metadata
  std::string _column_chunks_file_path;

  explicit parquet_writer_options(sink_info const& sink, table_view const& table)
    : _sink(sink), _table(table)
  {
  }

  friend class parquet_writer_options_builder;

 public:
  parquet_writer_options() = default;

  /**
   * @brief Build parquet_writer_options.
   *
   * @param sink sink to use for writer output
   * @param table Table to be written to output
   *
   * @return parquet_writer_options_builder parquet options builder with all arguments
   */
  static parquet_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Build parquet_writer_options.
   *
   * @return parquet_writer_options_builder parquet options builder with all arguments
   */
  static parquet_writer_options_builder builder();

  /**
   * @brief Returns sink info
   */
  sink_info const& sink() const { return _sink; }

  /**
   * @brief Returns compression format used
   */
  compression_type compression() const { return _compression; }

  /**
   * @brief Returns level of statistics requested in ouput file
   */
  statistics_freq stats_level() const { return _stats_level; }

  /**
   * @brief Returns table_view
   */
  table_view table() const { return _table; }

  /**
   * @brief Returns associated metadata
   */
  const table_metadata* metadata() const { return _metadata; }

  /**
   * @brief Returns True/False for filemetadata is requried or not
   */
  bool is_filemetadata_required() const { return _return_filemetadata; }

  /**
   * @brief Returns Column chunks file path to be set in the raw output metadata
   */
  std::string column_chunks_file_path() const { return _column_chunks_file_path; }
};

class parquet_writer_options_builder {
  parquet_writer_options options;

 public:
  explicit parquet_writer_options_builder(sink_info const& sink, table_view const& table)
    : options(sink, table)
  {
  }

  explicit parquet_writer_options_builder() {}

  /**
   * @brief Set metadata to parquet_writer_options
   */
  parquet_writer_options_builder& metadata(table_metadata const* m)
  {
    options._metadata = m;
    return *this;
  }

  /**
   * @brief Set statistics_freq to parquet_writer_options
   */
  parquet_writer_options_builder& stats_level(statistics_freq sf)
  {
    options._stats_level = sf;
    return *this;
  }

  /**
   * @brief Set compression type to parquet_writer_options
   */
  parquet_writer_options_builder& compression(compression_type comp_type)
  {
    options._compression = comp_type;
    return *this;
  }

  /**
   * @brief Set whether filemetadata is required or not to parquet_writer_options
   */
  parquet_writer_options_builder& filemetadata_required(bool req)
  {
    options._return_filemetadata = req;
    return *this;
  }

  /**
   * @brief Set column_chunks_file_path to parquet_writer_options
   */
  parquet_writer_options_builder& column_chunks_file_path(std::string file_path)
  {
    options._column_chunks_file_path.assign(file_path);
    return *this;
  }

  /**
   * @brief move parquet_writer_options member once options is built
   */
  operator parquet_writer_options &&() { return std::move(options); }

  /**
   * @brief move parquet_writer_options member once options is built
   */
  parquet_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to parquet format
 *
 * @ingroup io_writers
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::io::parquet_writer_options options{cudf::sink_info(filepath), table->view()};
 *  ...
 *  cudf::write_parquet(options);
 * @endcode
 *
 * @param options Settings for controlling writing behavior
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return A blob that contains the file metadata (parquet FileMetadata thrift message) if
 *         requested in parquet_writer_options (empty blob otherwise)
 */

std::unique_ptr<std::vector<uint8_t>> write_parquet(
  parquet_writer_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

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
 * @breif Builds options for chunked_parquet_writer_options
 */
class chunked_parquet_writer_options_builder;

/**
 * @brief Settings to use for `write_parquet_chunked()`
 *
 * @ingroup io_writers
 */

class chunked_parquet_writer_options {
  /// Specify the sink to use for writer output
  sink_info _sink;
  /// Specify the compression format to use
  compression_type _compression = compression_type::AUTO;
  /// Specify the level of statistics in the output file
  statistics_freq _stats_level = statistics_freq::STATISTICS_ROWGROUP;
  /// Optional associated metadata.
  const table_metadata_with_nullability* _nullable_metadata = nullptr;

  chunked_parquet_writer_options(sink_info const& sink) : _sink(sink) {}
  friend chunked_parquet_writer_options_builder;

 public:
  chunked_parquet_writer_options() = default;

  /**
   * @brief Returns sink info
   */
  sink_info const& sink() const { return _sink; }

  /**
   * @brief Returns compression format used
   */
  compression_type compression() const { return _compression; }

  /**
   * @brief Returns level of statistics requested in ouput file
   */
  statistics_freq stats_level() const { return _stats_level; }

  /**
   * @brief Returns nullable metadata information
   */
  const table_metadata_with_nullability* nullable_metadata() const { return _nullable_metadata; }

  static chunked_parquet_writer_options_builder builder(sink_info const& sink);
};

class chunked_parquet_writer_options_builder {
  chunked_parquet_writer_options options;

 public:
  chunked_parquet_writer_options_builder() = default;
  chunked_parquet_writer_options_builder(sink_info const& sink) : options(sink){};

  /**
   * @brief Set nullable metadata to parquet_writer_options
   */
  chunked_parquet_writer_options_builder& nullable_metadata(
    const table_metadata_with_nullability* metadata)
  {
    options._nullable_metadata = metadata;
    return *this;
  }

  /**
   * @brief Set statistics_freq to parquet_writer_options
   */
  chunked_parquet_writer_options_builder& stats_level(statistics_freq sf)
  {
    options._stats_level = sf;
    return *this;
  }

  /**
   * @brief Set compression type to parquet_writer_options
   */
  chunked_parquet_writer_options_builder& compression(compression_type comp_type)
  {
    options._compression = comp_type;
    return *this;
  }

  /**
   * @brief move chunked_parquet_writer_options member once options is built
   */
  operator chunked_parquet_writer_options &&() { return std::move(options); }

  /**
   * @brief move chunked_parquet_writer_options member once options is built
   */
  chunked_parquet_writer_options&& build() { return std::move(options); }
};

namespace detail {
namespace parquet {
/**
 * @brief Forward declaration of anonymous chunked-writer state struct.
 */
struct pq_chunked_state;
}  // namespace parquet
}  // namespace detail

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
 *  cudf::io::chunked_parquet_writer_options options{cudf::sink_info(filepath),
 *                                                          table->view()};
 *  ...
 *  auto state = cudf::write_parquet_chunked_begin(options);
 *    cudf::write_parquet_chunked(table0, state);
 *    cudf::write_parquet_chunked(table1, state);
 *    ...
 *  cudf_write_parquet_chunked_end(state);
 * @endcode
 *
 * @param[in] options Settings for controlling writing behavior
 * @param[in] mr Device memory resource to use for device memory allocation
 *
 * @returns pointer to an anonymous state structure storing information about the chunked write.
 * this pointer must be passed to all subsequent write_parquet_chunked() and
 * write_parquet_chunked_end() calls.
 */
std::shared_ptr<detail::parquet::pq_chunked_state> write_parquet_chunked_begin(
  chunked_parquet_writer_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

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
 * returned from write_parquet_chunked_begin()
 */
void write_parquet_chunked(table_view const& table,
                           std::shared_ptr<detail::parquet::pq_chunked_state> state);

/**
 * @brief Finish writing a chunked/stream parquet file.
 *
 * @ingroup io_writers
 *
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_parquet_chunked_begin()
 * @param[in] return_filemetadata If true, return the raw file metadata
 * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output metadata
 *
 * @return A blob that contains the file metadata (parquet FileMetadata thrift message) if
 *         requested in parquet_writer_options (empty blob otherwise)
 */
std::unique_ptr<std::vector<uint8_t>> write_parquet_chunked_end(
  std::shared_ptr<detail::parquet::pq_chunked_state>& state,
  bool return_filemetadata                   = false,
  const std::string& column_chunks_file_path = "");

}  // namespace io
}  // namespace cudf
