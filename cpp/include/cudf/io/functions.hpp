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
 * @brief Settings to use for `read_avro()`
 *
 * @ingroup io_readers
 */
struct read_avro_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  read_avro_args() = default;

  explicit read_avro_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads an Avro dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.avro";
 *  cudf::read_avro_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_avro(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_avro(
  read_avro_args const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Input arguments to the `read_json` interface
 *
 * @ingroup io_readers
 *
 * Available parameters and are closely patterned after PANDAS' `read_json` API.
 * Not all parameters are unsupported. If the matching PANDAS' parameter
 * has a default value of `None`, then a default value of `-1` or `0` may be
 * used as the equivalent.
 *
 * Parameters in PANDAS that are unavailable or in cudf:
 *
 * | Name | Description |
 * | ---- | ----------- |
 * | `orient`             | currently fixed-format |
 * | `typ`                | data is always returned as a cudf::table |
 * | `convert_axes`       | use column functions for axes operations instead |
 * | `convert_dates`      | dates are detected automatically |
 * | `keep_default_dates` | dates are detected automatically |
 * | `numpy`              | data is always returned as a cudf::table |
 * | `precise_float`      | there is only one converter |
 * | `date_unit`          | only millisecond units are supported |
 * | `encoding`           | only ASCII-encoded data is supported |
 * | `chunksize`          | use `byte_range_xxx` for chunking instead |
 *
 */
struct read_json_args {
  source_info source;

  ///< Data types of the column; empty to infer dtypes
  std::vector<std::string> dtype;
  /// Specify the compression format of the source or infer from file extension
  compression_type compression = compression_type::AUTO;

  ///< Read the file as a json object per line
  bool lines = false;

  ///< Bytes to skip from the start
  size_t byte_range_offset = 0;
  ///< Bytes to read; always reads complete rows
  size_t byte_range_size = 0;

  /// Whether to parse dates as DD/MM versus MM/DD
  bool dayfirst = false;

  read_json_args() = default;

  explicit read_json_args(const source_info& src) : source(src) {}
};

/**
 * @brief Reads a JSON dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.json";
 *  cudf::read_json_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_json(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_json(
  read_json_args const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_csv()`
 *
 * @ingroup io_readers
 */
struct read_csv_args {
  source_info source;

  // Read settings

  /// Specify the compression format of the source or infer from file extension
  compression_type compression = compression_type::AUTO;
  /// Bytes to skip from the source start
  size_t byte_range_offset = 0;
  /// Bytes to read; always reads complete rows
  size_t byte_range_size = 0;
  /// Names of all the columns; if empty then names are auto-generated
  std::vector<std::string> names;
  /// If there is no header or names, prepend this to the column ID as the name
  std::string prefix;
  /// Whether to rename duplicate column names
  bool mangle_dupe_cols = true;

  // Filter settings

  /// Names of columns to read; empty is all columns
  std::vector<std::string> use_cols_names;
  /// Indexes of columns to read; empty is all columns
  std::vector<int> use_cols_indexes;
  /// Rows to read; -1 is all
  size_type nrows = -1;
  /// Rows to skip from the start; -1 is none
  size_type skiprows = -1;
  /// Rows to skip from the end; -1 is none
  size_type skipfooter = -1;
  /// Header row index
  size_type header = 0;

  // Parsing settings

  /// Line terminator
  char lineterminator = '\n';
  /// Field delimiter
  char delimiter = ',';
  /// Numeric data thousands seperator; cannot match delimiter
  char thousands = '\0';
  /// Decimal point character; cannot match delimiter
  char decimal = '.';
  /// Comment line start character
  char comment = '\0';
  /// Treat `\r\n` as line terminator
  bool windowslinetermination = false;
  /// Treat whitespace as field delimiter; overrides character delimiter
  bool delim_whitespace = false;
  /// Skip whitespace after the delimiter
  bool skipinitialspace = false;
  /// Ignore empty lines or parse line values as invalid
  bool skip_blank_lines = true;
  /// Treatment of quoting behavior
  quote_style quoting = quote_style::MINIMAL;
  /// Quoting character (if `quoting` is true)
  char quotechar = '"';
  /// Whether a quote inside a value is double-quoted
  bool doublequote = true;
  /// Names of columns to read as datetime
  std::vector<std::string> infer_date_names;
  /// Indexes of columns to read as datetime
  std::vector<int> infer_date_indexes;

  // Conversion settings

  /// Per-column types; disables type inference on those columns
  std::vector<std::string> dtype;
  /// Additional values to recognize as boolean true values
  std::vector<std::string> true_values;
  /// Additional values to recognize as boolean false values
  std::vector<std::string> false_values;
  /// Additional values to recognize as null values
  std::vector<std::string> na_values;
  /// Whether to keep the built-in default NA values
  bool keep_default_na = true;
  /// Whether to disable null filter; disabling can improve performance
  bool na_filter = true;
  /// Whether to parse dates as DD/MM versus MM/DD
  bool dayfirst = false;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{type_id::EMPTY};

  read_csv_args() = default;
  explicit read_csv_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads a CSV dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf/io/functions.hpp>
 *  ...
 *  std::string filepath = "dataset.csv";
 *  cudf::io::read_csv_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_csv(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_csv(read_csv_args const& args,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `write_csv()`
 *
 * @ingroup io_writers
 */
struct write_csv_args : detail::csv::writer_options {
  write_csv_args(sink_info const& snk,
                 table_view const& table,
                 std::string const& na,
                 bool include_header,
                 int rows_per_chunk,
                 std::string line_term          = std::string{"\n"},
                 char delim                     = ',',
                 std::string true_v             = std::string{"true"},
                 std::string false_v            = std::string{"false"},
                 table_metadata const* metadata = nullptr)
    : writer_options(na, include_header, rows_per_chunk, line_term, delim, true_v, false_v),
      sink_(snk),
      table_(table),
      metadata_(metadata)
  {
  }

  detail::csv::writer_options const& get_options(void) const
  {
    return *this;  // sliced to base
  }

  sink_info const& sink(void) const { return sink_; }

  table_view const& table(void) const { return table_; }

  table_metadata const* metadata(void) const { return metadata_; }

  // Specify the sink to use for writer output:
  //
  sink_info const sink_;

  // Set of columns to output:
  //
  table_view const table_;

  // Optional associated metadata
  //
  table_metadata const* metadata_;
};

/**
 * @brief Writes a set of columns to CSV format
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  #include <cudf/io/functions.hpp>
 *  ...
 *  std::string filepath = "dataset.csv";
 *  cudf::io::sink_info sink_info(filepath);
 *
 *  cudf::io::write_csv_args args{sink_info, table->view(), na, include_header,
 * rows_per_chunk};
 *  ...
 *  cudf::io::write_csv(args);
 * @endcode
 *
 * @param args Settings for controlling writing behavior
 * @param mr Device memory resource to use for device memory allocation
 */
void write_csv(write_csv_args const& args,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_orc()`
 *
 * @ingroup io_readers
 */
struct read_orc_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Stripe to read; -1 is all
  size_type stripe = -1;
  /// Number of stripes to read starting from `stripe`; default is one if stripe >= 0
  size_type stripe_count = -1;
  /// List of individual stripes to read (ignored if empty)
  std::vector<size_type> stripe_list;
  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  /// Whether to use row index to speed-up reading
  bool use_index = true;

  /// Whether to use numpy-compatible dtypes
  bool use_np_dtypes = true;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{type_id::EMPTY};

  /// Whether to convert decimals to float64
  bool decimals_as_float = true;
  /// For decimals as int, optional forced decimal scale;
  /// -1 is auto (column scale), >=0: number of fractional digits
  int forced_decimals_scale = -1;

  read_orc_args() = default;

  explicit read_orc_args(source_info const& src) : source(src) {}
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
 *  cudf::read_orc_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns
 */
table_with_metadata read_orc(read_orc_args const& args,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_parquet()`
 */
struct read_parquet_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Row group to read; -1 is all
  size_type row_group = -1;
  /// Number of row groups to read starting from row_group; default is one if row_group >= 0
  size_type row_group_count = -1;
  /// List of individual row groups to read (ignored if empty)
  std::vector<size_type> row_group_list;
  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  /// Whether to store string data as categorical type
  bool strings_to_categorical = false;
  /// Whether to use PANDAS metadata to load columns
  bool use_pandas_metadata = true;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{type_id::EMPTY};

  explicit read_parquet_args() = default;

  explicit read_parquet_args(source_info const& src) : source(src) {}
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
 *  cudf::read_parquet_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_parquet(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_parquet(
  read_parquet_args const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `write_orc()`
 *
 * @ingroup io_writers
 */
struct write_orc_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression;
  /// Enable writing column statistics
  bool enable_statistics;
  /// Set of columns to output
  table_view table;
  /// Optional associated metadata
  const table_metadata* metadata;

  write_orc_args() = default;

  explicit write_orc_args(sink_info const& snk,
                          table_view const& table_,
                          const table_metadata* metadata_ = nullptr,
                          compression_type compression_   = compression_type::AUTO,
                          bool stats_en                   = true)
    : sink(snk),
      table(table_),
      metadata(metadata_),
      compression(compression_),
      enable_statistics(stats_en)
  {
  }
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
 *  cudf::write_orc_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  cudf::write_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource to use for device memory allocation
 */
void write_orc(write_orc_args const& args,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `write_orc_chunked()`
 *
 * @ingroup io_writers
 */
struct write_orc_chunked_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression;
  /// Enable writing column statistics
  bool enable_statistics;
  /// Optional associated metadata
  const table_metadata_with_nullability* metadata;

  explicit write_orc_chunked_args(sink_info const& sink_,
                                  const table_metadata_with_nullability* metadata_ = nullptr,
                                  compression_type compression_ = compression_type::AUTO,
                                  bool stats_en                 = true)
    : sink(sink_), metadata(metadata_), compression(compression_), enable_statistics(stats_en)
  {
  }
};

namespace detail {
namespace orc {
/**
 * @brief Forward declaration of anonymous chunked-writer state struct.
 */
struct orc_chunked_state;
};  // namespace orc
};  // namespace detail

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
 *  cudf::io::write_orc_chunked_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  auto state = cudf::write_orc_chunked_begin(args);
 *    cudf::write_orc_chunked(table0, state);
 *    cudf::write_orc_chunked(table1, state);
 *    ...
 *  cudf_write_orc_chunked_end(state);
 * @endcode
 *
 * @param[in] args Settings for controlling writing behavior
 * @param[in] mr Device memory resource to use for device memory allocation
 *
 * @returns pointer to an anonymous state structure storing information about the chunked write.
 * this pointer must be passed to all subsequent write_orc_chunked() and write_orc_chunked_end()
 *          calls.
 */
std::shared_ptr<detail::orc::orc_chunked_state> write_orc_chunked_begin(
  write_orc_chunked_args const& args,
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

/**
 * @brief Settings to use for `write_parquet()`
 *
 * @ingroup io_writers
 */
struct write_parquet_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression = compression_type::AUTO;
  /// Specify the level of statistics in the output file
  statistics_freq stats_level = statistics_freq::STATISTICS_ROWGROUP;
  /// Set of columns to output
  table_view table;
  /// Optional associated metadata
  const table_metadata* metadata;
  /// Optionally return the raw parquet file metadata output
  bool return_filemetadata = false;
  /// Column chunks file path to be set in the raw output metadata
  std::string metadata_out_file_path;

  write_parquet_args() = default;

  explicit write_parquet_args(sink_info const& sink_,
                              table_view const& table_,
                              const table_metadata* metadata_ = nullptr,
                              compression_type compression_   = compression_type::AUTO,
                              statistics_freq stats_lvl_ = statistics_freq::STATISTICS_ROWGROUP)
    : sink(sink_),
      compression(compression_),
      stats_level(stats_lvl_),
      table(table_),
      metadata(metadata_)
  {
  }
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
 *  cudf::io::write_parquet_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  cudf::write_parquet(args);
 * @endcode
 *
 * @param args Settings for controlling writing behavior
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return A blob that contains the file metadata (parquet FileMetadata thrift message) if
 *         requested in write_parquet_args (empty blob otherwise)
 */
std::unique_ptr<std::vector<uint8_t>> write_parquet(
  write_parquet_args const& args,
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
 * @brief Settings to use for `write_parquet_chunked()`
 *
 * @ingroup io_writers
 */
struct write_parquet_chunked_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression = compression_type::AUTO;
  /// Specify the level of statistics in the output file
  statistics_freq stats_level = statistics_freq::STATISTICS_ROWGROUP;
  /// Optional associated metadata.
  const table_metadata_with_nullability* metadata;

  write_parquet_chunked_args() = default;

  explicit write_parquet_chunked_args(
    sink_info const& sink_,
    const table_metadata_with_nullability* metadata_ = nullptr,
    compression_type compression_                    = compression_type::AUTO,
    statistics_freq stats_lvl_                       = statistics_freq::STATISTICS_ROWGROUP)
    : sink(sink_), compression(compression_), stats_level(stats_lvl_), metadata(metadata_)
  {
  }
};

namespace detail {
namespace parquet {
/**
 * @brief Forward declaration of anonymous chunked-writer state struct.
 */
struct pq_chunked_state;
};  // namespace parquet
};  // namespace detail

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
 *  cudf::io::write_parquet_chunked_args args{cudf::sink_info(filepath),
 *                                                          table->view()};
 *  ...
 *  auto state = cudf::write_parquet_chunked_begin(args);
 *    cudf::write_parquet_chunked(table0, state);
 *    cudf::write_parquet_chunked(table1, state);
 *    ...
 *  cudf_write_parquet_chunked_end(state);
 * @endcode
 *
 * @param[in] args Settings for controlling writing behavior
 * @param[in] mr Device memory resource to use for device memory allocation
 *
 * @returns pointer to an anonymous state structure storing information about the chunked write.
 * this pointer must be passed to all subsequent write_parquet_chunked() and
 * write_parquet_chunked_end() calls.
 */
std::shared_ptr<detail::parquet::pq_chunked_state> write_parquet_chunked_begin(
  write_parquet_chunked_args const& args,
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
 */
void write_parquet_chunked_end(std::shared_ptr<detail::parquet::pq_chunked_state>& state);

}  // namespace io
}  // namespace cudf
