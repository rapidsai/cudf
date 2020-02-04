/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace experimental {
//! IO interfaces
namespace io {

/**
 * @brief Settings to use for `read_avro()`
 */
struct read_avro_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  explicit read_avro_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads an Avro dataset into a set of columns
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.avro";
 *  cudf::read_avro_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_avro(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_avro(
    read_avro_args const& args,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_csv()`
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
  /// Rows to skip from the end
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
  data_type timestamp_type{EMPTY};

  explicit read_csv_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads a CSV dataset into a set of columns
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.csv";
 *  cudf::read_csv_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_csv(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_csv(
    read_csv_args const& args,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_orc()`
 */
struct read_orc_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Stripe to read; -1 is all
  size_type stripe = -1;
  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  /// Whether to use row index to speed-up reading
  bool use_index = true;

  /// Whether to use numpy-compatible dtypes
  bool use_np_dtypes = true;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{EMPTY};

  /// Whether to convert decimals to float64
  bool decimals_as_float = true;
  /// For decimals as int, optional forced decimal scale;
  /// -1 is auto (column scale), >=0: number of fractional digits
  int forced_decimals_scale = -1;

  explicit read_orc_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads an ORC dataset into a set of columns
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::read_orc_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 *
 * @return The set of columns
 */
table_with_metadata read_orc(
    read_orc_args const& args,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `write_orc()`
 */
struct write_orc_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression;
  /// Set of columns to output
  table_view table;
  /// Optional associated metadata
  const table_metadata *metadata;

  explicit write_orc_args(sink_info const& snk, table_view const& table_,
                          const table_metadata *metadata_ = nullptr,
                          compression_type compression_ = compression_type::AUTO)
      : sink(snk), table(table_), metadata(metadata_), compression(compression_) {}
};

/**
 * @brief Writes a set of columns to ORC format
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::write_orc_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  cudf::write_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 */
void write_orc(write_orc_args const& args, rmm::mr::device_memory_resource* mr =
                                               rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_parquet()`
 */
struct read_parquet_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Row group to read; -1 is all
  size_type row_group = -1;
  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  /// Whether to store string data as categorical type
  bool strings_to_categorical = false;
  /// Whether to use PANDAS metadata to load columns
  bool use_pandas_metadata = true;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{EMPTY};

  explicit read_parquet_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads a Parquet dataset into a set of columns
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::read_parquet_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_parquet(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_parquet(
    read_parquet_args const& args,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `write_parquet()`
 */
struct write_parquet_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression;
  /// Specify the level of statistics in the output file
  statistics_freq stats_level;
  /// Set of columns to output
  table_view table;
  /// Optional associated metadata
  const table_metadata *metadata;

  explicit write_parquet_args(sink_info const& sink_, table_view const& table_,
                              const table_metadata *metadata_ = nullptr,
                              compression_type compression_ = compression_type::AUTO,
                              statistics_freq stats_lvl_ = statistics_freq::STATISTICS_ROWGROUP)
      : sink(sink_), table(table_), metadata(metadata_), compression(compression_), stats_level(stats_lvl_) {}
};

/**
 * @brief Writes a set of columns to parquet format
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::experimental::io::write_parquet_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  cudf::write_parquet(args);
 * @endcode
 *
 * @param args Settings for controlling writing behavior
 * @param mr Optional resource to use for device memory allocation
 */
void write_parquet(write_parquet_args const& args, rmm::mr::device_memory_resource* mr =
                                               rmm::mr::get_default_resource());


}  // namespace io
}  // namespace experimental
}  // namespace cudf
