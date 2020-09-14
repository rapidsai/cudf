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

#include <iostream>
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
table_with_metadata read_csv(
  read_csv_args const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
               rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace io
}  // namespace cudf
