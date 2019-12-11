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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <cudf/types.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/legacy/io_types.h>

// Forward declarations
namespace arrow { namespace io {  class RandomAccessFile; } }

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Input source info for `xxx_read_arg` arguments
 *---------------------------------------------------------------------------**/
struct source_info {
  gdf_input_type type = FILE_PATH;
  std::string filepath;
  std::pair<const char*, size_t> buffer;
  std::shared_ptr<arrow::io::RandomAccessFile> file;

  explicit source_info(const std::string& file_path)
      : type(FILE_PATH), filepath(file_path) {}
  explicit source_info(const char* host_buffer, size_t size)
      : type(HOST_BUFFER), buffer(std::make_pair(host_buffer, size)) {}
  explicit source_info(
      const std::shared_ptr<arrow::io::RandomAccessFile> arrow_file)
      : type(ARROW_RANDOM_ACCESS_FILE), file(arrow_file) {}
};

/**---------------------------------------------------------------------------*
 * @brief Output sink info for `xxx_write_arg` arguments
 *---------------------------------------------------------------------------**/
struct sink_info {
  gdf_input_type type = FILE_PATH;
  std::string filepath;
  std::pair<const char*, size_t> buffer;
  std::shared_ptr<arrow::io::RandomAccessFile> file;

  explicit sink_info(const std::string& file_path)
      : type(FILE_PATH), filepath(file_path) {}
  explicit sink_info(const char* host_buffer, size_t size)
      : type(HOST_BUFFER), buffer(std::make_pair(host_buffer, size)) {}
  explicit sink_info(
      const std::shared_ptr<arrow::io::RandomAccessFile> arrow_file)
      : type(ARROW_RANDOM_ACCESS_FILE), file(arrow_file) {}
};

/**---------------------------------------------------------------------------*
 * @brief Input arguments to the `read_avro` interface
 *---------------------------------------------------------------------------**/
struct avro_read_arg {
  source_info source;                       ///< Info on source of data

  std::vector<std::string> columns;         ///< Names of column to read; empty is all

  int skip_rows = -1;                       ///< Rows to skip from the start; -1 is none
  int num_rows = -1;                        ///< Rows to read; -1 is all

  explicit avro_read_arg(const source_info& src) : source(src) {}
};

/**---------------------------------------------------------------------------*
 * @brief Input arguments to the `read_csv` interface
 *
 * Available parameters and are closely patterned after PANDAS' `read_csv` API.
 * Not all parameters are unsupported. If the matching PANDAS' parameter
 * has a default value of `None`, then a default value of `-1` or `0` may be
 * used as the equivalent.
 *
 * Parameters in PANDAS that are unavailable or in cudf:
 *  `sep`                   - use `delimiter` instead
 *  `squeeze`               - data is always returned as a cudf::table
 *  `engine`                - there is only one engine
 *  `converters`            - external lambda arguments are not supported
 *  `verbose`               - use column functions to count invalids instead
 *  `parse_dates`           - dates are detected automatically
 *  `infer_datetime_format` - date format are always inferred
 *  `keep_date_col`         - original raw data is not kept
 *  `date_parser`           - external lambda arguments are not supported
 *  `iterator`              - use `byte_range_xxx` for chunking instead
 *  `chunksize`             - use `byte_range_xxx` for chunking instead
 *  `escapechar`            - only ASCII-encoded data is supported
 *  `encoding`              - only ASCII-encoded data is supported
 *  `dialect`               - use each parameter and set that option instead
 *  `tupleize_cols`         - deprecated in PANDAS
 *  `error_bad_lines`       - exception is always raised detectable on bad data
 *  `warn_bad_lines`        - exception is always raised detectable on bad data
 *  `low_memory`            - use `byte_range_xxx` for chunking instead
 *  `memory_map`            - files are always memory-mapped
 *  `float_precision`       - there is only one converter
 *---------------------------------------------------------------------------**/
struct csv_read_arg {
  enum quote_style {
    QUOTE_MINIMAL = 0,  ///< Only quote those fields which contain special characters
    QUOTE_ALL,          ///< Quote all fields
    QUOTE_NONNUMERIC,   ///< Quote all non-numeric fields
    QUOTE_NONE          ///< Never quote fields; disable quotation when parsing
  };

  source_info source;                       ///< Info on source of data

  std::string compression = "infer";        ///< One of: `none`, `infer`, `bz2`, `gz`, `xz`, `zip`; default detects from file extension

  char lineterminator = '\n';               ///< Line terminator character
  char delimiter = ',';                     ///< Field separator; also known as `sep`
  bool windowslinetermination = false;      ///< Treat `\r\n` as line terminator
  bool delim_whitespace = false;            ///< Use white space as the delimiter; overrides the delimiter argument
  bool skipinitialspace = false;            ///< Skip white space after the delimiter
  bool skip_blank_lines = true;             ///< Ignore empty lines or parse line values as invalid

  cudf::size_type nrows = -1;                 ///< Rows to read
  cudf::size_type skiprows = -1;              ///< Rows to skip from the start
  cudf::size_type skipfooter = -1;            ///< Rows to skip from the end
  cudf::size_type header = 0;                 ///< Header row index, zero-based counting; default is no header reading

  std::vector<std::string> names;           ///< Names of the columns
  std::vector<std::string> dtype;           ///< Data types of the column; empty to infer dtypes

  std::vector<int> use_cols_indexes;        ///< Indexes of columns to read; empty is all columns
  std::vector<std::string> use_cols_names;  ///< Names of column to read; empty is all columns

  std::vector<std::string> true_values;     ///< Values to recognize as boolean True; default empty
  std::vector<std::string> false_values;    ///< Values to recognize as boolean False; default empty
  std::vector<std::string> na_values;       /**< Values to recognize as invalid; default values: 
                                            '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL',
                                            'NaN', 'n/a', 'nan', 'null'. */
  bool keep_default_na = true;              ///< Keep the default NA values
  bool na_filter = true;                    ///< Detect missing values (empty strings and the values in na_values); disabling can improve performance

  std::string prefix;                       ///< If there is no header or names, prepend this to the column ID as the name
  bool mangle_dupe_cols = true;             ///< If true, duplicate columns get a suffix; if false, data will be overwritten if there are columns with duplicate names

  bool dayfirst = false;                    ///< Is the first value in the date formatthe day?  DD/MM  versus MM/DD

  char thousands = '\0';                    ///< Numeric data thousands seperator; cannot match delimiter
  char decimal = '.';                       ///< Decimal point character; cannot match delimiter
  char comment = '\0';                      ///< Comment line start character; rest of the line will not be parsed

  char quotechar = '\"';                    ///< Character used to denote start and end of a quoted item
  quote_style quoting = QUOTE_MINIMAL;      ///< Treat string fields as quoted item and remove the first and last quotechar
  bool doublequote = true;                  ///< Whether to interpret two consecutive quotechar inside a field as a single quotechar

  size_t byte_range_offset = 0;             ///< Bytes to skip from the start
  size_t byte_range_size = 0;               ///< Bytes to read; always reads complete rows
  gdf_time_unit out_time_unit = TIME_UNIT_NONE; ///< The output resolution for date32, date64, and timestamp columns

  explicit csv_read_arg(const source_info& src) : source(src) {}
};

/**---------------------------------------------------------------------------*
 * @brief Input arguments to the `read_json` interface
 *
 * Available parameters and are closely patterned after PANDAS' `read_json` API.
 * Not all parameters are unsupported. If the matching PANDAS' parameter
 * has a default value of `None`, then a default value of `-1` or `0` may be
 * used as the equivalent.
 *
 * Parameters in PANDAS that are unavailable or in cudf:
 *  `orient`                - currently fixed-format
 *  `typ`                   - data is always returned as a cudf::table
 *  `convert_axes`          - use column functions for axes operations instead
 *  `convert_dates`         - dates are detected automatically
 *  `keep_default_dates`    - dates are detected automatically
 *  `numpy`                 - data is always returned as a cudf::table
 *  `precise_float`         - there is only one converter
 *  `date_unit`             - only millisecond units are supported
 *  `encoding`              - only ASCII-encoded data is supported
 *  `chunksize`             - use `byte_range_xxx` for chunking instead
 *---------------------------------------------------------------------------**/
struct json_read_arg {
  source_info source;                       ///< Info on source of data

  std::vector<std::string> dtype;           ///< Data types of the column; empty to infer dtypes
  std::string compression = "infer";        ///< For on-the-fly decompression, one of `none`, `infer`, `gzip`, `zip`

  bool lines = false;                       ///< Read the file as a json object per line

  size_t byte_range_offset = 0;             ///< Bytes to skip from the start
  size_t byte_range_size = 0;               ///< Bytes to read; always reads complete rows

  explicit json_read_arg(const source_info& src) : source(src) {}
};

/**---------------------------------------------------------------------------*
 * @brief Input arguments to the `read_orc` interface
 *---------------------------------------------------------------------------**/
struct orc_read_arg {
  source_info source;                       ///< Info on source of data

  std::vector<std::string> columns;         ///< Names of column to read; empty is all

  int stripe = -1;                          ///< Stripe to read; -1 is all
  int skip_rows = -1;                       ///< Rows to skip from the start; -1 is none
  int num_rows = -1;                        ///< Rows to read; -1 is all

  bool use_index = false;                   ///< Whether to use row index to speed-up reading
  bool use_np_dtypes = true;                ///< Whether to use numpy-compatible dtypes
  gdf_time_unit timestamp_unit = TIME_UNIT_NONE;  ///< Resolution of timestamps
  bool decimals_as_float = true;            ///< Whether to convert decimals to float64
  int forced_decimals_scale = -1;           /// Optional forced decimal scale; -1 is none

  explicit orc_read_arg(const source_info& src) : source(src) {}
};

/**---------------------------------------------------------------------------*
 * @brief Input arguments to the `write_orc` interface
 *---------------------------------------------------------------------------**/
struct orc_write_arg {
  sink_info sink;                           ///< Info on sink of data

  cudf::table table;                        ///< Table of columns to write

  explicit orc_write_arg(const sink_info& snk) : sink(snk) {}
};

/**---------------------------------------------------------------------------*
 * @brief Input arguments to the `read_parquet` interface
 *---------------------------------------------------------------------------**/
struct parquet_read_arg {
  source_info source;                       ///< Info on source of data

  std::vector<std::string> columns;         ///< Names of column to read; empty is all

  int row_group = -1;                       ///< Row group to read; -1 is all
  int skip_rows = -1;                       ///< Rows to skip from the start; -1 is none
  int num_rows = -1;                        ///< Rows to read; -1 is all

  bool strings_to_categorical = false;      ///< Whether to store string data as GDF_CATEGORY
  bool use_pandas_metadata = true;          ///< Whether to always load PANDAS index columns
  gdf_time_unit timestamp_unit = TIME_UNIT_NONE;  ///< Resolution of timestamps

  explicit parquet_read_arg(const source_info& src) : source(src) {}
};

}  // namespace cudf
