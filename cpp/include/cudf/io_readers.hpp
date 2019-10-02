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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cudf.h"
#include <cudf/legacy/table.hpp>

// Forward declarations
namespace arrow { namespace io {  class RandomAccessFile; } }

namespace cudf {
namespace io {
namespace avro {
/**---------------------------------------------------------------------------*
 * @brief Options for the Avro reader
 *---------------------------------------------------------------------------**/
struct reader_options {
  std::vector<std::string> columns;

  reader_options() = default;
  reader_options(reader_options const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief Constructor to populate reader options.
   *
   * @param[in] columns List of columns to read. If empty, all columns are read
   *---------------------------------------------------------------------------**/
  reader_options(std::vector<std::string> cols) : columns(std::move(cols)) {}
};

/**---------------------------------------------------------------------------*
 * @brief Class to read Apache Avro data into cuDF columns
 *---------------------------------------------------------------------------**/
class reader {
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructor for a file path source.
   *---------------------------------------------------------------------------**/
  explicit reader(std::string filepath, reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an existing memory buffer source.
   *---------------------------------------------------------------------------**/
  explicit reader(const char *buffer, size_t length,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an Arrow file source
   *---------------------------------------------------------------------------**/
  explicit reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns the entire data set.
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_all();

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns a range of rows.
   *
   * @param[in] skip_rows Number of rows to skip from the start
   * @param[in] num_rows Number of rows to read; use `0` for all remaining data
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_rows(size_t skip_rows, size_t num_rows);

  ~reader();
};

}  // namespace avro

namespace json {
/**---------------------------------------------------------------------------*
 * @brief Arguments to the read_json interface.
 *---------------------------------------------------------------------------**/
struct reader_options {
  gdf_input_type  source_type = HOST_BUFFER;      ///< Type of the data source.
  std::string     source;                         ///< If source_type is FILE_PATH, contains the filepath. If source_type is HOST_BUFFER, contains the input JSON data.

  std::vector<std::string>  dtype;                ///< Ordered list of data types; pass an empty vector to use data type deduction.
  std::string               compression = "infer";///< Compression type ("none", "infer", "gzip", "zip"); default is "infer".
  bool                      lines = false;        ///< Read the file as a json object per line; default is false.

  reader_options() = default;

  reader_options(reader_options const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief Constructor that sets the source data members.
   *
   * @param[in] src_type Enum describing the type of the data source.
   * @param[in] src If src_type is FILE_PATH, contains the filepath.
   * If source_type is HOST_BUFFER, contains the input JSON data.
   *---------------------------------------------------------------------------**/
  reader_options(gdf_input_type src_type, std::string const &src)
      : source_type(src_type), source(src) {}
};

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns.
 *
 *---------------------------------------------------------------------------**/
class reader {
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructor; throws if the arguments are not supported.
   *---------------------------------------------------------------------------**/
  explicit reader(reader_options const &args);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the reader_options
   * constuctor parameter.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read();

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the args_ data member.
   *
   * Stores the parsed gdf columns in an internal data member.
   * @param[in] offset ///< Offset of the byte range to read.
   * @param[in] size   ///< Size of the byte range to read. If set to zero,
   * all data after byte_range_offset is read.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_byte_range(size_t offset, size_t size);

  ~reader();
};

}  // namespace json

namespace csv {
/**---------------------------------------------------------------------------*
 * @brief Quoting behavior for CSV readers/writers
 *---------------------------------------------------------------------------**/
enum quote_style {
  QUOTE_MINIMAL = 0,                        ///< Only quote those fields which contain special characters; enable quotation when parsing.
  QUOTE_ALL,                                ///< Quote all fields; enable quotation when parsing.
  QUOTE_NONNUMERIC,                         ///< Quote all non-numeric fields; enable quotation when parsing.
  QUOTE_NONE                                ///< Never quote fields; disable quotation when parsing.
};

/**---------------------------------------------------------------------------*
 * @brief Options for the CSV reader
 *
 * TODO: Clean-up the parameters, as it is decoupled from the `read_csv`
 * interface. That interface allows it to be more closely aligned with PANDAS'
 * for user-friendliness.
 *---------------------------------------------------------------------------**/
struct reader_options {
  std::string compression = "none";         ///< Compression type ("none", "infer", "bz2", "gz", "xz", "zip"); with the default value, "infer", infers the compression from the file extension.

  char          lineterminator = '\n';      ///< Define the line terminator character; Default is '\n'.
  char          delimiter = ',';            ///< Define the field separator; Default is ','.
  char          decimal = '.';              ///< The decimal point character; default is '.'. Should not match the delimiter.
  char          thousands = '\0';           ///< Single character that separates thousands in numeric data; default is '\0'. Should not match the delimiter.
  char          comment = '\0';             ///< The character used to denote start of a comment line. The rest of the line will not be parsed. The default is '\0'.
  bool          dayfirst = false;           ///< Is day the first value in the date format (DD/MM versus MM/DD)? false by default.
  bool          delim_whitespace = false;   ///< Use white space as the delimiter; default is false. This overrides the delimiter argument.
  bool          skipinitialspace = false;   ///< Skip white spaces after the delimiter; default is false.
  bool          skip_blank_lines = true;    ///< Indicates whether to ignore empty lines, or parse and interpret values as NA. Default value is true.
  gdf_size_type header = 0;                 ///< Row of the header data, zero based counting; Default is zero.

  std::vector<std::string> names;           ///< Ordered List of column names; Empty by default.
  std::vector<std::string> dtype;           ///< Ordered List of data types; Empty by default.

  std::vector<int> use_cols_indexes;        ///< Indexes of columns to be processed and returned; Empty by default - process all columns.
  std::vector<std::string> use_cols_names;  ///< Names of columns to be processed and returned; Empty by default - process all columns.

  std::vector<int> infer_date_indexes;      ///< Column indexes to attempt to infer as date
  std::vector<std::string> infer_date_names;///< Column names to attempt to infer as date

  std::vector<std::string> true_values;     ///< List of values to recognize as boolean True; Empty by default.
  std::vector<std::string> false_values;    ///< List of values to recognize as boolean False; Empty by default.
  std::vector<std::string> na_values;       /**< Array of strings that should be considered as NA. By default the following values are interpreted as NA: 
                                            '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL',
                                            'NaN', 'n/a', 'nan', 'null'. */

  bool          keep_default_na = true;     ///< Keep the default NA values; true by default.
  bool          na_filter = true;           ///< Detect missing values (empty strings and the values in na_values); true by default. Passing false can improve performance.

  std::string   prefix;                     ///< If there is no header or names, prepend this to the column ID as the name; Default value is an empty string.
  bool          mangle_dupe_cols = true;    ///< If true, duplicate columns get a suffix. If false, data will be overwritten if there are columns with duplicate names; true by default.

  char          quotechar = '\"';           ///< Define the character used to denote start and end of a quoted item; default is '\"'.
  quote_style   quoting = QUOTE_MINIMAL;    ///< Defines reader's quoting behavior; default is QUOTE_MINIMAL.
  bool          doublequote = true;         ///< Indicates whether to interpret two consecutive quotechar inside a field as a single quotechar; true by default.

  gdf_time_unit out_time_unit = TIME_UNIT_NONE; ///< Defines the output resolution for date32, date64, and timestamp columns

  reader_options() = default;
};

/**---------------------------------------------------------------------------*
 * @brief Class used to parse CSV input and convert it into gdf columns.
 *---------------------------------------------------------------------------**/
class reader {
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructor for a file path source.
   *---------------------------------------------------------------------------**/
  explicit reader(std::string filepath, reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an existing memory buffer source.
   *---------------------------------------------------------------------------**/
  explicit reader(const char *buffer, size_t length,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an Arrow file source.
   *---------------------------------------------------------------------------**/
  explicit reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns the entire data set.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read();

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns all the rows within a byte range.
   *
   * The returned data includes the row that straddles the end of the range.
   * In other words, a row is included as long as the row begins within the byte
   * range.
   *
   * @param[in] offset Byte offset from the start
   * @param[in] size Number of bytes from the offset; set to 0 for all remaining
   *
   * @return cudf::table object that contains the array of gdf_columns
   *---------------------------------------------------------------------------**/
  table read_byte_range(size_t offset, size_t size);

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns a range of rows.
   *
   * Set num_skip_footer to zero when using num_rows parameter.
   *
   * @param[in] num_skip_header Number of rows at the start of the files to skip
   * @param[in] num_skip_footer Number of rows at the bottom of the file to skip
   * @param[in] num_rows Number of rows to read. Value of -1 indicates all rows
   *
   * @return cudf::table object that contains the array of gdf_columns
   *---------------------------------------------------------------------------**/
  table read_rows(gdf_size_type num_skip_header, gdf_size_type num_skip_footer,
                  gdf_size_type num_rows = -1);

  ~reader();
};

}  // namespace csv

namespace orc {
/**---------------------------------------------------------------------------*
 * @brief Options for the ORC reader
 *---------------------------------------------------------------------------**/
struct reader_options {
  std::vector<std::string> columns;
  bool use_index = true;
  bool use_np_dtypes = true;
  gdf_time_unit timestamp_unit = TIME_UNIT_NONE;

  reader_options() = default;
  reader_options(reader_options const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief Constructor to populate reader options.
   *
   * @param[in] cols List of columns to read. If empty, all columns are read
   * @param[in] use_index_lookup Whether to use row index for faster scanning
   * @param[in] np_compat Whether to use numpy-compatible dtypes
   * @param[in] timestamp_time_unit Resolution of timestamps; none for default
   *---------------------------------------------------------------------------**/
  reader_options(std::vector<std::string> cols, bool use_index_lookup,
                 bool np_compat, gdf_time_unit timestamp_time_unit)
      : columns(std::move(cols)),
        use_index(use_index_lookup),
        use_np_dtypes(np_compat),
        timestamp_unit(timestamp_time_unit) {}
};

/**---------------------------------------------------------------------------*
 * @brief Class to read Apache ORC data into cuDF columns
 *---------------------------------------------------------------------------**/
class reader {
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructor for a file path source.
   *---------------------------------------------------------------------------**/
  explicit reader(std::string filepath, reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an existing memory buffer source.
   *---------------------------------------------------------------------------**/
  explicit reader(const char *buffer, size_t length,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an Arrow file source
   *---------------------------------------------------------------------------**/
  explicit reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns the entire data set.
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_all();

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns a specific stripe.
   *
   * @param[in] stripe Index of the stripe
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_stripe(size_t stripe);

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns a range of rows.
   *
   * @param[in] skip_rows Number of rows to skip from the start
   * @param[in] num_rows Number of rows to read; use `0` for all remaining data
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_rows(size_t skip_rows, size_t num_rows);

  ~reader();
};

}  // namespace orc

namespace parquet {
/**---------------------------------------------------------------------------*
 * @brief Options for the Parquet reader
 *---------------------------------------------------------------------------**/
struct reader_options {
  std::vector<std::string> columns;
  bool strings_to_categorical = false;
  bool use_pandas_metadata = false;
  gdf_time_unit timestamp_unit = TIME_UNIT_NONE;

  reader_options() = default;
  reader_options(reader_options const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief Constructor to populate reader options.
   *
   * @param[in] cols List of columns to read. If empty, all columns are read
   * @param[in] strings_to_categorical Whether to store strings as GDF_CATEGORY
   * @param[in] read_pandas_indexes Whether to always load PANDAS index columns
   * @param[in] timestamp_time_unit Resolution of timestamps; none for default
   *---------------------------------------------------------------------------**/
  reader_options(std::vector<std::string> cols, bool strings_as_category,
                 bool read_pandas_indexes, gdf_time_unit timestamp_time_unit)
      : columns(std::move(cols)),
        strings_to_categorical(strings_as_category),
        use_pandas_metadata(read_pandas_indexes),
        timestamp_unit(timestamp_time_unit) {}
};

/**---------------------------------------------------------------------------*
 * @brief Class to read Apache Parquet data into cuDF columns
 *---------------------------------------------------------------------------**/
class reader {
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructor for a file path source.
   *---------------------------------------------------------------------------**/
  explicit reader(std::string filepath, reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an existing memory buffer source.
   *---------------------------------------------------------------------------**/
  explicit reader(const char *buffer, size_t length,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Constructor for an Arrow file source
   *---------------------------------------------------------------------------**/
  explicit reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
                  reader_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Returns the PANDAS-specific index column derived from the metadata.
   *
   * @return std::string Name of the column if it exists.
   *---------------------------------------------------------------------------**/
  std::string get_index_column();

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns the entire data set.
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_all();

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns a specific row group.
   *
   * @param[in] row_group Index of the row group
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_row_group(size_t row_group);

  /**---------------------------------------------------------------------------*
   * @brief Reads and returns a range of rows.
   *
   * @param[in] skip_rows Number of rows to skip from the start
   * @param[in] num_rows Number of rows to read; use `0` for all remaining data
   *
   * @return cudf::table Object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_rows(size_t skip_rows, size_t num_rows);

  ~reader();
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
