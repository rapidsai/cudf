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

/*
 * @brief Enumeration of supported input types for cudf reader interfaces
 */
typedef enum {
  FILE_PATH,                 ///< Source is specified as a file path
  HOST_BUFFER,               ///< Source is specified as a buffer in host memory,
  ARROW_RANDOM_ACCESS_FILE,  ///< Source is specified as an arrow::io::RandomAccessFile
} gdf_input_type;


/**---------------------------------------------------------------------------*
 * @brief These are the arguments to the CSV writer function.
 *---------------------------------------------------------------------------**/
typedef struct
{
    const gdf_column* const* columns; // columns to output
    int num_cols;                     // number of columns

    const char* filepath;         // full path to file to create
    const char* line_terminator;  // character to use for separating lines (default "\n")
    char delimiter;               // character to use between each column entry (default ',')

    const char* true_value;       // string to use for values !=0 in GDF_INT8 types (default 'true')
    const char* false_value;      // string to use for values ==0 in GDF_INT8 types (default 'false')
    const char* na_rep;           // string to use for null entries

} csv_write_arg;

/**---------------------------------------------------------------------------*
 * @brief Input and output arguments to the read_orc interface.
 *---------------------------------------------------------------------------**/
typedef struct {

  /*
   * Output arguments
   */
  int           num_cols_out;               ///< Out: Number of columns returned
  int           num_rows_out;               ///< Out: Number of rows returned
  gdf_column    **data;                     ///< Out: Array of gdf_columns*

  /*
   * Input arguments
   */
  gdf_input_type source_type;               ///< In: Type of data source
  const char    *source;                    ///< In: If source_type is FILE_PATH, contains the filepath. If input_data_type is HOST_BUFFER, points to the host memory buffer
  size_t        buffer_size;                ///< In: If source_type is HOST_BUFFER, represents the size of the buffer in bytes. Unused otherwise.

  const char    **use_cols;                 ///< In: Columns of interest; only these columns will be parsed and returned.
  int           use_cols_len;               ///< In: Number of columns

  int           stripe;                     ///< In: Stripe index of interest; only data in this stripe will be returned.
  int           skip_rows;                  ///< In: Number of rows to skip from the start
  int           num_rows;                   ///< In: Number of rows to read. Actual number of returned rows may be less

} orc_read_arg;

/**---------------------------------------------------------------------------*
 * @brief Input and output arguments to the read_parquet interface.
 *---------------------------------------------------------------------------**/
typedef struct {

  /*
   * Output arguments
   */
  int           num_cols_out;               ///< Out: Number of columns returned
  int           num_rows_out;               ///< Out: Number of rows returned
  gdf_column    **data;                     ///< Out: Array of gdf_columns*
  int           *index_col;                 ///< Out: If available, column index to use as row labels

  /*
   * Input arguments
   */
  gdf_input_type source_type;               ///< In: Type of data source
  const char    *source;                    ///< In: If source_type is FILE_PATH, contains the filepath. If input_data_type is HOST_BUFFER, points to the host memory buffer
  size_t        buffer_size;                ///< In: If source_type is HOST_BUFFER, represents the size of the buffer in bytes. Unused otherwise.

  int           row_group;                  ///< In: Row group index of interest; only data in this row group will be returned.
  int           skip_rows;                  ///< In: Rows to skip from the start of the dataset
  int           num_rows;                   ///< In: Number of rows to read and return

  const char    **use_cols;                 ///< In: Columns of interest; only these columns will be parsed and returned.
  int           use_cols_len;               ///< In: Number of columns

  bool          strings_to_categorical;     ///< In: If TRUE, returns string data as GDF_CATEGORY, otherwise GDF_STRING

} pq_read_arg;

#include <vector>
#include <string>

namespace cudf{

/*
 * @brief Enumeration of supported input types for CSV reader
 *
 * TODO: Remove and use gdf_input_type directly. This typedef is to reduce the
 * initial changes for Parquet review by not changing/including any CSV code.
 */
using gdf_csv_input_form = gdf_input_type;

/*
 * @brief Enumeration of quoting behavior for CSV readers/writers
 */
enum gdf_csv_quote_style{
  QUOTE_MINIMAL,                            ///< Only quote those fields which contain special characters; enable quotation when parsing.
  QUOTE_ALL,                                ///< Quote all fields; enable quotation when parsing.
  QUOTE_NONNUMERIC,                         ///< Quote all non-numeric fields; enable quotation when parsing.
  QUOTE_NONE                                ///< Never quote fields; disable quotation when parsing.
};

/**---------------------------------------------------------------------------*
 * @brief  This struct contains all input parameters to the read_csv function.
 *
 * Parameters are all stored in host memory.
 *
 * Parameters in PANDAS that are unavailable in cudf:
 *   squeeze          - data is always returned as a gdf_column array
 *   engine           - this is the only engine
 *   verbose
 *   keep_date_col    - will not maintain raw data
 *   date_parser      - there is only this parser
 *   float_precision  - there is only one converter that will cover all specified values
 *   dialect          - not used
 *   encoding         - always use UTF-8
 *   escapechar       - always use '\'
 *   parse_dates      - infer date data types and always parse as such
 *   infer_datetime_format - inference not supported

 *---------------------------------------------------------------------------**/
struct csv_read_arg{
  gdf_csv_input_form  input_data_form;      ///< Type of source of CSV data
  const char          *filepath_or_buffer;  ///< If input_data_form is FILE_PATH, contains the filepath. If input_data_type is HOST_BUFFER, points to the host memory buffer
  size_t              buffer_size;          ///< If input_data_form is HOST_BUFFER, represents the size of the buffer in bytes. Unused otherwise

  char          lineterminator = '\n';      ///< Define the line terminator character; Default is '\n'.
  char          delimiter = ',';            ///< Define the field separator; Default is ','.
  bool          delim_whitespace = false;   ///< Use white space as the delimiter; default is false. This overrides the delimiter argument.
  bool          skipinitialspace = false;   ///< Skip white spaces after the delimiter; default is false.

  gdf_size_type nrows = -1;                 ///< Number of rows to read; default value, -1, indicates all rows.
  gdf_size_type header = 0;                 ///< Row of the header data, zero based counting; Default is zero.

  std::vector<std::string> names;           ///< Ordered List of column names; Empty by default.
  std::vector<std::string> dtype;           ///< Ordered List of data types; Empty by default.

  std::vector<int> use_cols_indexes;        ///< Indexes of columns to be processed and returned; Empty by default - process all columns.
  std::vector<std::string> use_cols_names;  ///< Names of columns to be processed and returned; Empty by default - process all columns.

  gdf_size_type skiprows = 0;               ///< Number of rows at the start of the files to skip; default is zero.
  gdf_size_type skipfooter = 0;             ///< Number of rows at the bottom of the file to skip; default is zero.

  bool skip_blank_lines = true;             ///< Indicates whether to ignore empty lines, or parse and interpret values as NA. Default value is true.

  std::vector<std::string> true_values;     ///< List of values to recognize as boolean True; Empty by default.
  std::vector<std::string> false_values;    ///< List of values to recognize as boolean False; Empty by default.

  std::vector<std::string> na_values;       /**< Array of strings that should be considered as NA. By default the following values are interpreted as NA: 
                                            '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL',
                                            'NaN', 'n/a', 'nan', 'null'. */
  bool          keep_default_na = true;     ///< Keep the default NA values; true by default.
  bool          na_filter = true;           ///< Detect missing values (empty strings and the values in na_values); true by default. Passing false can improve performance.

  std::string   prefix;                     ///< If there is no header or names, prepend this to the column ID as the name; Default value is an empty string.
  bool          mangle_dupe_cols = true;    ///< If true, duplicate columns get a suffix. If false, data will be overwritten if there are columns with duplicate names; true by default.

  bool          dayfirst = false;           ///< Is day the first value in the date format (DD/MM versus MM/DD)? false by default.

  std::string   compression = "infer";      ///< Compression type ("none", "infer", "bz2", "gz", "xz", "zip"); with the default value, "infer", infers the compression from the file extension.
  char          thousands = '\0';           ///< Single character that separates thousands in numeric data; default is '\0'. Should not match the delimiter.

  char          decimal = '.';              ///< The decimal point character; default is '.'. Should not match the delimiter.

  char          quotechar = '\"';           ///< Define the character used to denote start and end of a quoted item; default is '\"'.
  gdf_csv_quote_style quoting = QUOTE_MINIMAL; ///< Defines reader's quoting behavior; default is QUOTE_MINIMAL.
  bool          doublequote = true;         ///< Indicates whether to interpret two consecutive quotechar inside a field as a single quotechar; true by default.


  char          comment = '\0';             ///< The character used to denote start of a comment line. The rest of the line will not be parsed. The default is '\0'.


  size_t        byte_range_offset = 0;      ///< Offset of the byte range to read; default is zero.
  size_t        byte_range_size = 0;        /**< Size of the byte range to read. Set to zero to read to the end of the file (default behavior).
                                            Reads the row that starts before or at the end of the range, even if it ends after the end of the range.*/
  csv_read_arg() = default;

};

}
