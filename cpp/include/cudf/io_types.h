/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

/*
 * @brief Enumeration of supported input types for CSV reader
 *
 * TODO: Remove and use gdf_input_type directly. This typedef is to reduce the
 * initial changes for Parquet review by not changing/including any CSV code.
 */
typedef gdf_input_type gdf_csv_input_form;

/*
 * @brief Enumeration of quoting behavior for CSV readers/writers
 */
typedef enum 
{
  QUOTE_MINIMAL,                            ///< Only quote those fields which contain special characters; enable quotation when parsing.
  QUOTE_ALL,                                ///< Quote all fields; enable quotation when parsing.
  QUOTE_NONNUMERIC,                         ///< Quote all non-numeric fields; enable quotation when parsing.
  QUOTE_NONE                                ///< Never quote fields; disable quotation when parsing.
} gdf_csv_quote_style;

/**---------------------------------------------------------------------------*
 * @brief  This struct contains all input parameters to the read_csv function.
 * Also contains the output dataframe.
 *
 * Input parameters are all stored in host memory. The output dataframe is in 
 * the device memory.
 *
 * Parameters in PANDAS that are unavailable in cudf:
 *   squeeze          - data is always returned as a gdf_column array
 *   engine           - this is the only engine
 *   verbose
 *   keep_date_col    - will not maintain raw data
 *   date_parser      - there is only this parser
 *   float_precision  - there is only one converter that will cover all specified values
 *   dialect          - not used
 *---------------------------------------------------------------------------**/
typedef struct {

  /*
   * Output Arguments - allocated in reader.
   */
  int           num_cols_out;               ///< Out: return the number of columns read in
  int           num_rows_out;               ///< Out: return the number of rows read in
  gdf_column    **data;                     ///< Out: return the array of *gdf_columns

  /*
   * Input arguments - all data is in the host memory
   */
  gdf_csv_input_form	input_data_form;	///< Type of source of CSV data
  const char			*filepath_or_buffer; ///< If input_data_form is FILE_PATH, contains the filepath. If input_data_type is HOST_BUFFER, points to the host memory buffer
  size_t				buffer_size;		///< If input_data_form is HOST_BUFFER, represents the size of the buffer in bytes. Unused otherwise

  bool          windowslinetermination;     ///< States if we should \r\n as our line termination
  char          lineterminator;             ///< define the line terminator character. Default is  '\n'
  char          delimiter;                  ///< define the field separator, default is ',' This argument is also called 'sep' 
  bool          delim_whitespace;           ///< Use white space as the delimiter - default is false. This overrides the delimiter argument
  bool          skipinitialspace;           ///< Skip white spaces after the delimiter - default is false

  gdf_size_type nrows;                      ///< Number of rows to read, -1 indicates all
  gdf_size_type header;                     ///< Row of the header data, zero based counting. Default states that header should not be read from file.

  int           num_cols;                   ///< Number of columns in the names and dtype arrays
  const char    **names;                    ///< Ordered List of column names
  const char    **dtype;                    ///< Ordered List of data types

  int           *index_col;                 ///< Indexes of columns to use as the row labels of the DataFrame.
  int           *use_cols_int;              ///< Indexes of columns to be returned. CSV reader will only process those columns, another read is needed to get full data
  int           use_cols_int_len;           ///< Number of elements in use_cols_int
  const char    **use_cols_char;            ///< Names of columns to be returned. CSV reader will only process those columns, another read is needed to get full data
  int           use_cols_char_len;          ///< Number of elements in use_cols_char_len

  gdf_size_type skiprows;                   ///< Number of rows at the start of the files to skip, default is 0
  gdf_size_type skipfooter;                 ///< Number of rows at the bottom of the file to skip - default is 0

  bool          skip_blank_lines;           ///< Indicates whether to ignore empty lines, or parse and interpret values as NaN 

  const char    **true_values;              ///< List of values to recognize as boolean True
  int           num_true_values;            ///< Number of values in the true_values list
  const char    **false_values;             ///< List of values to recognize as boolean False
  int           num_false_values;           ///< Number of values in the true_values list

  const char    **na_values;                /**< Array of strings that should be considered as NA. By default the following values are interpreted as NaN: 
                                            '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL',
                                            'NaN', 'n/a', 'nan', 'null'. */
  int           num_na_values;              ///< Number of values in the na_values list
  bool          keep_default_na;            ///< Keep the default NA values
  bool          na_filter;                  ///< Detect missing values (empty strings and the values in na_values). Passing false can improve performance.

  char          *prefix;                    ///< If there is no header or names, prepend this to the column ID as the name
  bool          mangle_dupe_cols;           ///< If true, duplicate columns get a suffix. If false, data will be overwritten if there are columns with duplicate names

  bool          parse_dates;                // Parse date field into date32 or date64.  If false then date fields are saved as a string. Specifying a date dtype overrides this
  bool          infer_datetime_format;      // Try and determine the date format
  bool          dayfirst;                   ///< Is the first value in the date formatthe day?  DD/MM  versus MM/DD

  char          *compression;               ///< Specify the type of compression (nullptr,"none","infer","gzip","zip"), "infer" infers the compression from the file extension, default(nullptr) is uncompressed
  char          thousands;                  ///< Single character that separates thousands in numeric data. If this matches the delimiter then system will return GDF_INVALID_API_CALL

  char          decimal;                    ///< The decimal point character. If this matches the delimiter then system will return GDF_INVALID_API_CALL

  char          quotechar;                  ///< Define the character used to denote start and end of a quoted item
  gdf_csv_quote_style quoting;              ///< Treat string fields as quoted item and remove the first and last quotechar
  bool          doublequote;                ///< Indicates whether to interpret two consecutive quotechar inside a field as a single quotechar

  char          escapechar;                 // Single character used as the escape character

  char          comment;                    ///< The character used to denote start of a comment line. The rest of the line will not be parsed.

  char          *encoding;                  // the data encoding, NULL = UTF-8

  size_t        byte_range_offset;          ///< offset of the byte range to read. 
  size_t        byte_range_size;            /**< size of the byte range to read. Set to zero to read all data after byte_range_offset.
                                            Reads the row that starts before or at the end of the range, even if it ends after the end of the range. */
} csv_read_arg;

/**---------------------------------------------------------------------------*
 * @brief These are the arguments to the CSV writer function.
 *---------------------------------------------------------------------------**/
typedef struct
{
    gdf_column** columns;         // columns to output
    int num_cols;                 // number of columns

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
