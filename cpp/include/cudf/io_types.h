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

/*
 * API
 *
 * gdf_error read_csv(csv_read_arg *args);
 *
 */
#pragma once

 /*
   * Enumerator for the supported forms of the input CSV file
   */
typedef enum 
{
  FILE_PATH,								///< Indicates that the input is specified with a file path
  HOST_BUFFER								///< Indicates that the input is passed as a buffer in host memory
} gdf_csv_input_form;

/**---------------------------------------------------------------------------*
 * @brief  This struct contains all input parameters to the read_csv function.
 * Also contains the output dataframe.
 *
 * Input parameters are all stored in host memory. The output dataframe is in 
 * the device memory.
 *
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

  bool          skip_blank_lines;           // Whether or not to ignore blank lines

  const char    **true_values;              ///< List of values to recognize as boolean True
  int           num_true_values;            ///< Number of values in the true_values list
  const char    **false_values;             ///< List of values to recognize as boolean False
  int           num_false_values;           ///< Number of values in the true_values list

  const char    **na_values;                // Array of strings that should be considered as NA

  char          *prefix;                    // If there is no header or names, append this to the column ID as the name
  bool          mangle_dupe_cols;           ///< If true, duplicate columns get a suffix. If false, data will be overwritten if there are columns with duplicate names

  bool          parse_dates;                // Parse date field into date32 or date64.  If false then date fields are saved as a string. Specifying a date dtype overrides this
  bool          infer_datetime_format;      // Try and determine the date format
  bool          dayfirst;                   ///< Is the first value in the date formatthe day?  DD/MM  versus MM/DD

  char          *compression;               ///< Specify the type of compression (nullptr,"none","infer","gzip","zip","bz2"), "infer" infers the compression from the file extension, default(nullptr) is uncompressed
  char          thousands;                  ///< Single character that separates thousands in numeric data. If this matches the delimiter then system will return GDF_INVALID_API_CALL

  char          decimal;                    ///< The decimal point character. If this matches the delimiter then system will return GDF_INVALID_API_CALL

  char          quotechar;                  ///< Define the character used to denote start and end of a quoted item
  bool          quoting;                    ///< Treat string fields as quoted item and remove the first and last quotechar
  bool          doublequote;                ///< Indicates whether to interpret two consecutive quotechar inside a field as a single quotechar

  char          escapechar;                 ///< Single character used as the escape character

  char          comment;                    // Single character indicating that the remainder of line is a comment

  char          *encoding;                  // the data encoding, NULL = UTF-8

  size_t        byte_range_offset;          ///< offset of the byte range to read. 
  size_t        byte_range_size;            /**< size of the byte range to read. Set to zero to read all data after byte_range_offset.
                                            Reads the row that starts before or at the end of the range, even if it ends after the end of the range. */

} csv_read_arg;


/*
 * NOT USED
 *
 * squeeze          - data is always returned as a gdf_column array
 * engine           - this is the only engine
 * keep_default_na  - this has no meaning since the field is marked invalid and the value not seen
 * na_filter        - empty fields are automatically tagged as invalid
 * verbose
 * keep_date_col    - will not maintain raw data
 * date_parser      - there is only this parser
 * float_precision  - there is only one converter that will cover all specified values
 * dialect          - not used
 *
 */
