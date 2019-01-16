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
  FILE_PATH,								//< Indicates that the input is specified with a file path
  HOST_BUFFER								//< Indicates that the input is passed as a buffer in host memory
} gdf_csv_input_form;

typedef struct {

  /*
   * Output Arguments - space created in reader.
   */
  int           num_cols_out;               /**< Out: return the number of columns read in  */
  int           num_rows_out;               /**< Out: return the number of rows read in     */
  gdf_column    **data;                     /**< Out: return the array of *gdf_columns      */

  /*
   * Input arguments - all data is in the host
   */
  gdf_csv_input_form	input_data_form;	/**< Type of source of CSV data */
  const char			*filepath_or_buffer;/**< If input_data_form is FILE_PATH, contains the filepath. If input_data_type is HOST_BUFFER, points to the host memory buffer*/
  size_t				buffer_size;		/**< If input_data_form is HOST_BUFFER, represents the size of the buffer in bytes. Unused otherwise */

  bool          windowslinetermination;     /**< States if we should \r\n as our line termination>**/
  char          lineterminator;             /**< define the line terminator character.  Default is  '\n'                                        */
  char          delimiter;                  /**< define the field separator, default is ','   This argument is also called 'sep'                */
  bool          delim_whitespace;           /**< use white space as the delimiter - default is false.  This overrides the delimiter argument    */
  bool          skipinitialspace;           /**< skip white spaces after the delimiter - default is false                                       */


  int           nrows;                      // number of rows to read,  -1 indicates all
  int           header;                     // Row of the header data,  zero based counting. Default states that header should not be read from file.

  int           num_cols;                   /**< number of columns in the names and dtype arrays                                                */
  const char    **names;                    /**< ordered List of column names, this is a required field                                         */
  const char    **dtype;                    /**< ordered List of data types, this is required                                                   */

  int           *index_col;                 // array of int:    Column to use as the row labels of the DataFrame.
  int           *use_cols_int;              // array of int:    Return a subset of the columns.  CSV reader will only process those columns,  another read is needed to get full data
  int           use_cols_int_len;           // int:    number of elements in list of returned columns
  const char    **use_cols_char;            // array of char:    Return a subset of the columns.  CSV reader will only process those columns,  another read is needed to get full data
  int           use_cols_char_len;          // int:    number of elements in list of returned columns

  gdf_size_type skiprows;                   /**< number of rows at the start of the files to skip, default is 0                                 */
  gdf_size_type skipfooter;                 /**< number of rows at the bottom of the file to skip - default is 0                                */

  bool          skip_blank_lines;           // whether or not to ignore blank lines

  const char    **true_values;              /**< list of values to recognize as boolean True */
  int           num_true_values;            /**< number of values in the true_values list */
  const char    **false_values;             /**< list of values to recognize as boolean False */
  int           num_false_values;           /**< number of values in the true_values list */

  const char    **na_values;                // array of char *    what should be considered as True - each char string contains {col ID, na value, ...} - this will allow multiple na values to be specified over multiple columns

  char          *prefix;                    // if there is no header or names, append this to the column ID as the name
  bool          mangle_dupe_cols;           // if true: duplicate columns will be specified as (deleted because utf-8 chars kill the build)

  bool          parse_dates;                // parse date field into date32 or date64.  If false then date fields are saved as a string. Specifying a date dtype overrides this
  bool          infer_datetime_format;      // try and determine the date format
  bool          dayfirst;                   // is the first value the day?  DD/MM  versus MM/DD

  char          *compression;               /**< specify the type of compression (nullptr,"none","infer","gzip","zip"), "infer" infers the compression from the file extension, default(nullptr) is uncompressed */
  char          thousands;                  /**< single character that separates thousands in numeric data. If this matches the delimiter then system will return GDF_INVALID_API_CALL */

  char          decimal;                    /**< the decimal point character. If this matches the delimiter then system will return GDF_INVALID_API_CALL */

  char          quotechar;                  /**< define the character used to denote start and end of a quoted item                             */
  bool          quoting;                    /**< treat string fields as quoted item and remove the first and last quotechar                     */
  bool          doublequote;                /**< indicates whether to interpret two consecutive quotechar inside a field as a single quotechar  */

  char          escapechar;                 // single char    - char used as the escape character

  char          comment;                    // single char    - treat line or remainder of line as an unprocessed comment

  char          *encoding;                  // the data encoding, NULL = UTF-8

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
