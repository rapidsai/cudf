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
 *
 * gdf_column ** read_csv(csv_read_arg *args,  int *num_col_out, gdf_error *error_out);
 *
 */
#pragma once

typedef struct {

  /*
   * Output Arguments - space created in reader.
   */
  int			num_cols_out;				/**< Out: return the number of columns read in	*/
  int			num_rows_out;				/**< Out: return the number of rows read in 	*/
  gdf_column	**data;						/**< Out: return the array of *gdf_columns 		*/
									

  /*
   * Input arguments - all data is in the host
   */
  char			*file_path;					/**< file location to read from	- currently the file cannot be compressed 							*/
  char			*buffer	;					// process data from a buffer,  pointer to Host memory
  char			*object	;					// this is a URL path

  bool			windowslinetermination;		/**< States if we should \r\n as our line termination>**/
  char			lineterminator;				/**< define the line terminator character.  Default is  '\n'  										*/
  char			delimiter;					/**< define the field separator, default is ','   This argument is also called 'sep'  				*/
  bool			delim_whitespace;			/**< use white space as the delimiter - default is false.  This overrides the delimiter argument 	*/
  bool			skipinitialspace;			/**< skip white spaces after the delimiter - default is false  										*/

  int			nrows;						// number of rows to read,  -1 indicates all
  int			header;						// Row of the header data,  zero based counting. Default states that header should not be read from file.

  int			num_cols;					/**< number of columns in the names and dtype arrays												*/
  const char	**names;					/**< ordered List of column names, this is a required field 										*/
  const char	**dtype;					/**< ordered List of data types, this is required													*/

  int			*index_col;					// array of int:	Column to use as the row labels of the DataFrame.
  int			*use_cols_int;				// array of int:	Return a subset of the columns.  CSV reader will only process those columns,  another read is needed to get full data
  int			use_cols_int_len;			// int:	number of elements in list of returned columns
  const char	**use_cols_char;			// array of char:	Return a subset of the columns.  CSV reader will only process those columns,  another read is needed to get full data
  int			use_cols_char_len;			// int:	number of elements in list of returned columns

  long			skiprows;					/**< number of rows at the start of the files to skip, default is 0									*/
  long			skipfooter;					/**< number of rows at the bottom of the file to skip - default is 0								*/

  bool			skip_blank_lines;			// whether or not to ignore blank lines

  char			**true_values;				// array of char *	what should be considered as True - each char string contains {col ID, true value, ...} - this will allow multiple true values to be specified over multiple columns
  char			**false_values;				// array of char *	what should be considered as True - each char string contains {col ID, false value, ...} - this will allow multiple false values to be specified over multiple columns
  char			**na_values;				// array of char *	what should be considered as True - each char string contains {col ID, na value, ...} - this will allow multiple na values to be specified over multiple columns
  char			*prefix;					// if there is no header or names, append this to the column ID as the name
  bool			mangle_dupe_cols;			// if true: duplicate columns will be specified as (deleted because utf-8 chars kill the build)

  bool			parse_dates;				// parse date field into date32 or date64.  If false then date fields are saved as a string. Specifying a date dtype overrides this
  bool			infer_datetime_format;		// try and determine the date format
  bool			dayfirst;					// is the first value the day?  DD/MM  versus MM/DD

  char			*compression;				// specify the type of compression
  char			*thousands;					// single character		a separate within numeric data  - if this matches the delimiter then system will return NULL

  char			decimal;					// the decimal point character

  char			*quotechar;					// single character 	character to use as a quote
  bool			doublequote;				// whether to treat two quotes as a quote in a string or empty

  char			escapechar;					// single char	- char used as the escape character

  char			comment;					// single char	- treat line or remainder of line as an unprocessed comment

  char			*encoding;					// the data encoding, NULL = UTF-8

  bool			keepQuotes;					// TRUE: keep the quotes in the text. FALSE: remove the quotes from the strings. Only for quotes starting and ending the strings.

} csv_read_arg;



/*
 * NOT USED
 *
 * squeeze			- data is always returned as a gdf_column array
 * engine			- this is the only engine
 * keep_default_na  - this has no meaning since the field is marked invalid and the value not seen
 * na_filter		- empty fields are automatically tagged as invalid
 * verbose
 * keep_date_col	- will not maintain raw data
 * date_parser		- there is only this parser
 * float_precision	- there is only one converter that will cover all specified values
 * quoting			- this is for out
 * dialect			- not used
 *
 */




