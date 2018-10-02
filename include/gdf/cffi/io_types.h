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
	int				num_cols_out;							// Out: number of columns
	int				num_rows_out;							// Out: number of rows
	gdf_column		**data;								// Out: array of *gdf_columns

	/*
	 * Input arguments - all data is in the host
	 */
	char		*file_path;				// file location to read from	- if the file is compressed, it needs proper file extensions {.gz}

	char		lineterminator;			// can change the end of line character
	char		delimiter;				// also called 'sep'  this is the field separator
	bool 		delim_whitespace;		// use white space as the delimiter
	bool		skipinitialspace;		// Skip spaces after delimiter

	int			num_cols;				// number of columns (array sizes)
	const char	**names;				// array of char *  Ordered List of column names to use.   names cannot be used with header
	const char	**dtype;				// array of char *	Ordered List of data types as strings

	int			skiprows;				// number of rows at the start of the files to skip
	int			skipfooter;				// number of rows at the bottom of the file to skip - counting is backwards from end, 0 = last line

	bool		dayfirst;				// is the first value the day?  DD/MM  versus MM/DD


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




