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


	int				num_cols_out;			/**< Out: return the number of columns read in	*/
	int				num_rows_out;			/**< Out: return the number of rows read in 	*/
	gdf_column		**data;					/**< Out: return the array of *gdf_columns 		*/

	/*
	 * Input arguments - all data is in the host
	 */
	char		*file_path;					/**< file location to read from	- currently the file cannot be compressed 							*/

	char		lineterminator;				/**< define the line terminator character.  Default is  '\n'  										*/
	char		delimiter;					/**< define the field separator, default is ','   This argument is also called 'sep'  				*/
	bool 		delim_whitespace;			/**< use white space as the delimiter - default is false.  This overrides the delimiter argument 	*/
	bool		skipinitialspace;			/**< skip white spaces after the delimiter - default is false  										*/

	int			num_cols;					/**< number of columns in the names and dtype arrays												*/
	const char	**names;					/**< ordered List of column names, this is a required field 										*/
	const char	**dtype;					/**< ordered List of data types, this is required													*/

	int			skiprows;					/**< number of rows at the start of the files to skip, default is 0									*/
	int			skipfooter;					/**< number of rows at the bottom of the file to skip - default is 0								*/

	bool		dayfirst;					/**< is the first value the day?  DD/MM  versus MM/DD, default is false								*/



} csv_read_arg;






