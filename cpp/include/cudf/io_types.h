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
    bool include_header;          // Indicates whether to write headers to csv

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

  bool          use_index;                  ///< In: Use row index if available for faster position seeking

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
