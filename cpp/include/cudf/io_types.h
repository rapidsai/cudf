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
