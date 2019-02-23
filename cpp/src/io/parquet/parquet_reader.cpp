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

#include "cudf.h"

#include <iostream>

/**---------------------------------------------------------------------------*
 * @brief Reads Apache Parquet-formatted data and returns an allocated array of
 * gdf_columns.
 * 
 * @param[in,out] args Structure containing input and output args 
 * 
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 *---------------------------------------------------------------------------**/
gdf_error read_parquet(pq_read_arg *args) {

  // Input parsing options
  if (args->use_cols) {
    std::cout << std::endl;
    for (int i = 0; i < args->use_cols_len; ++i) {
      std::cout << "Use column: " << args->use_cols[i] << std::endl;
    }
  }

  // Output return data
  auto rows_count = 1;
  auto cols_count = 1;
	auto cols = static_cast<gdf_column**>(malloc(sizeof(gdf_column*) * cols_count));
  for (int i = 0; i < cols_count; ++i) {
    // Populate columns fields
  }
  args->data = cols;
  args->num_cols_out = cols_count;
  args->num_rows_out = rows_count;

  return GDF_SUCCESS;
}
