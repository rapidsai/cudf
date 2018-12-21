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

#include "cudf.h"

/**
 * @brief Operations for copying from one column to another
 * @file copying_ops.cu
 */

gdf_error gdf_scatter(gdf_dataframe const* source_columns,
                      gdf_size_type const scatter_map[],
                      gdf_dataframe* destination_columns) {
  return GDF_SUCCESS;
}

gdf_error gdf_gather(gdf_dataframe const* source_columns,
                     gdf_size_type const gather_map[],
                     gdf_dataframe* destination_columns) {
  return GDF_SUCCESS;
}
