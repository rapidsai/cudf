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

#ifndef LEGACY_BITMASK_HPP
#define LEGACY_BITMASK_HPP

#include <cudf.h>
#include <utilities/cudf_utils.h>

CUDA_HOST_DEVICE_CALLABLE
bool gdf_is_valid(const gdf_valid_type *valid, gdf_index_type pos) {
  if (valid)
    return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
  else
    return true;
}

/**
 * Calculates the index of the last `gdf_valid_type` element in the validity
 * bitmask for a given column's size.
 *
 * @note Note that this function assumes that `gdf_valid_type` is unsigned char
 * @note This function is different gdf_get_num_bytes_for_valids_allocation
 * because it refers to the last `gdf_valid_type` element that refers to
 * elements in the column, NOT the last element in the allocation
 *
 * @param[in] column_size the number of elements
 * @return The index of the last `gdf_valid_type` element that refers to
 * elements in a column of size @p column_size
 */
CUDA_HOST_DEVICE_CALLABLE
gdf_size_type gdf_last_bitmask_index(gdf_size_type column_size) {
  return ((column_size + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE);
}


#endif
