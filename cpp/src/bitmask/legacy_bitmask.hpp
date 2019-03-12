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

/**---------------------------------------------------------------------------*
 * @brief Returns if the specified bit in a validity bit mask is set.
 *
 * @param valid The validity bitmask. If equal to `nullptr`, this function
 * always returns true
 * @param pos The specified bit
 * @return true If the bit is set (equal to 1), or if @p valid is nullptr
 * @return false If the bit is not set (equal to 0)
 *---------------------------------------------------------------------------**/
CUDA_HOST_DEVICE_CALLABLE
bool gdf_is_valid(const gdf_valid_type *valid, gdf_index_type pos) {
  if (valid)
    return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
  else
    return true;
}

/**
 * @brief Computes the number of `gdf_valid_type` elements required to provide
 * enough bits to represent the specified number of column elements.
 *
 * @note Note that this function assumes that the size of `gdf_valid_type` is 1
 * byte
 * @note This function is different gdf_valid_allocation_size
 * because gdf_valid_allocation_size returns the number of bytes required to
 * satisfy 64B padding. This function should be used when needing to access the
 * last `gdf_valid_type` element in the validity bitmask.
 *
 * @param[in] column_size the number of elements
 * @return The minimum number of `gdf_valid_type` elements to provide sufficient
 * bits to represent elements in a column of size @p column_size
 */
CUDA_HOST_DEVICE_CALLABLE
gdf_size_type gdf_last_bitmask_index(gdf_size_type column_size) {
  return ((column_size + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE);
}


#endif
