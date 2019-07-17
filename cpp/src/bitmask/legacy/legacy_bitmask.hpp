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

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>

/**---------------------------------------------------------------------------*
 * @brief Returns true if the specified bit in a validity bit mask is set.
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

#endif
