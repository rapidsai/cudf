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

#ifndef BITMASK_OPS_HPP
#define BITMASK_OPS_HPP

#include <types.hpp>
#include <cuda_runtime.h>

#include <rmm/thrust_rmm_allocator.h>

/**---------------------------------------------------------------------------*
 * @file bitmask_ops.hpp
 * @brief Internal functions for bitmask operations.
*---------------------------------------------------------------------------**/

/**---------------------------------------------------------------------------*
 * @brief Sets all bits in input valid mask to 1
 *
 * @param valid_out preallocated output valid mask
 * @param out_null_count number of nulls (0 bits) in valid mask. Always set to 0
 * @param num_values number of values in column associated with output mask
 * @param stream cuda stream to run in
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error all_bitmask_on(gdf_valid_type* valid_out,
                         gdf_size_type& out_null_count,
                         gdf_size_type num_values, cudaStream_t stream);

/**---------------------------------------------------------------------------*
 * @brief Computes bitwise AND on two valid masks and sets it in output
 *
 * @param out_null_count number of nulls (0 bits) in output valid mask
 * @param valid_out preallocated mask to set the result values in
 * @param valid_left input valid mask 1
 * @param valid_right input valid mask 2
 * @param stream cuda stream to run in
 * @param num_values number of values in each input mask valid_left and
 * valid_right
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error apply_bitmask_to_bitmask(gdf_size_type& out_null_count,
                                   gdf_valid_type* valid_out,
                                   gdf_valid_type* valid_left,
                                   gdf_valid_type* valid_right,
                                   cudaStream_t stream,
                                   gdf_size_type num_values);


namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Computes a bitmask indicating the presence of NULL values in rows of a
 * table.
 *
 * If a row `i` in `table` contains one or more NULL values, then bit `i` in the
 * returned bitmask will be 0.
 *
 * Otherwise, bit `i` will be 1.
 *
 * @param table The table to compute the row bitmask of.
 * @return bit_mask::bit_mask_t* The bitmask indicating the presence of NULLs in
 * a row
 *---------------------------------------------------------------------------**/
rmm::device_vector<bit_mask::bit_mask_t> row_bitmask(cudf::table const& table,
                                                     cudaStream_t stream = 0);
}  // namespace cudf

#endif
