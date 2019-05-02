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

#ifndef BITMASK_HPP
#define BITMASK_HPP

#include <cudf.h>
#include <types.hpp>

/**
 * @brief  Counts the number of valid bits for the specified number of rows
 * in a validity bitmask.
 *
 * If the bitmask is null, returns a count equal to the number of rows.
 *
 * @param[in] masks The validity bitmask buffer in device memory
 * @param[in] num_rows The number of bits to count
 * @param[out] count The number of valid bits in the buffer from [0, num_rows)
 *
 * @returns  GDF_SUCCESS upon successful completion
 *
 */
gdf_error gdf_count_nonzero_mask(gdf_valid_type const* masks,
                                 gdf_size_type num_rows, gdf_size_type* count);

/** ---------------------------------------------------------------------------*
 * @brief Concatenate the validity bitmasks of multiple columns
 *
 * Accounts for the differences between lengths of columns and their bitmasks
 * (e.g. because gdf_valid_type is larger than one bit).
 *
 * @param[out] output_mask The concatenated mask
 * @param[in] output_column_length The total length (in data elements) of the
 *                                 concatenated column
 * @param[in] masks_to_concat The array of device pointers to validity bitmasks
 *                            for the columns to concatenate
 * @param[in] column_lengths An array of lengths of the columns to concatenate
 * @param[in] num_columns The number of columns to concatenate
 * @return gdf_error GDF_SUCCESS or GDF_CUDA_ERROR if there is a runtime CUDA
           error
 *
 ---------------------------------------------------------------------------**/
gdf_error gdf_mask_concat(gdf_valid_type* output_mask,
                          gdf_size_type output_column_length,
                          gdf_valid_type* masks_to_concat[],
                          gdf_size_type* column_lengths,
                          gdf_size_type num_columns);


#endif
