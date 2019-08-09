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

#ifndef COLUMN_HPP
#define COLUMN_HPP

#include <cudf/cudf.h>
#include <cudf/types.h>

/**
 * @brief Concatenates multiple gdf_columns into a single, contiguous column,
 * including the validity bitmasks.
 *
 * Note that input columns with nullptr validity masks are treated as if all
 * elements are valid.
 *
 * @param[out] output_column A column whose buffers are already allocated that
 *             will contain the concatenation of the input columns data and
 *             validity bitmasks
 * @param[in] columns_to_concat[] The columns to concatenate
 * @param[in] num_columns The number of columns to concatenate
 *
 * @return gdf_error GDF_SUCCESS upon completion; GDF_DATASET_EMPTY if any data
 *         pointer is NULL, GDF_COLUMN_SIZE_MISMATCH if the output column size
 *         != the total size of the input columns; GDF_DTYPE_MISMATCH if the
 *         input columns have different datatypes.
 *
 */
gdf_error gdf_column_concat(gdf_column *output, gdf_column *columns_to_concat[],
                            int num_columns);

/**
 * @brief Return the size of the gdf_column data type.
 *
 * @returns gdf_size_type Size of the gdf_column data type.
 */
gdf_size_type gdf_column_sizeof();

/**
 * @brief Create a GDF column given data and validity bitmask pointers, size,
 * and datatype
 *
 * @param[out] column The output column.
 * @param[in] data Pointer to data.
 * @param[in] valid Pointer to validity bitmask for the data.
 * @param[in] size Number of rows in the column.
 * @param[in] dtype Data type of the column.
 *
 * @returns gdf_error returns GDF_SUCCESS upon successful creation.
 */
gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                          gdf_size_type size, gdf_dtype dtype);

/** 
 * @brief Create a GDF column given data and validity bitmask pointers, size, and
 *        datatype, and count of null (non-valid) elements
 *
 * @param[out] column The output column.
 * @param[in] data Pointer to data.
 * @param[in] valid Pointer to validity bitmask for the data.
 * @param[in] size Number of rows in the column.
 * @param[in] dtype Data type of the column.
 * @param[in] null_count The number of non-valid elements in the validity bitmask.
 * @param[in] extra_info see gdf_dtype_extra_info. Extra data for column description.
 * @param[in] name (optional) column name
 * 
 * @returns gdf_error returns GDF_SUCCESS upon successful creation.
 */
gdf_error gdf_column_view_augmented(gdf_column *column, void *data,
                                    gdf_valid_type *valid, gdf_size_type size,
                                    gdf_dtype dtype, gdf_size_type null_count,
                                    gdf_dtype_extra_info extra_info,
                                    const char *name = nullptr);

/**
 * @brief Free the CUDA device memory of a gdf_column
 *
 * @param[in,out] column Data and validity bitmask pointers of this column will
 * be freed
 *
 * @returns gdf_error GDF_SUCCESS or GDF_ERROR if there is an error freeing the
 * data
 */
gdf_error gdf_column_free(gdf_column *column);

namespace cudf {
namespace detail {

/**
 * @brief Allocate the data fields of column
 * 
 * @param column The column for which the data fields need to be allocated
 * @param allocate_mask Optional Whether or not to allocate bitmask
 * @param stream Optional stream in which to perform allocation
 */
void allocate_column_fields(gdf_column& column,
                            bool allocate_mask = true,
                            cudaStream_t stream = 0);

} // namespace detail

/**
 * @brief Allocates a new column of the specified size and type.
 * 
 * @param dtype Datatype of the column
 * @param size Size of the column
 * @param allocate_mask Optional Whether or not to allocate bitmask
 * @param info Optional Extra info for the column
 * @param stream Optional stream in which to perform allocation
 * @return gdf_column An allocated column of given size and type
 */
gdf_column allocate_column(gdf_dtype dtype, gdf_size_type size,
                           bool allocate_mask = true,
                           gdf_dtype_extra_info info = gdf_dtype_extra_info{},
                           cudaStream_t stream = 0);

} // namespace cudf

#endif
