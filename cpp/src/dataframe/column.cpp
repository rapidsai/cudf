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
/** ---------------------------------------------------------------------------*
 * @brief Operations on GDF columns
 *
 * @file column.cpp
 * ---------------------------------------------------------------------------**/

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "rmm/rmm.h"
#include "utilities/type_dispatcher.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>

// forward decl -- see validops.cu
gdf_error gdf_mask_concat(gdf_valid_type *output_mask,
                          gdf_size_type output_column_length,            
                          gdf_valid_type *masks_to_concat[], 
                          gdf_size_type *column_lengths, 
                          gdf_size_type num_columns);

/** ---------------------------------------------------------------------------*
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
 * ---------------------------------------------------------------------------**/
gdf_error gdf_column_concat(gdf_column *output_column, gdf_column *columns_to_concat[], int num_columns)
{
  GDF_REQUIRE(num_columns > 0, GDF_INVALID_API_CALL);
  GDF_REQUIRE(output_column != nullptr, GDF_DATASET_EMPTY);
  GDF_REQUIRE(columns_to_concat != nullptr, GDF_DATASET_EMPTY);
  GDF_REQUIRE(columns_to_concat[0] != nullptr, GDF_DATASET_EMPTY);

  const gdf_dtype column_type = columns_to_concat[0]->dtype;

  gdf_size_type total_size{0};

  // Ensure all the columns are properly allocated
  // and have matching types
  for (gdf_size_type i = 0; i < num_columns; ++i) {
    gdf_column *current_column = columns_to_concat[i];

    GDF_REQUIRE(current_column != nullptr, GDF_DATASET_EMPTY);

    if((current_column->size > 0) && (nullptr == current_column->data)){
        return GDF_DATASET_EMPTY;
    }

    GDF_REQUIRE(column_type == current_column->dtype, GDF_DTYPE_MISMATCH);

    total_size += current_column->size;
  }

  bool const at_least_one_mask_present{
      std::any_of(columns_to_concat, columns_to_concat + num_columns,
                  [](gdf_column *col) { return (nullptr != col->valid); })};

  GDF_REQUIRE(column_type == output_column->dtype, GDF_DTYPE_MISMATCH);
  GDF_REQUIRE(output_column->size == total_size, GDF_COLUMN_SIZE_MISMATCH);

  // TODO optimizations if needed
  // 1. Either 
  //    a) use cudaMemcpyAsync to copy the data and overlap copies with gdf_mask_concat
  //       (this will require getting rid of the allocations below because they will
  //       implicitly sync the device), or
  //    b) use a kernel to copy the data from all columns in one go. This will likely not
  //       overlap with the gdf_mask_concat
  // 2. Detect a zero total null count and skip gdf_mask_concat -- use cudaMemsetAsync

  int8_t* target = (int8_t*)(output_column->data);
  output_column->null_count = 0;
  int column_byte_width = 0;
  gdf_error result = get_column_byte_width(output_column, &column_byte_width);
  GDF_REQUIRE(GDF_SUCCESS == result, result);

  // copy data
  for (int i = 0; i < num_columns; ++i) {   
    gdf_size_type bytes = column_byte_width * columns_to_concat[i]->size;
    CUDA_TRY( cudaMemcpy(target, columns_to_concat[i]->data, bytes, cudaMemcpyDeviceToDevice) );
    target += bytes;
    output_column->null_count += columns_to_concat[i]->null_count;
  }
  
  if (at_least_one_mask_present) {
    gdf_valid_type** masks;
    gdf_size_type* column_lengths;
    CUDA_TRY( cudaMallocManaged((void**)&masks, sizeof(gdf_valid_type*)*num_columns) );
    CUDA_TRY( cudaMallocManaged((void**)&column_lengths, sizeof(gdf_size_type)*num_columns) );

    for (int i = 0; i < num_columns; ++i) {   
      masks[i] = columns_to_concat[i]->valid;
      column_lengths[i] = columns_to_concat[i]->size;
    }
  
    result = gdf_mask_concat(output_column->valid, 
                             output_column->size, 
                             masks, 
                             column_lengths, 
                             num_columns);

    CUDA_TRY( cudaFree(masks) );
    CUDA_TRY( cudaFree(column_lengths) );

    return result;
  }
  else if (nullptr != output_column->valid) {
    // no masks, so just fill output valid mask with all 1 bits
    // TODO: async
    CUDA_TRY( cudaMemset(output_column->valid, 
                         0xff, 
                         gdf_get_num_chars_bitmask(total_size) * sizeof(gdf_valid_type)) );
  }
  
  return GDF_SUCCESS;
}

/** ---------------------------------------------------------------------------*
 * @brief Return the size of the gdf_column data type.
 *
 * @return gdf_size_type Size of the gdf_column data type.
 *  ---------------------------------------------------------------------------**/
gdf_size_type gdf_column_sizeof() {
	return sizeof(gdf_column);
}

/** ---------------------------------------------------------------------------*
 * @brief Create a GDF column given data and validity bitmask pointers, size, and
 *        datatype
 *
 * @param[out] column The output column.
 * @param[in] data Pointer to data.
 * @param[in] valid Pointer to validity bitmask for the data.
 * @param[in] size Number of rows in the column.
 * @param[in] dtype Data type of the column.
 * @return gdf_error Returns GDF_SUCCESS upon successful creation.
 * ---------------------------------------------------------------------------**/
gdf_error gdf_column_view(gdf_column *column,
                          void *data,
                          gdf_valid_type *valid,
		                      gdf_size_type size,
                          gdf_dtype dtype)
{
	column->data = data;
	column->valid = valid;
	column->size = size;
	column->dtype = dtype;
	column->null_count = 0;
	return GDF_SUCCESS;
}

/** ---------------------------------------------------------------------------*
 * @brief Create a GDF column given data and validity bitmask pointers, size, and
 *        datatype, and count of null (non-valid) elements
 *
 * @param[out] column The output column.
 * @param[in] data Pointer to data.
 * @param[in] valid Pointer to validity bitmask for the data.
 * @param[in] size Number of rows in the column.
 * @param[in] dtype Data type of the column.
 * @param[in] null_count The number of non-valid elements in the validity bitmask
 * @return gdf_error Returns GDF_SUCCESS upon successful creation.
 * ---------------------------------------------------------------------------**/
gdf_error gdf_column_view_augmented(gdf_column *column,
                                    void *data,
                                    gdf_valid_type *valid,
		                                gdf_size_type size,
                                    gdf_dtype dtype,
                                    gdf_size_type null_count,
                                    gdf_dtype_extra_info extra_info)
{
	column->data = data;
	column->valid = valid;
	column->size = size;
	column->dtype = dtype;
	column->null_count = null_count;
	column->dtype_info = extra_info;
	return GDF_SUCCESS;
}

/** ---------------------------------------------------------------------------*
 * @brief Free the CUDA device memory of a gdf_column
 *
 * @param[inout] column Data and validity bitmask pointers of this column will be freed
 * @return gdf_error GDF_SUCCESS or GDF_ERROR if there is an error freeing the data
 * ---------------------------------------------------------------------------**/
gdf_error gdf_column_free(gdf_column *column) 
{
  RMM_TRY( RMM_FREE(column->data, 0)  );
  RMM_TRY( RMM_FREE(column->valid, 0) );
  return GDF_SUCCESS;
}


namespace{
  struct get_type_size{
    template <typename T>
    auto operator()()
    {
      return sizeof(T);
    }
  };
}
/** ---------------------------------------------------------------------------*
 * @brief Get the byte width of a column
 *
 * @param[in] col The input column
 * @param[out] width The data type size of col
 * @return gdf_error GDF_SUCCESS, or GDF_UNSUPPORTED_DTYPE if col has an invalid
 *         datatype
 * ---------------------------------------------------------------------------**/
gdf_error get_column_byte_width(gdf_column * col, 
                                int * width)
{
  *width = cudf::type_dispatcher(col->dtype, get_type_size{});
	return GDF_SUCCESS;
}
