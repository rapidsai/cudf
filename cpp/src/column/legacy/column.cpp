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

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <rmm/rmm.h>
#include <utilities/column_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>

// forward decl -- see validops.cu
gdf_error gdf_mask_concat(gdf_valid_type *output_mask,
                          gdf_size_type output_column_length,            
                          gdf_valid_type *masks_to_concat[], 
                          gdf_size_type *column_lengths, 
                          gdf_size_type num_columns);

// Concatenates multiple gdf_columns into a single, contiguous column,
// including the validity bitmasks.
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
  std::size_t column_byte_width = cudf::byte_width(*output_column);

  // copy data

  if(columns_to_concat[0]->dtype == GDF_STRING_CATEGORY){
    concat_categories(columns_to_concat,output_column,num_columns);
    for (int i = 0; i < num_columns; ++i) {
      output_column->null_count += columns_to_concat[i]->null_count;
    }
  }else{
    for (int i = 0; i < num_columns; ++i) {
      std::size_t bytes = column_byte_width * columns_to_concat[i]->size;
      CUDA_TRY( cudaMemcpy(target, columns_to_concat[i]->data, bytes, cudaMemcpyDeviceToDevice) );
      target += bytes;
      output_column->null_count += columns_to_concat[i]->null_count;
    }

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

    gdf_error result = gdf_mask_concat(output_column->valid,
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
                         gdf_num_bitmask_elements(total_size) * sizeof(gdf_valid_type)) );
  }

  return GDF_SUCCESS;
}

// Return the size of the gdf_column data type.
gdf_size_type gdf_column_sizeof() {
  return sizeof(gdf_column);
}

// Constructor for the gdf_context struct
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
  column->col_name = nullptr;
  column->dtype_info.category = nullptr;
  column->dtype_info.time_unit = TIME_UNIT_NONE;
  return GDF_SUCCESS;
}


// Create a GDF column given data and validity bitmask pointers, size, and
//        datatype, and count of null (non-valid) elements
gdf_error gdf_column_view_augmented(gdf_column *column,
                                    void *data,
                                    gdf_valid_type *valid,
                                    gdf_size_type size,
                                    gdf_dtype dtype,
                                    gdf_size_type null_count,
                                    gdf_dtype_extra_info extra_info,
                                    const char* name)
{
  gdf_column_view(column, data, valid, size, dtype);
  column->null_count = null_count;
  column->dtype_info = extra_info;
  if (name != nullptr) {
    size_t len = strlen(name);
    if (len > 0) {
      column->col_name = (char *)malloc(strlen(name) + 1);
      std::strcpy(column->col_name, name);
    }
  }
  return GDF_SUCCESS;
}

// Free the CUDA device memory of a gdf_column
gdf_error gdf_column_free(gdf_column *column) 
{
  if (column->dtype == GDF_STRING){
    if (column->data) {
      NVStrings::destroy(static_cast<NVStrings*>(column->data));
    }
    RMM_TRY( RMM_FREE(column->valid, 0) );
  } else if (column->dtype == GDF_STRING_CATEGORY) {
    if (column->dtype_info.category) {
      NVCategory::destroy(static_cast<NVCategory*>(column->dtype_info.category));
    }
    RMM_TRY( RMM_FREE(column->data, 0)  );
    RMM_TRY( RMM_FREE(column->valid, 0) );  
  } else {
    RMM_TRY( RMM_FREE(column->data, 0)  );
    RMM_TRY( RMM_FREE(column->valid, 0) );
  }
  return GDF_SUCCESS;
}

namespace cudf {

namespace detail {

void allocate_column_fields(gdf_column& column,
                            bool allocate_mask,
                            cudaStream_t stream)
{
  if (column.size > 0) {
    const auto byte_width = (column.dtype == GDF_STRING)
                          ? sizeof(std::pair<const char *, size_t>)
                          : cudf::size_of(column.dtype);
    RMM_TRY(RMM_ALLOC(&column.data, column.size * byte_width, stream));
    if (allocate_mask) {
      size_t valid_size = gdf_valid_allocation_size(column.size);
      RMM_TRY(RMM_ALLOC(&column.valid, valid_size, stream));
    }
  }
}

} // namespace detail


/*
 * Allocates a new column of the given size and type.
 */
gdf_column allocate_column(gdf_dtype dtype, gdf_size_type size,
                           bool allocate_mask,
                           gdf_dtype_extra_info info,
                           cudaStream_t stream)
{  
  gdf_column output{};
  output.size = size;
  output.dtype = dtype;
  output.dtype_info = info;

  detail::allocate_column_fields(output, allocate_mask, stream);

  return output;
}

} // namespace cudf

namespace{
  struct get_type_size{
    template <typename T>
    auto operator()()
    {
      return sizeof(T);
    }
  };
}

