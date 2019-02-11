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

/**
 * @file cudf_dlpack.cu  Code to convert between gdf_column and DLTensor
 */

#include <limits>

#include "cudf.h"
#include "utilities/error_utils.h"
#include "dlpack/dlpack.h"
#include "rmm/rmm.h"

/** ---------------------------------------------------------------------------*
 * @brief Convert a DLPack DLDataType struct to a gdf_dtype enum value
 *
 * @param[in] type The DLDataType struct
 * @return A valid gdf_dtype if the data type is supported, or GDF_invalid if not.
 * ---------------------------------------------------------------------------**/
gdf_dtype DLDataType_to_gdf_dtype(DLDataType type)
{
  if (type.lanes > 1)
    return GDF_invalid; // vector types not currently supported

  switch (type.bits) {
    case 8:  return (type.code == kDLInt) ?   GDF_INT8  : GDF_invalid;
    case 16: return (type.code == kDLInt) ?   GDF_INT16 : GDF_invalid;
    case 32: return (type.code == kDLInt) ?   GDF_INT32  : 
                    (type.code == kDLFloat) ? GDF_FLOAT32 : GDF_invalid;
    case 64: return (type.code == kDLInt) ?   GDF_INT64 : 
                    (type.code == kDLFloat) ? GDF_FLOAT64 : GDF_invalid;
    default: break;
  }

  return GDF_invalid;
}

// Convert a DLPack DLTensor into gdf_column(s)
gdf_error gdf_from_dlpack(gdf_column** columns,
                          int *num_columns,
                          DLManagedTensor const * tensor)
{
  // Make sure this Tensor uses CUDA memory
  GDF_REQUIRE(tensor->dl_tensor.ctx.device_type == kDLGPU, GDF_INVALID_API_CALL);
  // Make sure the current device ID matches the Tensor's device ID
  int device_id = 0;
  CUDA_TRY(cudaGetDevice(&device_id));
  GDF_REQUIRE(tensor->dl_tensor.ctx.device_id == device_id, GDF_INVALID_API_CALL);

  // Currently only 1D tensors are supported
  GDF_REQUIRE(tensor->dl_tensor.ndim == 1, GDF_NOTIMPLEMENTED_ERROR);

  // Ensure the column is not too big
  GDF_REQUIRE(tensor->dl_tensor.shape[0] > 0, GDF_DATASET_EMPTY);
  GDF_REQUIRE(tensor->dl_tensor.shape[0] < std::numeric_limits<gdf_size_type>::max(), 
              GDF_COLUMN_SIZE_TOO_BIG);
  
  // compute the GDF datatype
  gdf_dtype dtype = DLDataType_to_gdf_dtype(tensor->dl_tensor.dtype); 
  GDF_REQUIRE(dtype != GDF_invalid, GDF_UNSUPPORTED_DTYPE);
  
  // Note: since cuDF has no way to register a deleter for a column, we
  // have to make a copy of the data. In the future if we add the ability 
  // to register deleters, we could register a wrapper around the 
  // DLManagedTensor's deleter and just use the memory directly without copying
  // (Note: this is only true for 1D tensors and column-major 2D tensors. Other
  // layouts will require copying anyway.)

  // compute the size and allocate data
  *num_columns = tensor->dl_tensor.ndim;
  gdf_size_type byte_width = gdf_dtype_size(dtype);
  gdf_size_type length = tensor->dl_tensor.shape[0];
  size_t bytes = length * byte_width;
  void* col_data = 0; 
  RMM_TRY(RMM_ALLOC(&col_data, bytes, 0));

  // copy the dl_tensor data
  void *tensor_data = reinterpret_cast<void*>(
    reinterpret_cast<uintptr_t>(tensor->dl_tensor.data) + 
    tensor->dl_tensor.byte_offset);
  CUDA_TRY(cudaMemcpy(col_data, tensor_data, bytes, cudaMemcpyDefault));

  // Call the managed tensor's deleter since our "borrowing" is done
  tensor->deleter(const_cast<DLManagedTensor*>(tensor));
  
  // construct column view
  *columns = new gdf_column[*num_columns];
  GDF_REQUIRE(*columns != nullptr, GDF_MEMORYMANAGER_ERROR);
  return gdf_column_view(columns[0], col_data, nullptr, length, dtype);
}

// Convert an array of gdf_column(s) into a DLPack DLTensor
gdf_error gdf_to_dlpack(DLManagedTensor *tensor,
                        gdf_column const * columns[], 
                        int num_columns)
{
  return GDF_SUCCESS;
}
	