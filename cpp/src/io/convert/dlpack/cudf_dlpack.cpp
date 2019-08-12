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

#include <cudf/cudf.h>
#include <utilities/column_utils.hpp>
#include <utilities/error_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <dlpack/dlpack.h>
#include <rmm/rmm.h>

namespace {
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

  /** ---------------------------------------------------------------------------*
   * @brief Convert a gdf_dtype to a DLPack DLDataType struct
   *
   * This struct must be used with cudf::type_dispatcher like this:
   * tensor.dtype = cudf::type_dispatcher(gdf_type, gdf_dtype_to_DLDataType);
   * ---------------------------------------------------------------------------**/
  struct gdf_dtype_to_DLDataType {
    template <typename T>
    DLDataType operator()(){
      DLDataType type;
      if (std::is_integral<T>::value) {
        if (std::is_signed<T>::value) type.code = kDLInt;
        else                          type.code = kDLUInt;
      }
      else if (std::is_floating_point<T>::value) type.code = kDLFloat;
      // Unfortunately DLPack type codes don't have an error code, so use 0xFF
      else type.code = 0xFF;

      type.bits = sizeof(T) * 8;
      type.lanes = 1;
      return type;
    }
  };

  static inline size_t tensor_size(const DLTensor& t)
  {
    size_t size = 1;
    for (int i = 0; i < t.ndim; ++i) size *= t.shape[i];
    size *= (t.dtype.bits * t.dtype.lanes + 7) / 8;
    return size;
  }
}


// Convert a DLPack DLTensor into gdf_column(s)
// Currently 1D and 2D column-major (Fortran order) tensors are supported
gdf_error gdf_from_dlpack(gdf_column** columns,
                          gdf_size_type *num_columns,
                          DLManagedTensor const * tensor)
{
  // We can copy from host or device pointers
  GDF_REQUIRE(kDLGPU == tensor->dl_tensor.ctx.device_type ||
              kDLCPU == tensor->dl_tensor.ctx.device_type ||
              kDLCPUPinned == tensor->dl_tensor.ctx.device_type, 
              GDF_INVALID_API_CALL);
  // Make sure the current device ID matches the Tensor's device ID
  int device_id = 0;
  CUDA_TRY(cudaGetDevice(&device_id));
  GDF_REQUIRE(tensor->dl_tensor.ctx.device_id == device_id, 
              GDF_INVALID_API_CALL);

  // Currently only 1D and 2D tensors are supported
  GDF_REQUIRE(tensor->dl_tensor.ndim > 0, GDF_DATASET_EMPTY);
  GDF_REQUIRE(tensor->dl_tensor.ndim <= 2, GDF_NOTIMPLEMENTED_ERROR);

  // Ensure the column is not too big
  GDF_REQUIRE(tensor->dl_tensor.shape[0] > 0, GDF_DATASET_EMPTY);
  if (tensor->dl_tensor.ndim > 1)
    GDF_REQUIRE(tensor->dl_tensor.shape[1] > 0, GDF_DATASET_EMPTY);

  GDF_REQUIRE(tensor->dl_tensor.shape[0] < 
              std::numeric_limits<gdf_size_type>::max(), 
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
  // Note: assumes Fortran (column-major) data layout
  *num_columns = 1;
  if (tensor->dl_tensor.ndim == 2) *num_columns = tensor->dl_tensor.shape[1];
  *columns = new gdf_column[*num_columns]{};
  GDF_REQUIRE(*columns != nullptr, GDF_MEMORYMANAGER_ERROR);

  gdf_size_type byte_width = cudf::size_of(dtype);
  gdf_size_type num_rows = tensor->dl_tensor.shape[0];
  size_t bytes = num_rows * byte_width;

  // Determine the stride between the start of each column
  // For 1D tensors, stride is zero. For 2D tensors, if the strides pointer is
  // not null, then get the stride from the tensor struct. Otherwise, 
  // we assume the stride is just the size of the columns of the tensor in 
  // bytes.
  int64_t col_stride = 0;
  if (*num_columns > 1) {
    col_stride = byte_width * num_rows;
    if (nullptr != tensor->dl_tensor.strides)
      col_stride = byte_width * tensor->dl_tensor.strides[1];
  }
  
  // copy the dl_tensor data
  for (gdf_size_type c = 0; c < *num_columns; ++c) {
    void *tensor_data = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(tensor->dl_tensor.data) +
      tensor->dl_tensor.byte_offset +
      col_stride * c);

    void* col_data = 0;
    RMM_TRY(RMM_ALLOC(&col_data, bytes, 0));

    CUDA_TRY(cudaMemcpy(col_data, tensor_data, bytes, cudaMemcpyDefault));

    // construct column view
    gdf_error status = gdf_column_view(&(*columns)[c], col_data,
                                       nullptr, num_rows, dtype);
    GDF_REQUIRE(GDF_SUCCESS == status, status);
  }

  // Note: we do NOT call the input tensor's deleter currently because 
  // the caller of this function may still need it. Also, it is often 
  // packaged in a PyCapsule on the Python side, which (in the case of Cupy)
  // may be set up to call the deleter in its own destructor
  //tensor->deleter(const_cast<DLManagedTensor*>(tensor));

  return GDF_SUCCESS;
}

// Convert an array of gdf_column(s) into a DLPack DLTensor
// Supports 1D and 2D tensors (single or multiple columns)
gdf_error gdf_to_dlpack(DLManagedTensor *tensor,
                        gdf_column const * const * columns, 
                        gdf_size_type num_columns)
{
  GDF_REQUIRE(tensor != nullptr, GDF_DATASET_EMPTY);
  GDF_REQUIRE(columns && num_columns > 0, GDF_DATASET_EMPTY);

  // first column determines datatype and number of rows
  gdf_dtype type = columns[0]->dtype;
  gdf_size_type num_rows = columns[0]->size;

  GDF_REQUIRE(type != GDF_invalid, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(num_rows > 0, GDF_DATASET_EMPTY);

  // ensure all columns are the same type and size
  for (gdf_size_type i = 1; i < num_columns; ++i) {
    GDF_REQUIRE(columns[i]->dtype == type, GDF_DTYPE_MISMATCH);
    GDF_REQUIRE(columns[i]->size == num_rows, GDF_COLUMN_SIZE_MISMATCH);
  }

  tensor->dl_tensor.ndim = (num_columns > 1) ? 2 : 1;
  
  tensor->dl_tensor.dtype = 
    cudf::type_dispatcher(type, gdf_dtype_to_DLDataType() );
  GDF_REQUIRE(tensor->dl_tensor.dtype.code != 0xFF, GDF_UNSUPPORTED_DTYPE);
    
  tensor->dl_tensor.shape = new int64_t[tensor->dl_tensor.ndim];
  tensor->dl_tensor.shape[0] = num_rows;
  if (tensor->dl_tensor.ndim > 1) {
    tensor->dl_tensor.shape[1] = num_columns;
    tensor->dl_tensor.strides = new int64_t[2] {1, num_rows};
  }
  else {
    tensor->dl_tensor.strides = new int64_t[1] {1};
  }
  // tensor->dl_tensor.strides = nullptr;
  tensor->dl_tensor.byte_offset = 0;
  
  CUDA_TRY( cudaGetDevice(&tensor->dl_tensor.ctx.device_id) );
  tensor->dl_tensor.ctx.device_type = kDLGPU;

  // If there is only one column, then a 1D tensor can just copy the pointer
  // to the data in the column, and the deleter should not delete the original
  // data. However, this is inconsistent with the 2D cases where we must do a
  // copy of each column's data into the dense tensor array. Also, if we don't
  // copy, then the original column data could be changed, which would change
  // the contents of the tensor, which might be surprising or cause issues.
  // Therefore, for now we ALWAYS do a copy of the data. If this becomes
  // a performance issue we can reevaluate in the future.

  char *data = nullptr;
  const size_t N = num_rows * num_columns;
  size_t bytesize = tensor_size(tensor->dl_tensor);
  size_t column_bytesize = num_rows * (tensor->dl_tensor.dtype.bits / 8);

  RMM_TRY( RMM_ALLOC(&data, bytesize, 0) );

  char *d = data;
  for (gdf_size_type i = 0; i < num_columns; ++i) {
    CUDA_TRY(cudaMemcpy(d, columns[i]->data,
                        column_bytesize, cudaMemcpyDefault));
    d += column_bytesize;
  }

  tensor->dl_tensor.data = data;

  tensor->deleter = [](DLManagedTensor * arg)
  {
    // TODO switch assert to RMM_TRY once RMM supports throwing exceptions
    if (arg->dl_tensor.ctx.device_type == kDLGPU)
      RMM_TRY(RMM_FREE(arg->dl_tensor.data, 0));
    delete [] arg->dl_tensor.shape;
    delete [] arg->dl_tensor.strides;
    delete arg;
  };

  tensor->manager_ctx = nullptr;

  return GDF_SUCCESS;
}
