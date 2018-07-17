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

#include <gdf/gdf.h>
#include <gdf/errorutils.h>

#include "joining.h"

constexpr int HASH_KERNEL_BLOCK_SIZE = 256;
constexpr int HASH_KERNEL_ROWS_PER_THREAD = 1;

// convert to int dtype with the same size
gdf_dtype to_int_dtype(gdf_dtype type)
{
  switch (type) {
    case GDF_INT8:
    case GDF_INT16:
    case GDF_INT32:
    case GDF_INT64:
      return type;
    case GDF_FLOAT32:
      return GDF_INT32;
    case GDF_FLOAT64:
      return GDF_INT64;
    default:
      return GDF_invalid;
  }
}

__device__ __inline__
uint32_t hashed(void *ptr, int int_dtype, int index)
{
  // TODO: add switch to select the right hash class, currently we only support Murmur3 anyways
  switch (int_dtype) {
  case GDF_INT8:  { default_hash<int8_t> hasher; return hasher(((int8_t*)ptr)[index]); }
  case GDF_INT16: { default_hash<int16_t> hasher; return hasher(((int16_t*)ptr)[index]); }
  case GDF_INT32: { default_hash<int32_t> hasher; return hasher(((int32_t*)ptr)[index]); }
  case GDF_INT64: { default_hash<int64_t> hasher; return hasher(((int64_t*)ptr)[index]); }
  default:
    return 0;
  }
}

template<typename size_type>
__device__ __inline__
void hash_combine(size_type &seed, const uint32_t hash_val)
{
  seed ^= hash_val + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// one thread handles multiple rows
// d_col_data[i]: column's data (on device)
// d_col_int_dtype[i]: column's dtype (converted to int) 
__global__ void hash_cols(int num_rows, int num_cols, void **d_col_data, gdf_dtype *d_col_int_dtype, int *d_output)
{
  for (int row = threadIdx.x + blockIdx.x * blockDim.x; row < num_rows; row += blockDim.x * gridDim.x) {
    uint32_t seed = 0;
    for (int col = 0; col < num_cols; col++) {
      uint32_t hash_val = hashed(d_col_data[col], d_col_int_dtype[col], row);
      hash_combine(seed, hash_val);
    }
    d_output[row] = seed;
  }
}

gdf_error gdf_hash(int num_cols, gdf_column **input, gdf_hash_func hash, gdf_column *output)
{
  // check that all columns have the same size
  for (int i = 0; i < num_cols; i++)
    if (i > 0 && input[i]->size != input[i-1]->size) return GDF_COLUMN_SIZE_MISMATCH;
  // check that the output dtype is int32
  // TODO: do we need to support int64 as well?
  if (output->dtype != GDF_INT32) return GDF_UNSUPPORTED_DTYPE;
  int64_t num_rows = input[0]->size;

  // copy data pointers to device
  void **d_col_data, **h_col_data;
  cudaMalloc(&d_col_data, num_cols * sizeof(void*));
  cudaMallocHost(&h_col_data, num_cols * sizeof(void*));
  for (int i = 0; i < num_cols; i++)
    h_col_data[i] = input[i]->data;
  cudaMemcpy(d_col_data, h_col_data, num_cols * sizeof(void*), cudaMemcpyDefault);

  // copy dtype (converted to int) to device
  gdf_dtype *d_col_int_dtype, *h_col_int_dtype;
  cudaMalloc(&d_col_int_dtype, num_cols * sizeof(gdf_dtype));
  cudaMallocHost(&h_col_int_dtype, num_cols * sizeof(gdf_dtype));
  for (int i = 0; i < num_cols; i++)
    h_col_int_dtype[i] = to_int_dtype(input[i]->dtype);
  cudaMemcpy(d_col_int_dtype, h_col_int_dtype, num_cols * sizeof(gdf_dtype), cudaMemcpyDefault);

  // launch a kernel
  const int rows_per_block = HASH_KERNEL_BLOCK_SIZE * HASH_KERNEL_ROWS_PER_THREAD;
  const int64_t grid = (num_rows + rows_per_block-1) / rows_per_block;
  hash_cols<<<grid, HASH_KERNEL_BLOCK_SIZE>>>(num_rows, num_cols, d_col_data, d_col_int_dtype, (int32_t*)output->data);

  // TODO: do we need to synchronize here
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST();

  // free temp memory
  cudaFree(d_col_data);
  cudaFreeHost(h_col_data);
  cudaFree(d_col_int_dtype);
  cudaFreeHost(h_col_int_dtype);

  return GDF_SUCCESS;

}
