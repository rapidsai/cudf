/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#include <cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <stream_compaction.hpp>
#include <bitmask/legacy_bitmask.hpp>
#include <utilities/device_atomics.cuh>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>
#include <utilities/wrapper_types.hpp>
#include <cub/cub.cuh>


namespace {

__device__ __forceinline__
bool is_valid(uint32_t const * __restrict__ bitmask, gdf_index_type i) {
  return (bitmask[i / 32] >> (i %32)) & 1;
}

struct nonnull_and_true {
  nonnull_and_true(gdf_column const boolean_mask)
      : data{static_cast<cudf::bool8*>(boolean_mask.data)},
        bitmask{reinterpret_cast<uint32_t const *>(boolean_mask.valid)} {
    CUDF_EXPECTS(boolean_mask.dtype == GDF_BOOL8, "Expected boolean column");
    CUDF_EXPECTS(boolean_mask.data != nullptr, "Null boolean_mask data");
    CUDF_EXPECTS(boolean_mask.valid != nullptr, "Null boolean_mask bitmask");
  }

  __device__ bool operator()(gdf_index_type i) {
    bool valid = is_valid(bitmask, i);//(bitmask[i / 32] >> (i %32)) & 1;
    return ((cudf::true_v == data[i]) && valid);
  }

 private:
  cudf::bool8 const * __restrict__ const data;
  uint32_t const * __restrict__ const bitmask;
};
}  // namespace

namespace cudf {

template <int block_size, int per_thread, typename MaskFunc>
__global__ void compute_block_counts(gdf_size_type  * __restrict__ block_counts,
                                     gdf_size_type                 mask_size,
                                     MaskFunc                       mask)
{
  int tid = threadIdx.x + per_thread * block_size * blockIdx.x;

  int count = 0;
  
  for (int i = 0; i < per_thread; i++) {
    bool passes = (tid < mask_size) && mask(tid);
    
    count += __syncthreads_count(passes);
    tid += block_size;
  }
  
  if (threadIdx.x == 0) {
    block_counts[blockIdx.x] = count;
  }
}

template <int block_size>
__device__ gdf_index_type block_scan_mask(bool mask_true, 
                                          gdf_index_type &block_sum)
{
  int offset = 0;

  using BlockScan = cub::BlockScan<gdf_size_type, block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  BlockScan(temp_storage).ExclusiveSum(mask_true, offset, block_sum);
  
  return offset;
}

static constexpr int warp_size = 32;

template <typename T, int block_size, int per_thread, typename MaskFunc>
__global__ void scatter_no_valid(//gdf_column *output_column,
                                 //gdf_column const * input_column,
                                 T* __restrict__ output_data,
                                 gdf_valid_type * __restrict__ output_valid,
                                 gdf_size_type * output_null_count,
                                 T const * __restrict__ input_data,
                                 gdf_valid_type const * input_valid,
                                 gdf_size_type  * __restrict__ block_offsets,
                                 gdf_size_type mask_size,
                                 gdf_size_type num_columns,
                                 MaskFunc mask)
{
  int tid = threadIdx.x + per_thread * block_size * blockIdx.x;
  gdf_size_type block_offset = block_offsets[blockIdx.x];
  
  for (int i = 0; i < per_thread; i++) {

    bool mask_true = (tid < mask_size) && mask(tid);

    // get output location
    gdf_index_type block_sum = 0;
    const gdf_index_type local_index = block_scan_mask<block_size>(mask_true,
                                                                   block_sum);

    if (mask_true) {

      const gdf_index_type in_index = tid;
      const gdf_index_type out_index = local_index + block_offset;

      //for (int c = 0; c < num_columns; c++) {

      //T * __restrict__ out_ptr = static_cast<T*>(output_column->data);
      //T const * __restrict__ in_ptr  = static_cast<T const*>(input_column->data);

      //out_ptr[out_index] = in_ptr[in_index];
      output_data[out_index] = input_data[in_index];
    }
    block_offset += block_sum;
    tid += block_size;
  }
}

template <typename T, int block_size, int per_thread, typename MaskFunc>
__global__ void scatter_with_valid(//gdf_column *output_column,
                                   //gdf_column const * input_column,
                                   T* __restrict__ output_data,
                                   gdf_valid_type * __restrict__ output_valid,
                                   gdf_size_type * output_null_count,
                                   T const * __restrict__ input_data,
                                   gdf_valid_type const * input_valid,
                                   gdf_size_type  * __restrict__ block_offsets,
                                   gdf_size_type mask_size,
                                   gdf_size_type num_columns,
                                   MaskFunc mask)
{
  int tid = threadIdx.x + per_thread * block_size * blockIdx.x;
  gdf_size_type block_offset = block_offsets[blockIdx.x];
  
  // one extra warp worth in case the block is not aligned
  __shared__ int32_t temp_valids[block_size+warp_size];
  __shared__ T       temp_data[block_size];
  
  for (int i = 0; i < per_thread; i++) {

    bool mask_true = (tid < mask_size) && mask(tid);

    // get output location
    gdf_index_type block_sum = 0;
    const gdf_index_type local_index = block_scan_mask<block_size>(mask_true,
                                                                   block_sum);

    // determine if this warp's output offset is aligned to a warp size
    const gdf_size_type block_offset_aligned =
      warp_size * (block_offset / warp_size);
    const gdf_size_type aligned_offset = block_offset - block_offset_aligned;

    // zero the shared memory
    temp_valids[threadIdx.x] = 0;
    if (threadIdx.x < warp_size) temp_valids[block_size + threadIdx.x] = 0;

    __syncthreads();

    if (mask_true) {
      temp_data[local_index] = input_data[tid];

      // scatter validity mask to shared memory
      uint32_t const * __restrict__ valid_ptr =
        reinterpret_cast<uint32_t const*>(input_valid);

      bool valid = is_valid(valid_ptr, tid);

      temp_valids[local_index + aligned_offset] = valid;
    }

    // note maximum block size is limited to 1024 by this
    __shared__ uint32_t warp_valid_counts[warp_size];
    if (threadIdx.x < warp_size) warp_valid_counts[threadIdx.x] = 0;

    __syncthreads(); // wait for shared data and validity mask to be complete

    if (threadIdx.x < block_sum)
      output_data[block_offset + threadIdx.x] = temp_data[threadIdx.x];

    constexpr int num_warps = block_size / warp_size;
    const int last_warp = block_sum / warp_size;
    const int wid = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;

    if (block_sum > 0 && wid <= last_warp) {
      int valid_index = (block_offset / warp_size) + wid;

      // Cast the validity type to a type where atomicOr is natively supported
      uint32_t* const __restrict__ valid_ptr =
        reinterpret_cast<uint32_t*>(output_valid);

      // compute the valid mask for this warp
      int32_t valid_warp =
        __ballot_sync(0xffffffff, temp_valids[threadIdx.x]);

      if (lane == 0 && valid_warp != 0) {
        warp_valid_counts[wid] = __popc(valid_warp);
        if (wid > 0 && wid < last_warp)
          valid_ptr[valid_index] = valid_warp;
        else 
          atomicOr(&valid_ptr[valid_index], valid_warp);
      }

      // if the block is full and not aligned then we have one more warp to cover
      if (wid == 0) {
        int32_t valid_warp =
          __ballot_sync(0xffffffff, temp_valids[block_size + threadIdx.x]);
        //if (lane == 0) printf("X bid: %d, wid: %d, al_off: %d, valid_index: %d, valid_warp: %x\n", blockIdx.x, wid, aligned_offset, valid_index+num_warps, valid_warp);
        if (lane == 0 && valid_warp != 0) {
          warp_valid_counts[wid] += __popc(valid_warp);
          atomicOr(&valid_ptr[valid_index + num_warps], valid_warp);
        }
      }
    }

    __syncthreads();

    if (threadIdx.x < warp_size) {
      uint32_t my_valid_count = warp_valid_counts[threadIdx.x];

      __shared__ typename cub::WarpReduce<uint32_t>::TempStorage temp_storage;
      
      uint32_t block_valid_count =
        cub::WarpReduce<uint32_t>(temp_storage).Sum(my_valid_count);
      
        if (lane == 0) {
        atomicAdd(output_null_count, block_sum - block_valid_count);
      }
    }

    block_offset += block_sum;
    tid += block_size;
  }
}

template <int block_size, int per_thread>
struct scatter_functor 
{
  template <typename T, typename MaskFunc>
  void operator()(gdf_column *output_column,
                  gdf_column const * input_column,
                  gdf_size_type  *block_offsets,
                  gdf_size_type mask_size,
                  gdf_size_type num_columns,
                  bool has_valid,
                  MaskFunc mask) {
    constexpr int per_block = block_size * per_thread; 
    int num_blocks = (mask_size + per_block - 1) / per_block;

    auto scatter_kernel = (has_valid) ? 
      scatter_with_valid<T, block_size, per_thread, MaskFunc> :
      scatter_no_valid<T, block_size, per_thread, MaskFunc>;

    gdf_size_type *null_count = nullptr;
    if (has_valid) {
      RMM_ALLOC(&null_count, sizeof(gdf_size_type), 0);
      CUDA_TRY(cudaMemset(null_count, 0, sizeof(gdf_size_type)));
    }

    scatter_kernel<<<num_blocks, block_size>>>(static_cast<T*>(output_column->data),
                                               output_column->valid,
                                               null_count,
                                               static_cast<T const*>(input_column->data),
                                               input_column->valid,
                                               block_offsets,
                                               mask_size,
                                               num_columns,
                                               mask);

    if (has_valid) {
      CUDA_TRY(cudaMemcpy(&output_column->null_count, null_count, sizeof(gdf_size_type), cudaMemcpyDefault));
      RMM_FREE(null_count, 0);
    }
  }
};

gdf_size_type get_output_size(gdf_size_type *block_counts,
                              gdf_size_type *block_offsets,
                              gdf_size_type num_blocks)
{
  gdf_size_type last_block_count = 0;
  cudaMemcpy(&last_block_count, &block_counts[num_blocks - 1], 
             sizeof(gdf_size_type), cudaMemcpyDefault);
  gdf_size_type last_block_offset = 0;
  if (num_blocks > 1)
    cudaMemcpy(&last_block_offset, &block_offsets[num_blocks - 1], 
               sizeof(gdf_size_type), cudaMemcpyDefault);
  return last_block_count + last_block_offset;
}

/**
 * @brief Filters a column using a column of boolean values as a mask.
 *
 */
gdf_column apply_boolean_mask(gdf_column const *input,
                              gdf_column const *boolean_mask) {
  CUDF_EXPECTS(nullptr != input, "Null input");
  CUDF_EXPECTS(nullptr != boolean_mask, "Null boolean_mask");
  CUDF_EXPECTS(input->size == boolean_mask->size, "Column size mismatch");
  CUDF_EXPECTS(boolean_mask->dtype == GDF_BOOL8, "Mask must be Boolean type");

  // High Level Algorithm:
  // First, compute a `scatter_map` from the boolean_mask that will scatter
  // input[i] if boolean_mask[i] is non-null and "true". This is simply an 
  // exclusive scan of nonnull_and_true
  // Second, use the `scatter_map` to scatter elements from the `input` column
  // into the `output` column

  gdf_size_type h_output_size = 0;

  constexpr int block_size = 1024;
  constexpr int per_thread = 32;
  constexpr int per_block = block_size * per_thread;
  int num_blocks = (boolean_mask->size + per_block - 1) / per_block;

  gdf_size_type *block_counts = nullptr;
  RMM_ALLOC(&block_counts, num_blocks * sizeof(gdf_size_type), 0);
  
  compute_block_counts<block_size, per_thread>
    <<<num_blocks, block_size>>>(block_counts, boolean_mask->size,
                                 nonnull_and_true{*boolean_mask});

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  gdf_size_type *block_offsets = nullptr;
  RMM_ALLOC(&block_offsets, num_blocks * sizeof(gdf_size_type), 0);
  
  if (num_blocks > 1) {
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  block_counts,
                                  block_offsets,
                                  num_blocks);
    // Allocate temporary storage
    RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  block_counts,
                                  block_offsets,
                                  num_blocks);
  }
  else {
    cudaMemset(block_offsets, 0, num_blocks * sizeof(gdf_size_type));
  }

  h_output_size = get_output_size(block_counts, block_offsets, num_blocks);

  gdf_column output;
  gdf_column_view(&output, 0, 0, 0, input->dtype);
  output.dtype_info = input->dtype_info;

  if (h_output_size > 0) {    
    // Allocate/initialize output column
    gdf_size_type column_byte_width{gdf_dtype_size(input->dtype)};

    void *data = nullptr;
    gdf_valid_type *valid = nullptr;
    RMM_ALLOC(&data, h_output_size * column_byte_width, 0);
    
    if (input->valid != nullptr) {
      gdf_size_type bytes = gdf_valid_allocation_size(h_output_size);
      RMM_ALLOC(&valid, bytes, 0);
      CUDA_TRY(cudaMemset(valid, 0, bytes));
    }
    
    CUDF_EXPECTS(GDF_SUCCESS == gdf_column_view(&output, data, valid,
                                                h_output_size, input->dtype),
                "cudf::apply_boolean_mask failed to create output column view");
    //gdf_column *d_input = nullptr, *d_output = nullptr;
    //RMM_ALLOC(&d_input, sizeof(gdf_column), 0);
    //RMM_ALLOC(&d_output, sizeof(gdf_column), 0);
    //cudaMemcpy(d_input, input, sizeof(gdf_column), cudaMemcpyDefault);
    //cudaMemcpy(d_output, &output, sizeof(gdf_column), cudaMemcpyDefault);
    
    //gdf_column* outputs[1] = {&output};

    cudf::type_dispatcher(output.dtype, 
                          scatter_functor<block_size, per_thread>{},
                          &output, 
                          input, 
                          block_offsets,
                          boolean_mask->size,
                          gdf_size_type{1},
                          input->valid != nullptr,
                          nonnull_and_true{*boolean_mask});

    CHECK_STREAM(0);

    //cudaMemcpy(&output, d_output, sizeof(gdf_column), cudaMemcpyDefault);
    //RMM_FREE(d_input, 0);
    //RMM_FREE(d_output, 0);
  }
  //RMM_FREE(d_output_size, 0);
  return output;
}

}  // namespace cudf
