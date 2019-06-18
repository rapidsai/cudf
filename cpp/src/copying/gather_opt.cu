/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "gather.hpp"
#include <cudf/copying.hpp>
#include <cudf/cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/cudf_utils.h>
#include <utilities/type_dispatcher.hpp>
#include <bitmask/legacy_bitmask.hpp>
#include <cudf/table.hpp>
#include <string/nvcategory_util.hpp>

#include <algorithm>
#include <thrust/gather.h>

#include <table/device_table.cuh>

#include <cudf/table.hpp>

#include <utilities/column_utils.hpp>

/**
 * @brief Operations for copying from one column to another
 * @file copying_ops.cu
 */

namespace impl {

template<class T>
__device__ inline void warp_wise_reduce(T& f) {
  #pragma unroll
  for(int offset = 16; offset > 0; offset /= 2){
    // TODO: ONLY works for CUDA 9.2 or later
    float other_f = __shfl_down_sync(0xffffffffu, f, offset);
    f += other_f;
  }
}

template<class T>
__device__ inline void block_wise_reduce(T f, T* smem){
  __syncthreads();
  int lane_id = (threadIdx.x & 31);
  int warp_id = (threadIdx.x >> 5);
  // Assuming lane 0 of each warp holds the value that we want to perform reduction
  if(lane_id == 0){
    smem[warp_id] = f;
  }
  __syncthreads();
  if(warp_id == 0){ // Here I am assuming maximum block size is 1024 and 1024 >> 5 = 32
    f = (lane_id < (blockDim.x>>5)) ? smem[lane_id] : 0;
    warp_wise_reduce(f); 
    if(lane_id == 0){
      smem[0] = f;
    }
  }
  __syncthreads();
}

template<bool check_bounds>
__device__ __inline__ void gather_bitmask_device(
    gdf_valid_type const* const __restrict__ source_mask,
    gdf_size_type const source_row,
    gdf_size_type const num_source_rows, 
    gdf_valid_type* const destination_mask,
    gdf_size_type const destination_row, 
    gdf_size_type const num_destination_rows,
    gdf_size_type* smem
  ){
  using MaskType = uint32_t;
  constexpr uint32_t BITS_PER_MASK = sizeof(MaskType) * 8;

  // Cast bitmask to a type to a 4B type
  // TODO: Update to use new bit_mask_t
  MaskType* const __restrict__ destination_mask32 =
      reinterpret_cast<MaskType*>(destination_mask);

  auto active_threads =
      __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  bool source_bit_is_valid;
  if(check_bounds){
    source_bit_is_valid = source_row < num_source_rows && gdf_is_valid(source_mask, source_row);
  }else{
    source_bit_is_valid = gdf_is_valid(source_mask, source_row);
  }
  
  // Use ballot to find all valid bits in this warp and create the output
  // bitmask element
  MaskType const result_mask = __ballot_sync(active_threads, source_bit_is_valid);

  gdf_index_type const output_element = destination_row >> 5; // destination_row / 32

  // Only one thread writes output
  if (0 == threadIdx.x & 31) { // __idx % 32
    destination_mask32[output_element] = result_mask;
  }
  block_wise_reduce(__popc(result_mask), smem);
}

template<bool check_bounds>
__global__ void gather_kernel(const device_table source, 
                                  const gdf_size_type gather_map[], 
                                  device_table destination,
                                  gdf_size_type* d_count)
{
/**
  The following optimizations should be done to improve the performance of gather/scatter:

 - Gather/Scatter entire rows of n columns in a single kernel
 - In the same kernel, scatter/gather the bitmasks
 - In the same kernel, compute the null count for each output column
*/
  static __shared__ gdf_size_type smem_count[32];
  const gdf_index_type n_source_rows = source.num_rows();
  const gdf_index_type n_destination_rows = destination.num_rows();
  
  // Each element of gather_map[] only needs to be used once. 
  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;
  while(destination_row < n_destination_rows){
    gdf_index_type source_row = gather_map[destination_row];
    // gather the entire row
    if(!check_bounds || source_row < n_source_rows){
      copy_row(destination, destination_row, source, source_row);
    }
    for(int i = 0; i < destination.num_columns(); i++){
      if(destination.get_column(i)->valid){
        gather_bitmask_device<check_bounds>(source.get_column(i)->valid, source_row, n_source_rows,
          destination.get_column(i)->valid, destination_row, n_destination_rows, smem_count);
        if(threadIdx.x == 0){
          atomicAdd(d_count+i, smem_count[0]);
        }
      }
    }
    destination_row += blockDim.x * gridDim.x;
  } // while

}

}  // namespace

namespace cudf {
namespace opt   {
namespace detail {

void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table, bool check_bounds, cudaStream_t stream) {
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");

  // If the destination is empty, return immediately as there is nothing to
  // gather
  if (0 == destination_table->num_rows()) {
    return;
  }

  CUDF_EXPECTS(nullptr != gather_map, "gather_map is null");
  CUDF_EXPECTS(source_table->num_columns() == destination_table->num_columns(),
               "Mismatched number of columns");
  const gdf_size_type n_cols = source_table->num_columns();
  
  // In place check
  std::vector<gdf_column> v_cpy(n_cols);
  std::vector<gdf_column*> vp_cpy(n_cols);
  for(gdf_size_type c = 0; c < n_cols; c++){
    v_cpy[c] = *(destination_table->get_column(c)); // Actually create a new gdf_column object.
    vp_cpy[c] = &v_cpy[c];
  }
  table temp_output(vp_cpy);
  std::vector<bool> v_data_in_place(n_cols, false);
  std::vector<bool> v_valid_in_place(n_cols, false);
  for(gdf_size_type i = 0; i < n_cols; i++){
    
    gdf_column* tmp_col = temp_output.get_column(i);
    gdf_column* dest_col = destination_table->get_column(i);
    const gdf_column* src_col = source_table->get_column(i);
 
    CUDF_EXPECTS(src_col->dtype == dest_col->dtype, "Column type mismatch");
    // If the source column has a valid buffer, the destination column must
    // also have one
    bool const source_has_nulls {src_col->valid != nullptr};
    bool const dest_has_nulls {dest_col->valid != nullptr};
    CUDF_EXPECTS((source_has_nulls && dest_has_nulls) || (!source_has_nulls),
                 "Missing destination validity buffer");
    
    if(src_col->data == tmp_col->data && src_col != nullptr){
      CUDA_TRY(cudaMalloc(&(tmp_col->data), (tmp_col->size)*cudf::size_of(tmp_col->dtype)));
      v_data_in_place[i] = true;
    }
    if(src_col->valid == tmp_col->valid){
      CUDA_TRY(cudaMalloc(&(tmp_col->valid), gdf_valid_allocation_size(tmp_col->size)));
      v_valid_in_place[i] = true;
    }
  }

  // Allocate the device_table to be passed into the kernel
  auto d_source_table = device_table::create(*source_table);
  auto d_destination_table = device_table::create(temp_output);
  // Allocate memory for reduction
  gdf_size_type* d_count_p;
  CUDA_TRY(cudaMalloc(&d_count_p, sizeof(gdf_size_type)*n_cols));
  CUDA_TRY(cudaMemset(d_count_p, 0, sizeof(gdf_size_type)*n_cols));

  // Call the optimized gather kernel
  constexpr gdf_size_type BLOCK_SIZE{256};
  const gdf_size_type gather_grid_size =
      (destination_table->num_rows() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (check_bounds) {
    impl::gather_kernel<true><<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
        *d_source_table, gather_map, *d_destination_table, d_count_p);
  } else {
    impl::gather_kernel<false><<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
        *d_source_table, gather_map, *d_destination_table, d_count_p);
  }
  
  std::vector<gdf_size_type> h_count(n_cols);
  CUDA_TRY(cudaMemcpy(h_count.data(), d_count_p, sizeof(gdf_size_type)*n_cols, cudaMemcpyDeviceToHost));
  
  for(gdf_size_type i = 0; i < destination_table->num_columns(); i++){
    destination_table->get_column(i)->null_count = destination_table->get_column(i)->size - h_count[i];
  }
  
  CUDA_TRY(cudaFree(d_count_p));
  
  // Handle the in place part
  for(gdf_size_type i = 0; i < n_cols; i++){
    gdf_column* tmp_col = temp_output.get_column(i);
    gdf_column* dest_col = destination_table->get_column(i);
    if(v_data_in_place[i]){
      CUDA_TRY(cudaMemcpy(dest_col->data, tmp_col->data, (dest_col->size)*size_of(dest_col->dtype), cudaMemcpyDeviceToDevice));
      CUDA_TRY(cudaFree(tmp_col->data));
    }
    if(v_valid_in_place[i]){
      CUDA_TRY(cudaMemcpy(dest_col->valid, tmp_col->valid, gdf_valid_allocation_size(dest_col->size), cudaMemcpyDeviceToDevice));
      CUDA_TRY(cudaFree(tmp_col->valid));
    }
  }
}

}  // namespace detail

void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table) {
  detail::gather(source_table, gather_map, destination_table, false, 0);
  nvcategory_gather_table(*source_table, *destination_table);
}

} // namespace opt
}  // namespace cudf
