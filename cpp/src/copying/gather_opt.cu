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
#include <utilities/cudf_utils.h>
#include <utilities/type_dispatcher.hpp>
#include <cudf/table.hpp>
#include <string/nvcategory_util.hpp>

#include <algorithm>

#include <table/device_table.cuh>

#include <utilities/column_utils.hpp>

#include <bitmask/bit_mask.cuh>
#include <cub/cub.cuh>

#include <cooperative_groups.h>

/**
 * @brief Operations for copying from one column to another
 * @file copying_ops.cu
 */

using bit_mask::bit_mask_t;

namespace impl {

constexpr int warp_size = 32;
/**
template<class T>
__device__ __inline__ void warp_wise_reduce(T& f) {
  #pragma unroll
  for(int offset = 16; offset > 0; offset /= 2){
    // ONLY works for CUDA 9.2 or later
    float other_f = __shfl_down_sync(0xffffffffu, f, offset);
    f += other_f;
  }
}
*/
template<class CountType, int lane = 0>
__device__ __inline__ void single_lane_reduce(CountType f, CountType* d_output){
  
  static __shared__ gdf_size_type smem[warp_size];
  
  int lane_id = (threadIdx.x % warp_size);
  int warp_id = (threadIdx.x / warp_size);
  
  // Assuming lane 0 of each warp holds the value that we want to perform reduction
  if(lane_id == lane){
    smem[warp_id] = f;
  }
  __syncthreads();
  
  if(warp_id == 0){ // Here I am assuming maximum block size is 1024 and 1024 >> 5 = 32
    f = (lane_id < (blockDim.x / warp_size)) ? smem[lane_id] : 0;

    __shared__ typename cub::WarpReduce<CountType>::TempStorage temp_storage;
    f = cub::WarpReduce<CountType>(temp_storage).Sum(f);
    
    // warp_wise_reduce(f); 
    if(lane_id == 0){
      atomicAdd(d_output, f);
    }
  }
  __syncthreads();
}

template<bool check_bounds>
__device__ __inline__ void gather_bitmask_device(
    const bit_mask_t* __restrict__ source_valid,
    const gdf_size_type source_row,
    const gdf_size_type num_source_rows, 
    bit_mask_t* __restrict__ destination_valid,
    const gdf_size_type destination_row, 
    const gdf_size_type num_destination_rows,
    gdf_size_type* d_count 
){
  const uint32_t active_threads = __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  bool source_bit_is_valid = false;
  if(check_bounds){
    source_bit_is_valid = source_row < num_source_rows && bit_mask::is_valid(source_valid, source_row);
  }else{
    source_bit_is_valid = bit_mask::is_valid(source_valid, source_row);
  }
  
  // Use ballot to find all valid bits in this warp and create the output bitmask element
  const uint32_t valid_warp = __ballot_sync(active_threads, source_bit_is_valid);

  gdf_index_type const valid_index = destination_row / warp_size;
  // Only one thread writes output
  if(0 == threadIdx.x % warp_size){
    destination_valid[valid_index] = valid_warp;
  }
  single_lane_reduce(__popc(valid_warp), d_count);
}

template<int block_size>
struct copy_element_smem {
  template <typename T>
  __device__ inline void operator()(gdf_column const& target,
                                    gdf_size_type target_index,
                                    gdf_column const& source,
                                    gdf_size_type source_index) {
    reinterpret_cast<T*>(target.data)[target_index] 
      = reinterpret_cast<const T*>(source.data)[source_index];
    // static __shared__ T smem[block_size];
    // smem[threadIdx.x] = reinterpret_cast<const T*>(source.data)[source_index];
    // reinterpret_cast<T*>(target.data)[target_index] = smem[threadIdx.x];
  }
};

template<bool check_bounds, bool do_valid, int block_size>
__launch_bounds__(block_size, 2048/block_size)
__global__ void gather_kernel(const device_table source, 
                                  const gdf_size_type gather_map[], 
                                  device_table destination,
                                  gdf_size_type* d_count)
{
  const gdf_index_type n_source_rows = source.num_rows();
  const gdf_index_type n_destination_rows = destination.num_rows();
  
  // Each element of gather_map[] only needs to be used once. 
  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;
 
  const bool active_threads = destination_row < n_destination_rows;
  gdf_index_type source_row = active_threads ? gather_map[destination_row] : 0;
  for(gdf_index_type i = 0; i < destination.num_columns(); i++){
    
    if(active_threads && (!check_bounds || source_row < n_source_rows)){
      // gather the entire row
      cudf::type_dispatcher(source.get_column(i)->dtype,
                          copy_element_smem<block_size>{},
                          *destination.get_column(i), destination_row,
                          *source.get_column(i), source_row);
    }
    
    if(do_valid){
      // Before bit_mask_t is used in device_table we will do the cast here.
      bit_mask_t* __restrict__ src_valid =
        reinterpret_cast<bit_mask_t*>(source.get_column(i)->valid);
      bit_mask_t* __restrict__ dest_valid =
        reinterpret_cast<bit_mask_t*>(destination.get_column(i)->valid);
 
      gather_bitmask_device<check_bounds>(src_valid, source_row, n_source_rows,
        dest_valid, destination_row, n_destination_rows, d_count + i);
    }
  }
}

}  // namespace
namespace cudf {
namespace opt   {
namespace detail {

template<int block_size>
void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table, bool check_bounds, cudaStream_t stream) {
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");

  // If the destination is empty, return immediately as there is nothing to gather
  if (0 == destination_table->num_rows()) {
    return;
  }

  CUDF_EXPECTS(nullptr != gather_map, "gather_map is null");
  CUDF_EXPECTS(source_table->num_columns() == destination_table->num_columns(),
    "Mismatched number of columns");
  const gdf_size_type n_cols = source_table->num_columns();
 
  bool src_has_nulls = source_table->get_column(0)->valid != nullptr;
  bool dest_has_nulls = destination_table->get_column(0)->valid != nullptr;
  
  // Perform sanity checks
  for(gdf_size_type i = 0; i < n_cols; i++){
    gdf_column* dest_col = destination_table->get_column(i);
    const gdf_column* src_col = source_table->get_column(i);
 
    CUDF_EXPECTS(src_col->dtype == dest_col->dtype, "Column type mismatch");
    
    // If one column has a valid buffer then we require all columns to have one.
    CUDF_EXPECTS((src_col->valid != nullptr) == src_has_nulls, 
      "If one column has a valid buffer all columns must have one.");
    CUDF_EXPECTS((dest_col->valid != nullptr) == dest_has_nulls, 
      "If one column has a valid buffer all columns must have one.");
    
    CUDF_EXPECTS(src_col->data != nullptr, "Missing source data buffer.");
    CUDF_EXPECTS(dest_col->data != nullptr, "Missing source data buffer.");
   
    CUDF_EXPECTS(src_col->data != dest_col->data,
      "In place gather/scatter is NOT supported.");
    CUDF_EXPECTS(src_col->valid != dest_col->valid || src_col->valid != nullptr,
      "In place gather/scatter is NOT supported.");
  }
  // If the source column has a valid buffer, the destination column must also have one
  CUDF_EXPECTS((src_has_nulls && dest_has_nulls) || (!src_has_nulls),
    "Missing destination validity buffer");

  // Allocate the device_table to be passed into the kernel
  auto d_source_table = device_table::create(*source_table);
  auto d_destination_table = device_table::create(*destination_table);
  
  // Allocate memory for reduction
  gdf_size_type* d_count_p;
  CUDA_TRY(cudaMalloc(&d_count_p, sizeof(gdf_size_type)*n_cols));
  CUDA_TRY(cudaMemset(d_count_p, 0, sizeof(gdf_size_type)*n_cols));

  // Call the optimized gather kernel
  const gdf_size_type gather_grid_size =
      (destination_table->num_rows() + block_size - 1) / block_size;

  if (check_bounds) {
    if(dest_has_nulls){
      impl::gather_kernel<true , true , block_size><<<gather_grid_size, block_size, 0, stream>>>(
          *d_source_table, gather_map, *d_destination_table, d_count_p);
    }else{
      impl::gather_kernel<true , false, block_size><<<gather_grid_size, block_size, 0, stream>>>(
          *d_source_table, gather_map, *d_destination_table, d_count_p);
    }
  }else{
    if(dest_has_nulls){
      impl::gather_kernel<false, true , block_size><<<gather_grid_size, block_size, 0, stream>>>(
          *d_source_table, gather_map, *d_destination_table, d_count_p);
    }else{
      impl::gather_kernel<false, false, block_size><<<gather_grid_size, block_size, 0, stream>>>(
          *d_source_table, gather_map, *d_destination_table, d_count_p);
    }
  }
  
  std::vector<gdf_size_type> h_count(n_cols);
  CUDA_TRY(cudaMemcpy(h_count.data(), d_count_p, sizeof(gdf_size_type)*n_cols, cudaMemcpyDeviceToHost));
  if(dest_has_nulls){  
    for(gdf_size_type i = 0; i < destination_table->num_columns(); i++){
      destination_table->get_column(i)->null_count = destination_table->get_column(i)->size - h_count[i];
    }
  }
  CUDA_TRY(cudaFree(d_count_p));
}

}  // namespace detail

void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table, int block_size) {
  switch(block_size){
  case  64:  
    detail::gather< 64>(source_table, gather_map, destination_table, false, 0); break;
  case 128:  
    detail::gather<128>(source_table, gather_map, destination_table, false, 0); break;
  case 192:  
    detail::gather<192>(source_table, gather_map, destination_table, false, 0); break;
  case 256:  
    detail::gather<256>(source_table, gather_map, destination_table, false, 0); break;
  default:
    CUDF_EXPECTS(false, "Unsupported block size.");
  }
  nvcategory_gather_table(*source_table, *destination_table);
}

} // namespace opt
}  // namespace cudf
