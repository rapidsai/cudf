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

#include <thrust/gather.h>
#include <table/device_table.cuh>

#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <bitmask/bit_mask.cuh>
#include <cub/cub.cuh>

using bit_mask_t = bit_mask::bit_mask_t;

namespace impl {

constexpr int warp_size = 32;

template<class BitType, class CountType, int lane = 0>
__device__ __inline__ void single_lane_reduce(BitType f, CountType* d_output){
  
  static __shared__ gdf_size_type smem[warp_size];
  
  int lane_id = (threadIdx.x % warp_size);
  int warp_id = (threadIdx.x / warp_size);
  
  // Assuming lane 0 of each warp holds the value that we want to perform reduction
  if(lane_id == lane){
    smem[warp_id] = __popc(f);
  }
  __syncthreads();
  
  if(warp_id == 0){ // Here I am assuming maximum block size is 1024 and 1024 >> 5 = 32
    f = (lane_id < (blockDim.x / warp_size)) ? smem[lane_id] : 0;

    __shared__ typename cub::WarpReduce<CountType>::TempStorage temp_storage;
    f = cub::WarpReduce<CountType>(temp_storage).Sum(f);
    
    if(lane_id == 0){
      atomicAdd(d_output, f);
    }
  }
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
  // instead of setting it to true by default.
  const uint32_t active_threads = __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  bool source_bit_is_valid = false;
  if(check_bounds){
    if(source_row < num_source_rows){
      source_bit_is_valid = bit_mask::is_valid(source_valid, source_row);
    }else{
      // If gather_map does not include this row we should just keep the originla value,
      source_bit_is_valid = bit_mask::is_valid(destination_valid, destination_row);
    }
  }else{
    source_bit_is_valid = bit_mask::is_valid(source_valid, source_row);
  }
  
  // Use ballot to find all valid bits in this warp and create the output bitmask element
  const uint32_t valid_warp = __ballot_sync(active_threads, source_bit_is_valid);

  const gdf_index_type valid_index = destination_row / warp_size;
  // Only one thread writes output
  if(0 == threadIdx.x % warp_size){
    destination_valid[valid_index] = valid_warp;
  }
  single_lane_reduce(valid_warp, d_count);
}
  
template<bool check_bounds, int block_size>
__launch_bounds__(block_size, 2048/block_size)
__global__ void gather_bitmask_kernel(
    const bit_mask_t* const * source_valid, 
    gdf_size_type num_source_rows,
    const gdf_index_type* gather_map, 
    bit_mask_t** destination_valid,
    gdf_size_type num_destination_rows,
    gdf_size_type* d_count,
    gdf_size_type num_columns
)
{
//  const gdf_index_type num_source_rows = source.num_rows();
//  const gdf_index_type num_destination_rows = destination.num_rows();
  for(gdf_index_type i = 0; i < num_columns; i++){
    // bitmask part  
    gdf_index_type destination_row_base = blockIdx.x * blockDim.x;
/**
    // Before bit_mask_t is used in device_table we will do the cast here.
    // TODO: modify device_table to use bit_mask_t
    bit_mask_t* __restrict__ src_valid =
      reinterpret_cast<bit_mask_t*>(source.get_column(i)->valid);
    bit_mask_t* __restrict__ dest_valid =
      reinterpret_cast<bit_mask_t*>(destination.get_column(i)->valid);
    // Gather the bitmasks and do the null count 
*/
    while(destination_row_base < num_destination_rows){
        gdf_index_type destination_row = destination_row_base + threadIdx.x;
        const bool active_threads = destination_row < num_destination_rows;
        gdf_index_type source_row = active_threads ? gather_map[destination_row] : 0;
        
        gather_bitmask_device<check_bounds>(source_valid[i], source_row, num_source_rows,
          destination_valid[i], destination_row, num_destination_rows, d_count + i);
      
        destination_row_base += blockDim.x * gridDim.x; 
    }
  }
}

/**---------------------------------------------------------------------------*
 * @brief Function object for gathering a type-erased
 * gdf_column. To be used with the cudf::type_dispatcher.
 *
 *---------------------------------------------------------------------------**/
struct column_gatherer {
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * @tparam ColumnType Dispatched type for the column being gathered
   * @param source_column The column to gather from
   * @param gather_map Array of indices that maps source elements to destination
   * elements
   * @param destination_column The column to gather into
   * @param check_bounds Optionally perform bounds checking on the values of
   * `gather_map`
   * @param stream Optional CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename ColumnType>
  void operator()(gdf_column const* source_column,
                  gdf_index_type const gather_map[],
                  gdf_column* destination_column, bool check_bounds = false,
                  cudaStream_t stream = 0) {
    ColumnType const* const source_data{
        static_cast<ColumnType const*>(source_column->data)};
    ColumnType* destination_data{
        static_cast<ColumnType*>(destination_column->data)};

    gdf_size_type const num_destination_rows{destination_column->size};

    // If gathering in-place, allocate temporary buffers to hold intermediate
    // results
    rmm::device_vector<ColumnType> temp_destination;
 
    if (check_bounds) {/**
      thrust::gather_if(rmm::exec_policy(stream)->on(stream), gather_map,
                        gather_map + num_destination_rows, gather_map,
                        source_data, destination_data,
                        bounds_checker{0, source_column->size});*/
    } else {
      thrust::gather(rmm::exec_policy(stream)->on(stream), gather_map,
                     gather_map + num_destination_rows, source_data,
                     destination_data);
    }

    CHECK_STREAM(stream);
  }
};


}  // namespace

namespace cudf {
namespace opt   {
namespace detail {

template<int block_size>
void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table, bool check_bounds, cudaStream_t stream, int grid_size) {
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
 
  // We create (n_cols+1) streams for the (n_cols+1) kernels we are gonna launch.
  // std::vector<util::cuda::scoped_stream> v_stream(n_cols+1);
  std::vector<cudaStream_t> v_stream(n_cols+1, nullptr);
  for(int i = 0; i < n_cols+1; i++){
    CUDA_TRY(cudaStreamCreate(&v_stream[i]));
  }
  
  cudaStream_t bit_stream = v_stream[n_cols];
  
  // Allocate memory for bitmask reduction
  gdf_size_type* d_count_p;
  RMM_TRY(RMM_ALLOC(&d_count_p, sizeof(gdf_size_type)*n_cols, bit_stream));
  CUDA_TRY(cudaMemsetAsync(d_count_p, 0, sizeof(gdf_size_type)*n_cols, bit_stream));
 
  std::vector<bit_mask_t*> h_bit_src(n_cols);
  std::vector<bit_mask_t*> h_bit_dest(n_cols);

  // Perform sanity checks
  for(gdf_size_type i = 0; i < n_cols; i++){

    gdf_column* dest_col = destination_table->get_column(i);
    const gdf_column* src_col = source_table->get_column(i);
   
    // h_vp_src_valid[i] = reinterpret_cast<bit_mask_t*>(src_col->valid); 
    // h_vp_dest_valid[i] = reinterpret_cast<bit_mask_t*>(dest_col->valid); 

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
    
    // The data gather for n columns will be put on the first n streams
    
    // Somehow putting the following on to one or multiple non-default streams significantly deceases the L2 hit rate.
    // cudf::type_dispatcher(src_col->dtype, impl::column_gatherer{}, src_col, gather_map, dest_col, check_bounds, v_stream[i]);
    cudf::type_dispatcher(src_col->dtype, impl::column_gatherer{}, src_col, gather_map, dest_col, check_bounds);
    
    h_bit_src[i] = reinterpret_cast<bit_mask_t*>(src_col->valid);
    h_bit_dest[i] = reinterpret_cast<bit_mask_t*>(dest_col->valid);
  }

  rmm::device_vector<bit_mask_t*> d_bit_src(n_cols);
  rmm::device_vector<bit_mask_t*> d_bit_dest(n_cols);

  CUDA_TRY(cudaMemcpyAsync(d_bit_src.data().get(), h_bit_src.data(), n_cols*sizeof(bit_mask_t*), cudaMemcpyHostToDevice, bit_stream));
  CUDA_TRY(cudaMemcpyAsync(d_bit_dest.data().get(), h_bit_dest.data(), n_cols*sizeof(bit_mask_t*), cudaMemcpyHostToDevice, bit_stream));

  // If the source column has a valid buffer, the destination column must also have one
  CUDF_EXPECTS((src_has_nulls && dest_has_nulls) || (!src_has_nulls),
    "Missing destination validity buffer");

  // constexpr gdf_size_type gather_grid_size = 64/(block_size/impl::warp_size)*80;
  int gather_grid_size;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gather_grid_size, impl::gather_bitmask_kernel<false, block_size>, block_size, 0);
  gather_grid_size /= 2;

  if (check_bounds) {
    if(dest_has_nulls){
      // impl::gather_bitmask_kernel<true , block_size><<<gather_grid_size, block_size, 0, bit_stream>>>(
      //   *d_source_table, gather_map, *d_destination_table, d_count_p);
    }
  }else{
    if(dest_has_nulls){
      impl::gather_bitmask_kernel<false, block_size><<<gather_grid_size, block_size, 0, bit_stream>>>(
        d_bit_src.data().get(), source_table->num_rows(), gather_map, d_bit_dest.data().get(), destination_table->num_rows(), d_count_p, n_cols);
    }
  }
  
  std::vector<gdf_size_type> h_count(n_cols);
  CUDA_TRY(cudaMemcpyAsync(h_count.data(), d_count_p, sizeof(gdf_size_type)*n_cols, cudaMemcpyDeviceToHost, bit_stream));
  RMM_TRY(RMM_FREE(d_count_p, bit_stream));
  CUDA_TRY(cudaStreamSynchronize(bit_stream));
  if(dest_has_nulls){  
    for(gdf_size_type i = 0; i < destination_table->num_columns(); i++){
      destination_table->get_column(i)->null_count = destination_table->get_column(i)->size - h_count[i];
    }
  }

  for(int t = 0; t < n_cols+1; t++){
    CHECK_STREAM(v_stream[t]);
    CUDA_TRY(cudaStreamSynchronize(v_stream[t]));
    CUDA_TRY(cudaStreamDestroy(v_stream[t]));
  }
}

}  // namespace detail

void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table, int block_size, int grid_size) {
  switch(block_size){
  case  64:
    detail::gather< 64>(source_table, gather_map, destination_table, false, 0, grid_size); break;
  case 128:
    detail::gather<128>(source_table, gather_map, destination_table, false, 0, grid_size); break;
  case 192:
    detail::gather<192>(source_table, gather_map, destination_table, false, 0, grid_size); break;
  case 256:
    detail::gather<256>(source_table, gather_map, destination_table, false, 0, grid_size); break;
  default:
    CUDF_EXPECTS(false, "Unsupported block size.");
  }
  nvcategory_gather_table(*source_table, *destination_table);
}

__global__ void invert_map(gdf_index_type gather_map[], const gdf_size_type destination_rows,
                            gdf_index_type const scatter_map[], const gdf_size_type source_rows){
  gdf_index_type source_row = threadIdx.x + blockIdx.x * blockDim.x;
  if(source_row < source_rows){
    gdf_index_type destination_row = scatter_map[source_row];
    if(destination_row < destination_rows){
      gather_map[destination_row] = source_row;
    }
  }
}

void scatter(table const* source_table, gdf_index_type const scatter_map[],
            table* destination_table, int block_size) {
  const gdf_size_type num_source_rows = source_table->num_rows();
  const gdf_size_type num_destination_rows = destination_table->num_rows();
  // Turn the scatter_map[] into a gather_map[] and then call gather(...).
  rmm::device_vector<gdf_index_type> v_gather_map(num_destination_rows, num_destination_rows);
  
  const gdf_size_type invert_grid_size =
    (destination_table->num_rows() + block_size - 1) / block_size;

  invert_map<<<invert_grid_size, block_size>>>(v_gather_map.data().get(), num_destination_rows, scatter_map, num_source_rows);
  
  gather(source_table, v_gather_map.data().get(), destination_table, block_size);    
}

} // namespace opt
}  // namespace cudf
