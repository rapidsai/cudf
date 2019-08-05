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

#ifndef __BITMASK_VALID_IF_CUH__
#define __BITMASK_VALID_IF_CUH__

#include <cudf/types.h>
#include <utilities/error_utils.hpp>
#include <utilities/bit_util.cuh>
#include <utilities/cuda_utils.hpp>
#include <bitmask/legacy/bit_mask.cuh>

using bit_mask::bit_mask_t;

namespace cudf {

namespace detail {

constexpr int warp_size = 32;

/**
 * @brief for each warp in the block do a reduction (summation) of the
 * `__popc(bit_mask)` on a certain lane (default is lane 0).
 * @param[in] bit_mask The bit_mask to be reduced.
 * @return[out] result of each block is returned in thread 0.
 */
template <class bit_container, int lane = 0>
__device__ __inline__ gdf_size_type single_lane_popc_block_reduce(bit_container bit_mask) {
  
  static __shared__ gdf_size_type smem[warp_size];
  
  int lane_id = (threadIdx.x % warp_size);
  int warp_id = (threadIdx.x / warp_size);

  // Assuming one lane of each warp holds the value that we want to perform
  // reduction
  if (lane_id == lane) {
    smem[warp_id] = __popc(bit_mask);
  }
  __syncthreads();

  if (warp_id == 0) {
    // Here I am assuming maximum block size is 1024 and 1024 / 32 = 32
    // so one single warp is enough to do the reduction over different warps
    bit_mask = (lane_id < (blockDim.x / warp_size)) ? smem[lane_id] : 0;
    
    // The cub::warpReduce could be replaced with a __shfl_down_sync?
    /**
      #pragma unroll
      for(int offset = warp_size/2; offset > 0; offset /= 2){
        bit_mask += __shfl_down_sync(0xffffffffu, bit_mask, offset);
      }
    */
    __shared__
        typename cub::WarpReduce<gdf_size_type>::TempStorage temp_storage;
    bit_mask = cub::WarpReduce<gdf_size_type>(temp_storage).Sum(bit_mask);
  }

  return bit_mask;

}

template <bool source_mask_valid, typename bit_container, typename predicate, typename size_type>
__global__ void valid_if_kernel(
    const bit_container* source_mask, 
    bit_container* destination_mask, 
    predicate p,
    size_type num_bits,
    size_type* p_valid_count
  ){
 
  static_assert(warp_size == util::size_in_bits<bit_container>(), 
      "warp size is different from bit_container size.");

  size_type bit_index_base = blockIdx.x * blockDim.x;

  while (bit_index_base < num_bits) {
  
    size_type bit_index = bit_index_base + threadIdx.x;
   
    bool thread_active = bit_index < num_bits;
    bit_container active_threads =
        __ballot_sync(0xffffffff, thread_active);

    bit_container result_mask = 0;

    if(thread_active){
      
      bool const predicate_is_true = p(bit_index);
      const bit_container ballot_result =
          __ballot_sync(active_threads, predicate_is_true);

      // Only one thread writes output
      if (0 == threadIdx.x % warp_size) {
        const size_type container_index = 
          util::detail::bit_container_index<bit_container>(bit_index);

        result_mask = source_mask_valid ?
          source_mask[container_index] & ballot_result : ballot_result;
        destination_mask[container_index] = result_mask;
      }
    
    }
    
    result_mask = single_lane_popc_block_reduce(result_mask);
    if(0 == threadIdx.x){
      atomicAdd(p_valid_count, result_mask);
    }
    
    bit_index_base += blockDim.x * gridDim.x;
  
  }

}

} // namespace detail

template <typename bit_container, typename predicate, typename size_type>
std::pair<bit_container*, size_type> valid_if(
    const bit_container* source_mask, 
    const predicate& p,
    size_type num_bits,
    cudaStream_t stream = 0
  ){
  
  bit_container* destination_mask = nullptr;
  CUDF_EXPECTS(GDF_SUCCESS == bit_mask::create_bit_mask(&destination_mask, num_bits), 
      "Failed to allocate bit_mask buffer.");

  auto kernel = source_mask ? 
    detail::valid_if_kernel<true,  bit_container, predicate, size_type> :
    detail::valid_if_kernel<false, bit_container, predicate, size_type> ;

  rmm::device_vector<size_type> valid_count(1);
  
  constexpr int block_size = 256;
  const int grid_size = util::cuda::grid_config_1d(num_bits, block_size).num_blocks;
  
  // launch the kernel
  kernel<<<grid_size, block_size, 0, stream>>>(
      source_mask, destination_mask, p, num_bits, valid_count.data().get());

  size_type valid_count_host;
  CUDA_TRY(cudaMemcpy(&valid_count_host, valid_count.data().get(),
        sizeof(size_type), cudaMemcpyDeviceToHost));
  
  // Synchronize the stream before null_count is updated on the host.
  cudaStreamSynchronize(stream);
  size_type null_count = num_bits - valid_count_host;

  CHECK_STREAM(stream);
  return std::pair<bit_container*, size_type>(destination_mask, null_count);

}

} // namespace cudf
#endif
