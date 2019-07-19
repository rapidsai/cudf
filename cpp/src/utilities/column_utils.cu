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

#include "column_utils.hpp"
#include <utilities/error_utils.hpp>
#include <cudf/types.h>
#include <utilities/bit_util.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <utilities/cuda_utils.hpp>
using bit_mask::bit_mask_t;

namespace cudf {

constexpr int warp_size = 32;

template <bool source_mask_valid, typename BitContainer, typename Predicate, typename Size>
__global__ void null_if_kernel(
    const BitContainer* source_mask, 
    BitContainer* destination_mask, 
    Predicate p,
    Size num_bits
  ){
 
  static_assert(warp_size == util::size_in_bits<BitContainer>());

  Size bit_index = threadIdx.x + blockIdx.x * blockDim.x;

  BitContainer active_threads =
      __ballot_sync(0xffffffff, bit_index < num_bits);

  while (bit_index < num_bits) {
    
    bool const predicate_is_true = p(bit_index);
    // This function name is `null_if` so if predicate is true the bit is to be set to off
    const BitContainer ballot_result =
        __ballot_sync(active_threads, !predicate_is_true);

    const Size container_index = 
      util::detail::bit_container_index<BitContainer>(bit_index);
    
    const BitContainer result_mask = source_mask_valid ?
        source_mask[container_index] & ballot_result : ballot_result;

    // Only one thread writes output
    if (0 == threadIdx.x % warp_size) {
      destination_mask[container_index] = result_mask;
    }

    bit_index += blockDim.x * gridDim.x;
    active_threads =
        __ballot_sync(active_threads, bit_index < num_bits);
  }

}

template <typename BitContainer, typename Predicate, typename Size>
BitContainer* null_if(
    const BitContainer* source_mask, 
    const Predicate& p,
    Size num_bits
  ){
  
  BitContainer* destination_mask = nullptr;
  CUDF_EXPECTS(GDF_SUCCESS == bit_mask::create_bit_mask(&destination_mask, num_bits), 
      "Failed to allocate bit_mask buffer.");

  auto f = source_mask ? null_if_kernel<true, BitContainer, Predicate, Size> :
    null_if_kernel<false, BitContainer, Predicate, Size> ;

  constexpr int block_size = 256;
  const int grid_size = util::cuda::grid_config_1d(num_bits, block_size).num_blocks;
  // launch the kernel
  f<<<grid_size, block_size>>>(source_mask, destination_mask, p, num_bits);
  return destination_mask;

}


template <typename T>
struct predicate_is_nan{
  
  CUDA_HOST_DEVICE_CALLABLE
  bool operator()(gdf_index_type index) const {
      return isnan(static_cast<T*>(input.data)[index]);
  }
  
  gdf_column input;

  predicate_is_nan() = delete;
  
  predicate_is_nan(const gdf_column input_): input(input_) {}

};


} // namespace cudf

bit_mask_t* nans_to_nulls(gdf_column const* col){
  
  const bit_mask_t* source_mask = reinterpret_cast<bit_mask_t*>(col->valid);
  
  switch(col->dtype){
    case GDF_FLOAT32:
      return cudf::null_if(source_mask, cudf::predicate_is_nan<float>(*col), col->size);
    case GDF_FLOAT64:
      return cudf::null_if(source_mask, cudf::predicate_is_nan<double>(*col), col->size);
    default:
      CUDF_EXPECTS(false, "Unsupported data type for is_nan()");
      return nullptr;
  }

}

