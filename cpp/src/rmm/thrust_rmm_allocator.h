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

/**
 Allocator class compatible with thrust arrays that uses RMM device memory manager.

 Author: Mark Harris
 */

#ifndef THRUST_RMM_ALLOCATOR_H
#define THRUST_RMM_ALLOCATOR_H

#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/execution_policy.h>

#include "rmm.h"

template<class T>
class rmm_allocator : public thrust::device_malloc_allocator<T>
{
  public:
    using value_type = T;

    rmm_allocator(cudaStream_t stream = 0) : stream(stream) {}
    ~rmm_allocator() {}

    typedef thrust::device_ptr<value_type>  pointer;
    inline pointer allocate(size_t n)
    {
      value_type* result = nullptr;
  
      rmmError_t error = RMM_ALLOC((void**)&result, n*sizeof(value_type), stream);
     
      if(error != RMM_SUCCESS)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "rmm_allocator::allocate(): RMM_ALLOC");
      }
  
      return thrust::device_pointer_cast(result);
    }
  
    inline void deallocate(pointer ptr, size_t)
    {
      rmmError_t error = RMM_FREE(thrust::raw_pointer_cast(ptr), stream);
  
      if(error != RMM_SUCCESS)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "rmm_allocator::deallocate(): RMM_FREE");
      }
    }

  private:
  	cudaStream_t stream;
};


namespace rmm
{
/**
 * @brief Alias for a thrust::device_vector that uses RMM for memory allocation.
 * 
 */
template <typename T>
using device_vector = thrust::device_vector<T, rmm_allocator<T>>;

/* --------------------------------------------------------------------------*/
/** 
 * @brief Returns a Thrust CUDA execution policy that uses RMM for temporary memory
 * allocation and executes on the specified stream.
 * 
 * @Param stream The stream that the execution policy will execute on.
 * 
 * @Returns A Thrust execution policy that will use RMM for temporary memory allocation
 * that runs on the specified stream.
 */
/* ----------------------------------------------------------------------------*/
inline auto exec_policy(cudaStream_t stream = 0){
  // par_t::operator() can't accept a r-value, so need to pass it an l-value
  rmm_allocator<char> allocator(stream);
  return thrust::cuda::par(allocator).on(stream);
}

} // namespace rmm

#endif // THRUST_RMM_ALLOCATOR_H
