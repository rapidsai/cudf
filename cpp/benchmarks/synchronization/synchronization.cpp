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

#include "synchronization.hpp"

cuda_event_timer::cuda_event_timer(benchmark::State& state,
                                   bool flush_l2_cache,
                                   cudaStream_t stream):
  p_state(&state), stream(stream) {
  // flush all of L2$
  if(flush_l2_cache) {
    int current_device = 0;
    CUDA_TRY(cudaGetDevice(&current_device));
    
    int l2_cache_bytes = 0;
    CUDA_TRY(cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));
    
    if (l2_cache_bytes > 0) {
      const int memset_value = 0;
      int* l2_cache_buffer = nullptr;
      RMM_TRY(RMM_ALLOC(&l2_cache_buffer, l2_cache_bytes, stream));
      CUDA_TRY(cudaMemsetAsync(l2_cache_buffer, memset_value, l2_cache_bytes, stream));
      RMM_TRY(RMM_FREE(l2_cache_buffer, stream));
    }
  }
 
  CUDA_TRY(cudaEventCreate(&start));
  CUDA_TRY(cudaEventCreate(&stop));
  CUDA_TRY(cudaEventRecord(start, stream));
}

cuda_event_timer::~cuda_event_timer() {
  CUDA_TRY(cudaEventRecord(stop, stream));
  CUDA_TRY(cudaEventSynchronize(stop));
 
  float milliseconds = 0.0f;
  CUDA_TRY(cudaEventElapsedTime(&milliseconds, start, stop));
  p_state->SetIterationTime(milliseconds/(1000.0f));
  CUDA_TRY(cudaEventDestroy(start));
  CUDA_TRY(cudaEventDestroy(stop));
}

