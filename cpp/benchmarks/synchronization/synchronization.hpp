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

/**---------------------------------------------------------------------------*
 * @file synchroniazation.hpp
 * @brief This is the header file for `cuda_event_timer`.
 *---------------------------------------------------------------------------**/

/**---------------------------------------------------------------------------*
 * @brief  This class serves as a wrapper for using `cudaEvent_t` as the user 
 * defined timer within the framework of google benchmark 
 * (https://github.com/google/benchmark).
 *
 * It is built on top of the idea of Resource acquisition is initialization 
 * (RAII). In the following we show a minimal example of how to use this class.

    #include <benchmark/benchmark.h>
    
    static void sample_cuda_benchmark(benchmark::State& state) {
      
      for (auto _ : state){
        
        cudaStream_t stream = 0;
        
        // Create (Construct) an object of this class. You HAVE to pass in the
        // benchmark::State object you are using. It measures the time from its 
        // creation to its destruction that is spent on the specified CUDA stream.
        // It also clears the L2 cache by cudaMemset'ing a device buffer that is of
        // the size of the L2 cache (if flush_l2_cache is set to true and there is 
        // an L2 cache on the current device).
        cuda_event_timer raii(state, true, stream); // flush_l2_cache = true
        
        // Now perform the operations that is to be benchmarked
        sample_kernel<<<1, 256, 0, stream>>>(); // Possibly launching a CUDA kernel

      }
    }

    // Register the function as a benchmark. You will need to set the `UseManualTime()`
    // flag in order to use the timer embeded in this class.
    BENCHMARK(sample_cuda_benchmark)->UseManualTime();


 *---------------------------------------------------------------------------**/

#ifndef CUDF_BENCH_SYNCHRONIZATION_H
#define CUDF_BENCH_SYNCHRONIZATION_H

// Google Benchmark library
#include <benchmark/benchmark.h>

#include <tests/utilities/cudf_test_utils.cuh>

class cuda_event_timer {
 
public:

  /**---------------------------------------------------------------------------*
   * @brief This c'tor clears the L2$ by cudaMemset'ing a buffer of L2$ size
   * and starts the timer.
   *
   * @param[in,out] state  This is the benchmark::State whose timer we are going 
   * to update.
   * @param[in] flush_l2_cache_ whether or not to flush the L2 cache before
   *                            every iteration.
   * @param[in] stream_ The CUDA stream we are measuring time on.
   *---------------------------------------------------------------------------**/
  cuda_event_timer(benchmark::State& state,
                   bool flush_l2_cache,
                   cudaStream_t stream_ = 0);
 
  // The user will HAVE to provide a benchmark::State object to set 
  // the timer so we disable the default c'tor.
  cuda_event_timer() = delete;  
  
  // The d'tor stops the timer and performs a synchroniazation. 
  // Time of the benchmark::State object provided to the c'tor
  // will be set to the value given by `cudaEventElapsedTime`.
  ~cuda_event_timer();

private:
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaStream_t stream;
  benchmark::State* p_state;
};

#endif
