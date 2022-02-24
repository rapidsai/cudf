/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "generate_input.hpp"
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState* state, int size, unsigned seed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence
     number, no offset */
  if (id < size) curand_init(seed, id, 0, &state[id]);
}

//  int8_t,  int16_t,  int32_t,  int64_t,
// uint8_t, uint16_t, uint32_t, uint64_t
// float,   double

// uniform, normal, log_normal, poisson.

// will this work for int64_t, uint64_t without any overflow issues?
template <typename T>
__global__ void generate_uniform(
  curandState* state, T lower_bound, T upper_bound, unsigned size, T* result)
{
  auto range            = upper_bound - lower_bound;
  auto const tid        = threadIdx.x + blockIdx.x * blockDim.x;
  auto const gridstride = blockDim.x * gridDim.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[tid];
  /* Generate uniform random double (0, 1.0] and shift to bounds */
  for (auto id = tid; id < size; id += gridstride) {
    double x   = curand_uniform_double(&localState);
    result[id] = static_cast<T>(x * range + lower_bound);
  }
  /* Copy state back to global memory */
  state[tid] = localState;
}

template <typename T>
__global__ void generate_normal(
  curandState* state, T lower_bound, T upper_bound, unsigned size, T* result)
{
  auto range            = upper_bound - lower_bound;
  auto const tid        = threadIdx.x + blockIdx.x * blockDim.x;
  auto const gridstride = blockDim.x * gridDim.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[tid];
  /* Generate uniform random double (0, 1.0] and shift to bounds */
  for (auto id = tid; id < size; id += gridstride) {
    double x   = curand_normal_double(&localState);
    result[id] = static_cast<T>(x * range + lower_bound);
  }
  /* Copy state back to global memory */
  state[tid] = localState;
}

template <typename T>
__global__ void generate_log_normal(
  curandState* state, T lower_bound, T upper_bound, unsigned size, T* result)
{
  auto range            = upper_bound - lower_bound;
  auto const tid        = threadIdx.x + blockIdx.x * blockDim.x;
  auto const gridstride = blockDim.x * gridDim.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[tid];
  /* Generate uniform random double (0, 1.0] and shift to bounds */
  for (auto id = tid; id < size; id += gridstride) {
    double x   = curand_log_normal_double(&localState, 0.0, 1.0);
    result[id] = static_cast<T>(x * range + lower_bound);
  }
  /* Copy state back to global memory */
  state[tid] = localState;
}
