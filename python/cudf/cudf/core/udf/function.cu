/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cfloat>
#include <cstdint>

using size_type = int;

// double atomicAdd
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);

  return __longlong_as_double(old);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double* address, double val)
{
  unsigned long long old = __double_as_longlong(*address);
  while (val > __longlong_as_double(old)) {
    unsigned long long assumed = old;
    if ((old = atomicCAS((unsigned long long*)address, assumed, __double_as_longlong(val))) ==
        assumed)
      break;
  }
  return __longlong_as_double(old);
}

// double atomicMin
__device__ __forceinline__ double atomicMin(double* address, double val)
{
  unsigned long long old = __double_as_longlong(*address);
  while (val < __longlong_as_double(old)) {
    unsigned long long assumed = old;
    if ((old = atomicCAS((unsigned long long*)address, assumed, __double_as_longlong(val))) ==
        assumed)
      break;
  }
  return __longlong_as_double(old);
}

extern "C" __device__ int BlockSum_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_sum           = 0;

  __shared__ int64_t sum;

  if (tid == 0) sum = 0;

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd((unsigned long long*)&sum, (unsigned long long)local_sum);

  __syncthreads();

  *numba_return_value = sum;

  return 0;
}

extern "C" __device__ int BlockSum_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_sum            = 0;

  __shared__ double sum;

  if (tid == 0) sum = 0;

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd(&sum, local_sum);

  __syncthreads();

  *numba_return_value = sum;

  return 0;
}

extern "C" __device__ int BlockMean_int64(double* numba_return_value,
                                          int64_t const* data,
                                          int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_sum           = 0;
  double mean;

  __shared__ int64_t sum;

  if (tid == 0) sum = 0;

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd((unsigned long long*)&sum, (unsigned long long)local_sum);

  __syncthreads();

  mean = sum / static_cast<double>(size);

  *numba_return_value = mean;

  return 0;
}

extern "C" __device__ int BlockMean_float64(double* numba_return_value,
                                            double const* data,
                                            int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_sum            = 0;
  double mean;

  __shared__ double sum;

  if (tid == 0) sum = 0;

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd(&sum, local_sum);

  __syncthreads();

  mean = sum / static_cast<double>(size);

  *numba_return_value = mean;

  return 0;
}

extern "C" __device__ int BlockStd_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_sum           = 0;
  double local_var            = 0;
  double mean;
  double std;

  __shared__ int64_t sum;
  __shared__ double var;

  if (tid == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd((unsigned long long*)&sum, (unsigned long long)local_sum);

  __syncthreads();

  mean = sum / static_cast<double>(size);

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      double temp  = load - mean;
      temp         = pow(temp, 2);
      local_var += temp;
    }
  }

  atomicAdd(&var, local_var);

  __syncthreads();

  std = sqrt(var / (size - 1));

  *numba_return_value = std;

  return 0;
}

extern "C" __device__ int BlockStd_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_sum            = 0;
  double local_var            = 0;
  double mean;
  double std;

  __shared__ double sum;
  __shared__ double var;

  if (tid == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd(&sum, local_sum);

  __syncthreads();

  mean = sum / static_cast<double>(size);

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      double temp = load - mean;
      temp        = pow(temp, 2);
      local_var += temp;
    }
  }

  atomicAdd(&var, local_var);

  __syncthreads();

  std = sqrt(var / (size - 1));

  *numba_return_value = std;

  return 0;
}

extern "C" __device__ int BlockVar_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_sum           = 0;
  double local_var            = 0;
  double mean;

  __shared__ int64_t sum;
  __shared__ double var;

  if (tid == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd((unsigned long long*)&sum, (unsigned long long)local_sum);

  __syncthreads();

  mean = sum / static_cast<double>(size);

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      double temp  = load - mean;
      temp         = pow(temp, 2);
      local_var += temp;
    }
  }

  atomicAdd(&var, local_var);

  __syncthreads();

  var = var / (size - 1);

  *numba_return_value = var;

  return 0;
}

extern "C" __device__ int BlockVar_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_sum            = 0;
  double local_var            = 0;
  double mean;

  __shared__ double sum;
  __shared__ double var;

  if (tid == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd(&sum, local_sum);

  __syncthreads();

  mean = sum / static_cast<double>(size);

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      double temp = load - mean;
      temp        = pow(temp, 2);
      local_var += temp;
    }
  }

  atomicAdd(&var, local_var);

  __syncthreads();

  var = var / (size - 1);

  *numba_return_value = var;

  return 0;
}

// Calculate maximum of the group, return the scalar
extern "C" __device__ int BlockMax_int32(int* numba_return_value, int* data, int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int local_max               = INT_MIN;

  __shared__ int smax;

  if (tid == 0) smax = INT_MIN;

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int load  = data[tid + item * tb_size];
      local_max = max(local_max, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax(&smax, local_max);

  __syncthreads();

  *numba_return_value = smax;

  return 0;
}

// Calculate maximum of the group, return the scalar
extern "C" __device__ int BlockMax_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_max           = INT64_MIN;

  __shared__ int64_t smax;

  if (tid == 0) smax = INT64_MIN;

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      local_max    = max(local_max, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax((long long*)(&smax), (long long)local_max);

  __syncthreads();

  *numba_return_value = smax;

  return 0;
}

// Calculate maximum of the group, return the scalar
extern "C" __device__ int BlockMax_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_max            = -DBL_MAX;

  __shared__ double smax;

  if (tid == 0) smax = -DBL_MAX;

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      local_max   = max(local_max, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax((&smax), local_max);

  __syncthreads();

  *numba_return_value = smax;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockMin_int32(int* numba_return_value, int* data, int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int local_min               = INT_MAX;

  __shared__ int smin;

  if (tid == 0) smin = INT_MAX;

  __syncthreads();

#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int load  = data[tid + item * tb_size];
      local_min = min(local_min, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMin(&smin, local_min);

  __syncthreads();

  *numba_return_value = smin;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockMin_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_min           = INT64_MAX;

  __shared__ int64_t smin;

  if (tid == 0) smin = INT64_MAX;

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      local_min    = min(local_min, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMin((long long*)(&smin), (long long)local_min);

  __syncthreads();

  *numba_return_value = smin;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockMin_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_min            = DBL_MAX;

  __shared__ double smin;

  if (tid == 0) smin = DBL_MAX;

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      local_min   = min(local_min, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMin((&smin), local_min);

  __syncthreads();

  *numba_return_value = smin;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMax_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_max           = INT64_MIN;
  int64_t local_idx           = -1;

  __shared__ int64_t smax;
  __shared__ int64_t sidx;

  if (tid == 0) {
    smax = INT64_MIN;
    sidx = INT64_MAX;
  }

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      if (load > local_max) {
        local_max = load;
        local_idx = index[tid + item * tb_size];
      }
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax((long long*)(&smax), (long long)local_max);

  __syncthreads();

  if (local_max == smax) { atomicMin((long long*)(&sidx), (long long)local_idx); }

  __syncthreads();

  *numba_return_value = sidx;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMax_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_max            = -DBL_MAX;
  int64_t local_idx           = -1;

  __shared__ double smax;
  __shared__ int64_t sidx;

  if (tid == 0) {
    smax = -DBL_MAX;
    sidx = INT64_MAX;
  }

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      if (load > local_max) {
        local_max = load;
        local_idx = index[tid + item * tb_size];
      }
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax((&smax), local_max);

  __syncthreads();

  if (local_max == smax) { atomicMin((long long*)(&sidx), (long long)local_idx); }

  __syncthreads();

  *numba_return_value = sidx;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMin_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  int64_t local_min           = INT64_MAX;
  int64_t local_idx           = -1;

  __shared__ int64_t smin;
  __shared__ int64_t sidx;

  if (tid == 0) {
    smin = INT64_MAX;
    sidx = INT64_MAX;
  }

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      int64_t load = data[tid + item * tb_size];
      if (load < local_min) {
        local_min = load;
        local_idx = index[tid + item * tb_size];
      }
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMin((long long*)(&smin), (long long)local_min);

  __syncthreads();

  if (local_min == smin) { atomicMin((long long*)(&sidx), (long long)local_idx); }

  __syncthreads();

  *numba_return_value = sidx;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMin_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  int tid     = threadIdx.x;
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  double local_min            = DBL_MAX;
  int64_t local_idx           = -1;

  __shared__ double smin;
  __shared__ int64_t sidx;

  if (tid == 0) {
    smin = DBL_MAX;
    sidx = INT64_MAX;
  }

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      double load = data[tid + item * tb_size];
      if (load < local_min) {
        local_min = load;
        local_idx = index[tid + item * tb_size];
      }
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMin((&smin), local_min);

  __syncthreads();

  if (local_min == smin) { atomicMin((long long*)(&sidx), (long long)local_idx); }

  __syncthreads();

  *numba_return_value = sidx;

  return 0;
}