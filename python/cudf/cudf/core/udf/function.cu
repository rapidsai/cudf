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

// int64_t atomicAdd
__device__ __forceinline__ int64_t atomicAdd(int64_t* address, int64_t val)
{
  return atomicAdd((unsigned long long*)address, (unsigned long long)val);
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

// int64_t atomicMax
__device__ __forceinline__ int64_t atomicMax(int64_t* address, int64_t val)
{
  return atomicMax((long long*)address, (long long)val);
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

// int64_t atomicMin
__device__ __forceinline__ int64_t atomicMin(int64_t* address, int64_t val)
{
  return atomicMin((long long*)address, (long long)val);
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ T device_sum(T const* data, int const items_per_thread, size_type size) {
  __shared__ T sum;
  int tid = threadIdx.x;
  int tb_size = blockDim.x;
  T local_sum = 0;

  if (tid == 0) sum = 0;

  __syncthreads();

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      T load = data[tid + item * tb_size];
      local_sum += load;
    }
  }

  atomicAdd(&sum, local_sum);

  __syncthreads();

  return sum;
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ T device_var(T const* data, int const items_per_thread, size_type size) {

  int tid = threadIdx.x;
  int tb_size = blockDim.x;

  double local_var            = 0;
  __shared__ double var;
  if (tid == 0) var = 0;

  T sum = device_sum<T>(data, items_per_thread, size);
  double mean = sum / static_cast<double>(size);

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      T load = data[tid + item * tb_size];
      double temp  = load - mean;
      temp         = pow(temp, 2);
      local_var += temp;
    }
  }

  atomicAdd(&var, local_var);

  __syncthreads();

  return (var / (size - 1));
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ T device_max(T const* data, int const items_per_thread, size_type size, T init_val) {

  int tid = threadIdx.x;
  int tb_size = blockDim.x;

  T local_max            = init_val;
  __shared__ T smax;

  if (tid == 0) smax = init_val;

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      T load = data[tid + item * tb_size];
      local_max   = max(local_max, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax((&smax), local_max);

  __syncthreads();

  return smax;
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ T device_min(T const* data, int const items_per_thread, size_type size, T init_val) {

  int tid = threadIdx.x;
  int tb_size = blockDim.x;

  T local_min           = init_val;
  __shared__ T smin;

  if (tid == 0) smin = init_val;

  __syncthreads();

// Calculate local min for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      T load = data[tid + item * tb_size];
      local_min    = min(local_min, load);
    }
  }

  __syncthreads();

  // Calculate local min for each group
  atomicMin((&smin), local_min);

  __syncthreads();

  return smin;
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ T device_idxmax(T const* data, int const items_per_thread, int64_t const* index, size_type size, T init_val) {

  int tid     = threadIdx.x;
  int tb_size = blockDim.x;

  // Calculate how many elements each thread is working on
  T local_max            = init_val;
  int64_t local_idx           = -1;

  __shared__ T smax;
  __shared__ int64_t sidx;

  if (tid == 0) {
    smax = init_val;
    sidx = INT64_MAX;
  }

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      T load = data[tid + item * tb_size];
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

  if (local_max == smax) { atomicMin((&sidx),local_idx); }

  __syncthreads();

  return sidx;
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ T device_idxmin(T const* data, int const items_per_thread, int64_t const* index, size_type size, T init_val) {

  int tid = threadIdx.x;
  int tb_size = blockDim.x;

  T local_min            = init_val;
  int64_t local_idx           = -1;

  __shared__ T smin;
  __shared__ int64_t sidx;

  if (tid == 0) {
    smin = init_val;
    sidx = INT64_MAX;
  }

  __syncthreads();

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (tid + (item * tb_size) < size) {
      T load = data[tid + item * tb_size];
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

  if (local_min == smin) { atomicMin((&sidx), local_idx); }

  __syncthreads();

  return sidx;
}

extern "C" __device__ int BlockSum_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;
  
  int64_t sum = device_sum<int64_t>(data, items_per_thread, size);

  *numba_return_value = sum;

  return 0;
}

extern "C" __device__ int BlockSum_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double sum = device_sum<double>(data, items_per_thread, size);

  *numba_return_value = sum;

  return 0;
}

extern "C" __device__ int BlockMean_int64(double* numba_return_value,
                                          int64_t const* data,
                                          int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  int64_t sum = device_sum<int64_t>(data, items_per_thread, size);

  double mean = sum / static_cast<double>(size);

  *numba_return_value = mean;

  return 0;
}

extern "C" __device__ int BlockMean_float64(double* numba_return_value,
                                            double const* data,
                                            int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double sum = device_sum<double>(data, items_per_thread, size);

  double mean = sum / static_cast<double>(size);

  *numba_return_value = mean;

  return 0;
}

extern "C" __device__ int BlockStd_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double var = device_var<int64_t>(data, items_per_thread, size);

  *numba_return_value = sqrt(var);

  return 0;
}

extern "C" __device__ int BlockStd_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double var = device_var<double>(data, items_per_thread, size);

  *numba_return_value = sqrt(var);

  return 0;
}

extern "C" __device__ int BlockVar_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double var = device_var<int64_t>(data, items_per_thread, size);

  *numba_return_value = var;

  return 0;
}

extern "C" __device__ int BlockVar_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double var = device_var<double>(data, items_per_thread, size);

  *numba_return_value = var;

  return 0;
}

// Calculate maximum of the group, return the scalar
extern "C" __device__ int BlockMax_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  int64_t max_val = device_max<int64_t>(data, items_per_thread, size, INT64_MIN);

  *numba_return_value = max_val;

  return 0;
}

// Calculate maximum of the group, return the scalar
extern "C" __device__ int BlockMax_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double max_val = device_max<double>(data, items_per_thread, size, -DBL_MAX);

  *numba_return_value = max_val;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockMin_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  int64_t min_val = device_min<int64_t>(data, items_per_thread, size, INT64_MAX);

  *numba_return_value = min_val;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockMin_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  double min_val = device_min<double>(data, items_per_thread, size, DBL_MAX);

  *numba_return_value = min_val;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMax_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  int64_t idxmax = device_idxmax<int64_t>(data, items_per_thread, index, size, INT64_MIN);

  *numba_return_value = idxmax;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMax_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  int64_t idxmax = device_idxmax<double>(data, items_per_thread, index, size, -DBL_MAX);

  *numba_return_value = idxmax;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMin_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  int64_t idxmin = device_idxmin<int64_t>(data, items_per_thread, index, size, INT64_MAX);

  *numba_return_value = idxmin;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMin_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  int tb_size = blockDim.x;
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + tb_size - 1) / tb_size;

  int64_t idxmin = device_idxmin<double>(data, items_per_thread, index, size, DBL_MAX);

  *numba_return_value = idxmin;

  return 0;
}