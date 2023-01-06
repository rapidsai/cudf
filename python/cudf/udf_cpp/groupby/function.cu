/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
__device__ __forceinline__ double atomicAdds(double* address, double val)
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
__device__ __forceinline__ int64_t atomicAdds(int64_t* address, int64_t val)
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
__device__ void device_sum(T const* data, int const items_per_thread, size_type size, T* sum)
{
  T local_sum = 0;

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load = data[threadIdx.x + item * blockDim.x];
      local_sum += load;
    }
  }

  atomicAdds(sum, local_sum);

  __syncthreads();
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ void device_var(
  T const* data, int const items_per_thread, size_type size, T* sum, double* var)
{
  // Calculate how many elements each thread is working on
  T local_sum      = 0;
  double local_var = 0;
  double mean;

  device_sum<T>(data, items_per_thread, size, sum);

  __syncthreads();

  mean = (*sum) / static_cast<double>(size);

// Calculate local sum for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load      = data[threadIdx.x + item * blockDim.x];
      double temp = load - mean;
      temp        = pow(temp, 2);
      local_var += temp;
    }
  }

  atomicAdds(var, local_var);

  __syncthreads();

  *var = *var / (size - 1);

  __syncthreads();
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ void device_max(
  T const* data, int const items_per_thread, size_type size, T init_val, T* smax)
{
  T local_max = init_val;

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load    = data[threadIdx.x + item * blockDim.x];
      local_max = max(local_max, load);
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax(smax, local_max);

  __syncthreads();
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ void device_min(
  T const* data, int const items_per_thread, size_type size, T init_val, T* smin)
{
  T local_min = init_val;

// Calculate local min for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load    = data[threadIdx.x + item * blockDim.x];
      local_min = min(local_min, load);
    }
  }

  __syncthreads();

  // Calculate local min for each group
  atomicMin(smin, local_min);

  __syncthreads();
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ void device_idxmax(T const* data,
                              int const items_per_thread,
                              int64_t const* index,
                              size_type size,
                              T init_val,
                              T* smax,
                              int64_t* sidx)
{
  // Calculate how many elements each thread is working on
  T local_max       = init_val;
  int64_t local_idx = -1;

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load = data[threadIdx.x + item * blockDim.x];
      if (load > local_max) {
        local_max = load;
        local_idx = index[threadIdx.x + item * blockDim.x];
      }
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMax(smax, local_max);

  __syncthreads();

  if (local_max == (*smax)) { atomicMin(sidx, local_idx); }

  __syncthreads();
}

// Use a C++ templated __device__ function to implement the body of the algorithm.
template <typename T>
__device__ void device_idxmin(T const* data,
                              int const items_per_thread,
                              int64_t const* index,
                              size_type size,
                              T init_val,
                              T* smin,
                              int64_t* sidx)
{
  T local_min       = init_val;
  int64_t local_idx = -1;

// Calculate local max for each thread
#pragma unroll
  for (size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load = data[threadIdx.x + item * blockDim.x];
      if (load < local_min) {
        local_min = load;
        local_idx = index[threadIdx.x + item * blockDim.x];
      }
    }
  }

  __syncthreads();

  // Calculate local max for each group
  atomicMin(smin, local_min);

  __syncthreads();

  if (local_min == (*smin)) { atomicMin(sidx, local_idx); }

  __syncthreads();
}

extern "C" __device__ int BlockSum_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t sum;
  if (threadIdx.x == 0) { sum = 0; }

  __syncthreads();

  device_sum<int64_t>(data, items_per_thread, size, &sum);

  *numba_return_value = sum;

  return 0;
}

extern "C" __device__ int BlockSum_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double sum;
  if (threadIdx.x == 0) { sum = 0; }

  __syncthreads();

  device_sum<double>(data, items_per_thread, size, &sum);

  *numba_return_value = sum;

  return 0;
}

extern "C" __device__ int BlockMean_int64(double* numba_return_value,
                                          int64_t const* data,
                                          int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t sum;
  if (threadIdx.x == 0) { sum = 0; }

  __syncthreads();

  device_sum<int64_t>(data, items_per_thread, size, &sum);

  double mean = sum / static_cast<double>(size);

  *numba_return_value = mean;

  return 0;
}

extern "C" __device__ int BlockMean_float64(double* numba_return_value,
                                            double const* data,
                                            int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double sum;
  if (threadIdx.x == 0) { sum = 0; }

  __syncthreads();

  device_sum<double>(data, items_per_thread, size, &sum);

  double mean = sum / static_cast<double>(size);

  *numba_return_value = mean;

  return 0;
}

extern "C" __device__ int BlockStd_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t sum;
  __shared__ double var;

  if (threadIdx.x == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

  device_var<int64_t>(data, items_per_thread, size, &sum, &var);

  *numba_return_value = sqrt(var);

  return 0;
}

extern "C" __device__ int BlockStd_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double sum;
  __shared__ double var;

  if (threadIdx.x == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

  device_var<double>(data, items_per_thread, size, &sum, &var);

  *numba_return_value = sqrt(var);

  return 0;
}

extern "C" __device__ int BlockVar_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t sum;
  __shared__ double var;

  if (threadIdx.x == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

  device_var<int64_t>(data, items_per_thread, size, &sum, &var);

  *numba_return_value = var;

  return 0;
}

extern "C" __device__ int BlockVar_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double sum;
  __shared__ double var;

  if (threadIdx.x == 0) {
    sum = 0;
    var = 0;
  }

  __syncthreads();

  device_var<double>(data, items_per_thread, size, &sum, &var);

  *numba_return_value = var;

  return 0;
}

// Calculate maximum of the group, return the scalar
extern "C" __device__ int BlockMax_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t smax;

  if (threadIdx.x == 0) { smax = INT64_MIN; }

  __syncthreads();

  device_max<int64_t>(data, items_per_thread, size, INT64_MIN, &smax);

  *numba_return_value = smax;

  return 0;
}

// Calculate maximum of the group, return the scalar
extern "C" __device__ int BlockMax_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double smax;

  if (threadIdx.x == 0) { smax = -DBL_MAX; }

  __syncthreads();

  device_max<double>(data, items_per_thread, size, -DBL_MAX, &smax);

  *numba_return_value = smax;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockMin_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t smin;

  if (threadIdx.x == 0) { smin = INT64_MAX; }

  __syncthreads();

  device_min<int64_t>(data, items_per_thread, size, INT64_MAX, &smin);

  *numba_return_value = smin;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockMin_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double smin;

  if (threadIdx.x == 0) { smin = DBL_MAX; }

  __syncthreads();

  device_min<double>(data, items_per_thread, size, DBL_MAX, &smin);

  *numba_return_value = smin;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMax_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t smax;
  __shared__ int64_t sidx;

  if (threadIdx.x == 0) {
    smax = INT64_MIN;
    sidx = INT64_MAX;
  }

  __syncthreads();

  device_idxmax<int64_t>(data, items_per_thread, index, size, INT64_MIN, &smax, &sidx);

  *numba_return_value = sidx;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMax_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double smax;
  __shared__ int64_t sidx;

  if (threadIdx.x == 0) {
    smax = -DBL_MAX;
    sidx = INT64_MAX;
  }

  __syncthreads();

  device_idxmax<double>(data, items_per_thread, index, size, -DBL_MAX, &smax, &sidx);

  *numba_return_value = smax;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMin_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ int64_t smin;
  __shared__ int64_t sidx;

  if (threadIdx.x == 0) {
    smin = INT64_MAX;
    sidx = INT64_MAX;
  }

  __syncthreads();

  device_idxmin<int64_t>(data, items_per_thread, index, size, INT64_MAX, &smin, &sidx);

  *numba_return_value = sidx;

  return 0;
}

// Calculate minimum of the group, return the scalar
extern "C" __device__ int BlockIdxMin_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  // Calculate how many elements each thread is working on
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ double smin;
  __shared__ int64_t sidx;

  if (threadIdx.x == 0) {
    smin = DBL_MAX;
    sidx = INT64_MAX;
  }

  __syncthreads();

  device_idxmin<double>(data, items_per_thread, index, size, DBL_MAX, &smin, &sidx);

  *numba_return_value = sidx;

  return 0;
}
