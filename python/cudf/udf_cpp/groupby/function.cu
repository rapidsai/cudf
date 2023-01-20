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

#include <cudf/types.hpp>

#include <cuda/atomic>

#include <limits>

__device__ __forceinline__ double atomicMax(double* address, double val)
{
  unsigned long long old = __double_as_longlong(*address);
  while (val > __longlong_as_double(old)) {
    unsigned long long assumed = old;
    if ((old = atomicCAS(
           reinterpret_cast<unsigned long long*>(address), assumed, __double_as_longlong(val))) ==
        assumed)
      break;
  }
  return __longlong_as_double(old);
}

__device__ __forceinline__ int64_t atomicMax(int64_t* address, int64_t val)
{
  return atomicMax(reinterpret_cast<long long*>(address), static_cast<long long>(val));
}

__device__ __forceinline__ double atomicMin(double* address, double val)
{
  unsigned long long old = __double_as_longlong(*address);
  while (val < __longlong_as_double(old)) {
    unsigned long long assumed = old;
    if ((old = atomicCAS(
           reinterpret_cast<unsigned long long*>(address), assumed, __double_as_longlong(val))) ==
        assumed)
      break;
  }
  return __longlong_as_double(old);
}

__device__ __forceinline__ int64_t atomicMin(int64_t* address, int64_t val)
{
  return atomicMin(reinterpret_cast<long long*>(address), static_cast<long long>(val));
}

template <typename T>
__device__ void device_sum(T const* data, int const items_per_thread, cudf::size_type size, T* sum)
{
  T local_sum = 0;

#pragma unroll
  for (cudf::size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load = data[threadIdx.x + item * blockDim.x];
      local_sum += load;
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_device> ref{*sum};
  ref.fetch_add(local_sum, cuda::std::memory_order_relaxed);

  __syncthreads();
}

template <typename T>
__device__ void device_var(
  T const* data, int const items_per_thread, cudf::size_type size, T* sum, double* var)
{
  T local_sum      = 0;
  double local_var = 0;
  double mean;

  device_sum<T>(data, items_per_thread, size, sum);

  mean = (*sum) / static_cast<double>(size);

#pragma unroll
  for (cudf::size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load      = data[threadIdx.x + item * blockDim.x];
      double temp = load - mean;
      temp        = pow(temp, 2);
      local_var += temp;
    }
  }

  cuda::atomic_ref<double, cuda::thread_scope_device> ref{*var};
  ref.fetch_add(local_var, cuda::std::memory_order_relaxed);

  __syncthreads();

  *var = *var / (size - 1);

  __syncthreads();
}

template <typename T>
__device__ void device_idxmax(T const* data,
                              int const items_per_thread,
                              int64_t const* index,
                              cudf::size_type size,
                              T init_val,
                              T* smax,
                              int64_t* sidx)
{
  T local_max       = init_val;
  int64_t local_idx = -1;

#pragma unroll
  for (cudf::size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load = data[threadIdx.x + item * blockDim.x];
      if (load > local_max) {
        local_max = load;
        local_idx = index[threadIdx.x + item * blockDim.x];
      }
    }
  }

  atomicMax(smax, local_max);

  __syncthreads();

  if (local_max == (*smax)) { atomicMin(sidx, local_idx); }

  __syncthreads();
}

template <typename T>
__device__ void device_idxmin(T const* data,
                              int const items_per_thread,
                              int64_t const* index,
                              cudf::size_type size,
                              T init_val,
                              T* smin,
                              int64_t* sidx)
{
  T local_min       = init_val;
  int64_t local_idx = -1;

#pragma unroll
  for (cudf::size_type item = 0; item < items_per_thread; item++) {
    if (threadIdx.x + (item * blockDim.x) < size) {
      T load = data[threadIdx.x + item * blockDim.x];
      if (load < local_min) {
        local_min = load;
        local_idx = index[threadIdx.x + item * blockDim.x];
      }
    }
  }

  atomicMin(smin, local_min);

  __syncthreads();

  if (local_min == (*smin)) { atomicMin(sidx, local_idx); }

  __syncthreads();
}

template <typename T>
__device__ T BlockSum(T const* data, int64_t size)
{
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;
  __shared__ T sum;

  if (threadIdx.x == 0) { sum = 0; }
  __syncthreads();
  device_sum<T>(data, items_per_thread, size, &sum);
  return sum;
}

template <typename T>
__device__ T BlockMean(T const* data, int64_t size)
{
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;

  __shared__ T sum;
  if (threadIdx.x == 0) { sum = 0; }

  __syncthreads();
  device_sum<T>(data, items_per_thread, size, &sum);
  double mean = sum / static_cast<double>(size);
  return mean;
}

template <typename T>
__device__ T BlockStd(T const* data, int64_t size)
{
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;
  __shared__ T sum;
  __shared__ double var;
  if (threadIdx.x == 0) {
    sum = 0;
    var = 0;
  }
  __syncthreads();
  device_var<T>(data, items_per_thread, size, &sum, &var);
  return sqrt(var);
}

template <typename T>
__device__ T BlockVar(T const* data, int64_t size)
{
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;
  __shared__ T sum;
  __shared__ double var;
  if (threadIdx.x == 0) {
    sum = 0;
    var = 0;
  }
  __syncthreads();
  device_var<T>(data, items_per_thread, size, &sum, &var);
  return var;
}

template <typename T>
__device__ T BlockMax(T const* data, int64_t size)
{
  T local_max = std::numeric_limits<T>::min();
  __shared__ T block_max;
  if (threadIdx.x == 0) { block_max = local_max; }
  __syncthreads();

#pragma unroll
  for (int64_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
    local_max = max(local_max, data[idx]);
  }

  atomicMax(&block_max, local_max);
  __syncthreads();

  return block_max;
}

template <typename T>
__device__ T BlockMin(T const* data, int64_t size)
{
  T local_min = std::numeric_limits<T>::max();
  __shared__ T block_min;
  if (threadIdx.x == 0) { block_min == local_min; }
  __syncthreads();

#pragma unroll
  for (int64_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
    local_min = min(local_min, data[idx]);
  }

  atomicMin(&block_min, local_min);
  __syncthreads();

  return block_min;
}

template <typename T>
__device__ T BlockIdxMax(T const* data, int64_t* index, int64_t size)
{
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;
  __shared__ T smax;
  __shared__ int64_t sidx;
  if (threadIdx.x == 0) {
    smax = std::numeric_limits<int64_t>::min();
    sidx = std::numeric_limits<int64_t>::max();
  }
  __syncthreads();
  device_idxmax<T>(
    data, items_per_thread, index, size, std::numeric_limits<int64_t>::min(), &smax, &sidx);
  return sidx;
}

template <typename T>
__device__ T BlockIdxMin(T const* data, int64_t* index, T min, int64_t size)
{
  auto const items_per_thread = (size + blockDim.x - 1) / blockDim.x;
  __shared__ T smin;
  __shared__ int64_t sidx;
  if (threadIdx.x == 0) {
    smin = min;
    sidx = std::numeric_limits<int64_t>::max();
  }
  __syncthreads();
  device_idxmin<T>(data, items_per_thread, index, size, min, &smin, &sidx);
  return sidx;
}

extern "C" __device__ int BlockSum_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  *numba_return_value = BlockSum<int64_t>(data, size);
  return 0;
}

extern "C" __device__ int BlockSum_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  *numba_return_value = BlockSum<double>(data, size);
  return 0;
}

extern "C" __device__ int BlockMean_int64(int64_t* numba_return_value,
                                          int64_t const* data,
                                          int64_t size)
{
  *numba_return_value = BlockMean<int64_t>(data, size);
  return 0;
}

extern "C" __device__ int BlockMean_float64(double* numba_return_value,
                                            double const* data,
                                            int64_t size)
{
  *numba_return_value = BlockMean<double>(data, size);
  return 0;
}

extern "C" __device__ int BlockStd_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  *numba_return_value = BlockStd<int64_t>(data, size);
  return 0;
}

extern "C" __device__ int BlockStd_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  *numba_return_value = BlockStd<double>(data, size);
  return 0;
}

extern "C" __device__ int BlockVar_int64(double* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  *numba_return_value = BlockVar<int64_t>(data, size);
  return 0;
}

extern "C" __device__ int BlockVar_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  *numba_return_value = BlockVar<double>(data, size);
  return 0;
}

extern "C" __device__ int BlockMax_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  *numba_return_value = BlockMax<int64_t>(data, size);
  return 0;
}

extern "C" __device__ int BlockMax_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  *numba_return_value = BlockMax<double>(data, size);
  return 0;
}

extern "C" __device__ int BlockMin_int64(int64_t* numba_return_value,
                                         int64_t const* data,
                                         int64_t size)
{
  *numba_return_value = BlockMin<int64_t>(data, size);
  return 0;
}

extern "C" __device__ int BlockMin_float64(double* numba_return_value,
                                           double const* data,
                                           int64_t size)
{
  *numba_return_value = BlockMin<double>(data, size);
  return 0;
}

extern "C" __device__ int BlockIdxMax_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  *numba_return_value = BlockIdxMax<int64_t>(data, index, size);
  return 0;
}

extern "C" __device__ int BlockIdxMax_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  *numba_return_value = BlockIdxMax<double>(data, index, size);
  return 0;
}

extern "C" __device__ int BlockIdxMin_int64(int64_t* numba_return_value,
                                            int64_t const* data,
                                            int64_t* index,
                                            int64_t size)
{
  *numba_return_value =
    BlockIdxMin<int64_t>(data, index, std::numeric_limits<int64_t>::max(), size);
  return 0;
}

extern "C" __device__ int BlockIdxMin_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  *numba_return_value = BlockIdxMin<double>(data, index, std::numeric_limits<double>::max(), size);
  return 0;
}
