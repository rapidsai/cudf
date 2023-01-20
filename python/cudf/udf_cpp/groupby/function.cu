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
__device__ void device_sum(T const* data, int64_t size, T* sum)
{
  T local_sum = 0;

#pragma unroll
  for (int64_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
    local_sum += data[idx];
  }

  cuda::atomic_ref<T, cuda::thread_scope_device> ref{*sum};
  ref.fetch_add(local_sum, cuda::std::memory_order_relaxed);

  __syncthreads();
}

template <typename T>
__device__ void device_var(T const* data, int64_t size, double* var)
{
  T local_sum      = 0;
  double local_var = 0;

  __shared__ T block_sum;
  if (threadIdx.x == 0) { block_sum = 0; }
  __syncthreads();

  device_sum<T>(data, size, &block_sum);

  auto const mean = static_cast<double>(block_sum) / static_cast<double>(size);

#pragma unroll
  for (int64_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
    auto temp = static_cast<double>(data[idx]) - mean;
    temp *= temp;
    local_var += temp;
  }

  cuda::atomic_ref<double, cuda::thread_scope_device> ref{*var};
  ref.fetch_add(local_var, cuda::std::memory_order_relaxed);

  __syncthreads();

  *var = *var / static_cast<double>(size - 1);

  __syncthreads();
}

template <typename T>
__device__ T BlockSum(T const* data, int64_t size)
{
  __shared__ T block_sum;
  if (threadIdx.x == 0) { block_sum = 0; }
  __syncthreads();

  device_sum<T>(data, size, &block_sum);
  return block_sum;
}

template <typename T>
__device__ T BlockMean(T const* data, int64_t size)
{
  __shared__ T block_sum;
  if (threadIdx.x == 0) { block_sum = 0; }
  __syncthreads();

  device_sum<T>(data, size, &block_sum);
  return block_sum / static_cast<T>(size);
}

template <typename T>
__device__ double BlockStd(T const* data, int64_t size)
{
  __shared__ double var;
  if (threadIdx.x == 0) { var = 0; }
  __syncthreads();

  device_var<T>(data, size, &var);
  return sqrt(var);
}

template <typename T>
__device__ double BlockVar(T const* data, int64_t size)
{
  __shared__ double block_var;
  if (threadIdx.x == 0) { block_var = 0; }
  __syncthreads();

  device_var<T>(data, size, &block_var);
  return block_var;
}

template <typename T>
__device__ T BlockMax(T const* data, int64_t size)
{
  auto local_max = std::numeric_limits<T>::min();
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
  auto local_min = std::numeric_limits<T>::max();
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
__device__ int64_t BlockIdxMax(T const* data, int64_t* index, int64_t size)
{
  __shared__ T block_max;
  __shared__ int64_t block_idx_max;

  // TODO: this is wrong but can pass tests!!!
  auto local_max     = std::numeric_limits<int64_t>::min();
  auto local_idx_max = std::numeric_limits<int64_t>::max();

  if (threadIdx.x == 0) {
    block_max     = local_max;
    block_idx_max = local_idx_max;
  }
  __syncthreads();

#pragma unroll
  for (int64_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
    auto const current_data = data[idx];
    if (current_data > local_max) {
      local_max     = current_data;
      local_idx_max = index[idx];
    }
  }

  atomicMax(&block_max, local_max);
  __syncthreads();

  if (local_max == block_max) { atomicMin(&block_idx_max, local_idx_max); }
  __syncthreads();

  return block_idx_max;
}

template <typename T>
__device__ int64_t BlockIdxMin(T const* data, int64_t* index, int64_t size)
{
  __shared__ T block_min;
  __shared__ int64_t block_idx_min;

  auto local_min     = std::numeric_limits<T>::max();
  auto local_idx_min = std::numeric_limits<int64_t>::max();

  if (threadIdx.x == 0) {
    block_min     = local_min;
    block_idx_min = local_idx_min;
  }
  __syncthreads();

#pragma unroll
  for (int64_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
    auto const current_data = data[idx];
    if (current_data < local_min) {
      local_min     = current_data;
      local_idx_min = index[idx];
    }
  }

  atomicMin(&block_min, local_min);
  __syncthreads();

  if (local_min == block_min) { atomicMin(&block_idx_min, local_idx_min); }
  __syncthreads();

  return block_idx_min;
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
  *numba_return_value = BlockIdxMin<int64_t>(data, index, size);
  return 0;
}

extern "C" __device__ int BlockIdxMin_float64(int64_t* numba_return_value,
                                              double const* data,
                                              int64_t* index,
                                              int64_t size)
{
  *numba_return_value = BlockIdxMin<double>(data, index, size);
  return 0;
}
