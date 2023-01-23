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

#include <cooperative_groups.h>

#include <limits>

template <typename T>
__device__ void device_sum(cooperative_groups::thread_block const& block,
                           T const* data,
                           int64_t size,
                           T* sum)
{
  T local_sum = 0;

#pragma unroll
  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_sum += data[idx];
  }

  cuda::atomic_ref<T, cuda::thread_scope_device> ref{*sum};
  ref.fetch_add(local_sum, cuda::std::memory_order_relaxed);

  block.sync();
}

template <typename T>
__device__ void device_var(cooperative_groups::thread_block const& block,
                           T const* data,
                           int64_t size,
                           double* var)
{
  T local_sum      = 0;
  double local_var = 0;

  __shared__ T block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  device_sum<T>(block, data, size, &block_sum);

  auto const mean = static_cast<double>(block_sum) / static_cast<double>(size);

#pragma unroll
  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto temp = static_cast<double>(data[idx]) - mean;
    temp *= temp;
    local_var += temp;
  }

  cuda::atomic_ref<double, cuda::thread_scope_device> ref{*var};
  ref.fetch_add(local_var, cuda::std::memory_order_relaxed);
  block.sync();

  *var = *var / static_cast<double>(size - 1);
  block.sync();
}

template <typename T>
__device__ T BlockSum(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  device_sum<T>(block, data, size, &block_sum);
  return block_sum;
}

template <typename T>
__device__ T BlockMean(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  device_sum<T>(block, data, size, &block_sum);
  return block_sum / static_cast<T>(size);
}

template <typename T>
__device__ double BlockStd(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ double var;
  if (block.thread_rank() == 0) { var = 0; }
  block.sync();

  device_var<T>(block, data, size, &var);
  return sqrt(var);
}

template <typename T>
__device__ double BlockVar(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ double block_var;
  if (block.thread_rank() == 0) { block_var = 0; }
  block.sync();

  device_var<T>(block, data, size, &block_var);
  return block_var;
}

template <typename T>
__device__ T BlockMax(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  auto local_max = []() {
    if constexpr (std::is_floating_point_v<T>) { return -std::numeric_limits<T>::max(); }
    return std::numeric_limits<T>::min();
  }();
  __shared__ T block_max;
  if (block.thread_rank() == 0) { block_max = local_max; }
  block.sync();

#pragma unroll
  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_max = max(local_max, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_device> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);

  block.sync();

  return block_max;
}

template <typename T>
__device__ T BlockMin(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  auto local_min = std::numeric_limits<T>::max();
  __shared__ T block_min;
  if (block.thread_rank() == 0) { block_min = local_min; }
  block.sync();

#pragma unroll
  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_min = min(local_min, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_device> ref{block_min};
  ref.fetch_min(local_min, cuda::std::memory_order_relaxed);

  block.sync();

  return block_min;
}

template <typename T>
__device__ int64_t BlockIdxMax(T const* data, int64_t* index, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_max;
  __shared__ int64_t block_idx_max;

  auto local_max = []() {
    if constexpr (std::is_floating_point_v<T>) { return -std::numeric_limits<T>::max(); }
    return std::numeric_limits<T>::min();
  }();
  auto local_idx_max = std::numeric_limits<int64_t>::max();

  if (block.thread_rank() == 0) {
    block_max     = local_max;
    block_idx_max = local_idx_max;
  }
  block.sync();

#pragma unroll
  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data > local_max) {
      local_max     = current_data;
      local_idx_max = index[idx];
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_device> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);
  block.sync();

  cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref_idx{block_idx_max};
  if (local_max == block_max) { ref_idx.fetch_min(local_idx_max, cuda::std::memory_order_relaxed); }
  block.sync();

  return block_idx_max;
}

template <typename T>
__device__ int64_t BlockIdxMin(T const* data, int64_t* index, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_min;
  __shared__ int64_t block_idx_min;

  auto local_min     = std::numeric_limits<T>::max();
  auto local_idx_min = std::numeric_limits<int64_t>::max();

  if (block.thread_rank() == 0) {
    block_min     = local_min;
    block_idx_min = local_idx_min;
  }
  block.sync();

#pragma unroll
  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data < local_min) {
      local_min     = current_data;
      local_idx_min = index[idx];
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_device> ref{block_min};
  ref.fetch_min(local_min, cuda::std::memory_order_relaxed);
  block.sync();

  cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref_idx{block_idx_min};
  if (local_min == block_min) { ref_idx.fetch_min(local_idx_min, cuda::std::memory_order_relaxed); }
  block.sync();

  return block_idx_min;
}

extern "C" {
#define make_definition(name, cname, type)                                                   \
  __device__ int name##_##cname(int64_t* numba_return_value, type* const data, int64_t size) \
  {                                                                                          \
    *numba_return_value = name<type>(data, size);                                            \
    return 0;                                                                                \
  }

// make_definition(BlockSum, int8, int8_t);
// make_definition(BlockSum, int16, int16_t);
make_definition(BlockSum, int32, int);
make_definition(BlockSum, int64, int64_t);
make_definition(BlockSum, float32, float);
make_definition(BlockSum, float64, double);
// make_definition(BlockSum, bool, bool);
// make_definition(BlockMean, int8, int8_t);
// make_definition(BlockMean, int16, int16_t);
make_definition(BlockMean, int32, int);
make_definition(BlockMean, int64, int64_t);
make_definition(BlockMean, float32, float);
make_definition(BlockMean, float64, double);
// make_definition(BlockMean, bool, bool);
// make_definition(BlockStd, int8, int8_t);
// make_definition(BlockStd, int16, int16_t);
make_definition(BlockStd, int32, int);
make_definition(BlockStd, int64, int64_t);
make_definition(BlockStd, float32, float);
make_definition(BlockStd, float64, double);
// make_definition(BlockStd, bool, bool);
// make_definition(BlockVar, int8, int8_t);
// make_definition(BlockVar, int16, int16_t);
make_definition(BlockVar, int32, int);
make_definition(BlockVar, int64, int64_t);
make_definition(BlockVar, float32, float);
make_definition(BlockVar, float64, double);
// make_definition(BlockVar, bool, bool);
// make_definition(BlockMin, int8, int8_t);
// make_definition(BlockMin, int16, int16_t);
make_definition(BlockMin, int32, int);
make_definition(BlockMin, int64, int64_t);
make_definition(BlockMin, float32, float);
make_definition(BlockMin, float64, double);
// make_definition(BlockMin, bool, bool);
// make_definition(BlockMax, int8, int8_t);
// make_definition(BlockMax, int16, int16_t);
make_definition(BlockMax, int32, int);
make_definition(BlockMax, int64, int64_t);
make_definition(BlockMax, float32, float);
make_definition(BlockMax, float64, double);
// make_definition(BlockMax, bool, bool);
#undef make_definition
}

extern "C" {
#define make_definition_idx(name, cname, type)                                   \
  __device__ int name##_##cname(                                                 \
    int64_t* numba_return_value, type* const data, int64_t* index, int64_t size) \
  {                                                                              \
    *numba_return_value = name<type>(data, index, size);                         \
    return 0;                                                                    \
  }

// make_definition_idx(BlockIdxMin, int8, int8_t);
// make_definition_idx(BlockIdxMin, int16, int16_t);
make_definition_idx(BlockIdxMin, int32, int);
make_definition_idx(BlockIdxMin, int64, int64_t);
make_definition_idx(BlockIdxMin, float32, float);
make_definition_idx(BlockIdxMin, float64, double);
// make_definition_idx(BlockIdxMin, bool, bool);
// make_definition_idx(BlockIdxMax, int8, int8_t);
// make_definition_idx(BlockIdxMax, int16, int16_t);
make_definition_idx(BlockIdxMax, int32, int);
make_definition_idx(BlockIdxMax, int64, int64_t);
make_definition_idx(BlockIdxMax, float32, float);
make_definition_idx(BlockIdxMax, float64, double);
// make_definition_idx(BlockIdxMax, bool, bool);
#undef make_definition_idx
}
