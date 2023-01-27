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

#include <cudf/detail/utilities/device_operators.cuh>

#include <cuda/atomic>

#include <cooperative_groups.h>

#include <limits>
#include <type_traits>

template <typename T>
__device__ bool are_all_nans(cooperative_groups::thread_block const& block,
                             T const* data,
                             int64_t size)
{
  // TODO: to be refactored with CG vote functions once
  // block size is known at build time
  __shared__ int64_t count;

  if (block.thread_rank() == 0) { count = 0; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    if (not std::isnan(data[idx])) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref{count};
      ref.fetch_add(1, cuda::std::memory_order_relaxed);
      break;
    }
  }

  block.sync();
  return count == 0;
}

template <typename T>
__device__ void device_sum(cooperative_groups::thread_block const& block,
                           T const* data,
                           int64_t size,
                           T* sum)
{
  T local_sum = 0;

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_sum += data[idx];
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{*sum};
  ref.fetch_add(local_sum, cuda::std::memory_order_relaxed);

  block.sync();
}

template <typename T>
__device__ T BlockSum(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return 0; }
  }

  __shared__ T block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  device_sum<T>(block, data, size, &block_sum);
  return block_sum;
}

template <typename T>
__device__ double BlockMean(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  device_sum<T>(block, data, size, &block_sum);
  return static_cast<double>(block_sum) / static_cast<double>(size);
}

template <typename T>
__device__ double BlockVar(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ double block_var;
  __shared__ T block_sum;
  if (block.thread_rank() == 0) {
    block_var = 0;
    block_sum = 0;
  }
  block.sync();

  T local_sum      = 0;
  double local_var = 0;

  device_sum<T>(block, data, size, &block_sum);

  auto const mean = static_cast<double>(block_sum) / static_cast<double>(size);

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const delta = static_cast<double>(data[idx]) - mean;
    local_var += delta * delta;
  }

  cuda::atomic_ref<double, cuda::thread_scope_block> ref{block_var};
  ref.fetch_add(local_var, cuda::std::memory_order_relaxed);
  block.sync();

  if (block.thread_rank() == 0) { block_var = block_var / static_cast<double>(size - 1); }
  block.sync();
  return block_var;
}

template <typename T>
__device__ double BlockStd(T const* data, int64_t size)
{
  auto const var = BlockVar(data, size);
  return sqrt(var);
}

template <typename T>
__device__ T BlockMax(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return std::numeric_limits<T>::quiet_NaN(); }
  }

  auto local_max = cudf::DeviceMax::identity<T>();
  __shared__ T block_max;
  if (block.thread_rank() == 0) { block_max = local_max; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_max = max(local_max, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);

  block.sync();

  return block_max;
}

template <typename T>
__device__ T BlockMin(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return std::numeric_limits<T>::quiet_NaN(); }
  }

  auto local_min = cudf::DeviceMin::identity<T>();

  __shared__ T block_min;
  if (block.thread_rank() == 0) { block_min = local_min; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_min = min(local_min, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_min};
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
  __shared__ bool found_max;

  auto local_max     = cudf::DeviceMax::identity<T>();
  auto local_idx_max = cudf::DeviceMin::identity<int64_t>();

  if (block.thread_rank() == 0) {
    block_max     = local_max;
    block_idx_max = local_idx_max;
    found_max     = false;
  }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data > local_max) {
      local_max     = current_data;
      local_idx_max = index[idx];
      found_max     = true;
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);
  block.sync();

  if (found_max) {
    if (local_max == block_max) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref_idx{block_idx_max};
      ref_idx.fetch_min(local_idx_max, cuda::std::memory_order_relaxed);
    }
  } else {
    if (block.thread_rank() == 0) { block_idx_max = index[0]; }
  }
  block.sync();

  return block_idx_max;
}

template <typename T>
__device__ int64_t BlockIdxMin(T const* data, int64_t* index, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_min;
  __shared__ int64_t block_idx_min;
  __shared__ bool found_min;

  auto local_min     = cudf::DeviceMin::identity<T>();
  auto local_idx_min = cudf::DeviceMin::identity<int64_t>();

  if (block.thread_rank() == 0) {
    block_min     = local_min;
    block_idx_min = local_idx_min;
    found_min     = false;
  }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data < local_min) {
      local_min     = current_data;
      local_idx_min = index[idx];
      found_min     = true;
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_min};
  ref.fetch_min(local_min, cuda::std::memory_order_relaxed);
  block.sync();

  if (found_min) {
    if (local_min == block_min) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref_idx{block_idx_min};
      ref_idx.fetch_min(local_idx_min, cuda::std::memory_order_relaxed);
    }
  } else {
    if (block.thread_rank() == 0) { block_idx_min = index[0]; }
  }
  block.sync();

  return block_idx_min;
}

extern "C" {
#define make_definition(name, cname, type, return_type)                                          \
  __device__ int name##_##cname(return_type* numba_return_value, type* const data, int64_t size) \
  {                                                                                              \
    return_type const res = name<type>(data, size);                                              \
    if (threadIdx.x == 0) { *numba_return_value = res; }                                         \
    __syncthreads();                                                                             \
    return 0;                                                                                    \
  }

make_definition(BlockSum, int64, int64_t, int64_t);
make_definition(BlockSum, float64, double, double);
make_definition(BlockMean, int64, int64_t, double);
make_definition(BlockMean, float64, double, double);
make_definition(BlockStd, int64, int64_t, double);
make_definition(BlockStd, float64, double, double);
make_definition(BlockVar, int64, int64_t, double);
make_definition(BlockVar, float64, double, double);
make_definition(BlockMin, int64, int64_t, int64_t);
make_definition(BlockMin, float64, double, double);
make_definition(BlockMax, int64, int64_t, int64_t);
make_definition(BlockMax, float64, double, double);
#undef make_definition
}

extern "C" {
#define make_definition_idx(name, cname, type)                                   \
  __device__ int name##_##cname(                                                 \
    int64_t* numba_return_value, type* const data, int64_t* index, int64_t size) \
  {                                                                              \
    auto const res = name<type>(data, index, size);                              \
    if (threadIdx.x == 0) { *numba_return_value = res; }                         \
    __syncthreads();                                                             \
    return 0;                                                                    \
  }

make_definition_idx(BlockIdxMin, int64, int64_t);
make_definition_idx(BlockIdxMin, float64, double);
make_definition_idx(BlockIdxMax, int64, int64_t);
make_definition_idx(BlockIdxMax, float64, double);
#undef make_definition_idx
}
