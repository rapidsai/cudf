/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>
#include <cuda/std/type_traits>

namespace cudf {
namespace detail {
/**
 * @brief Size of a warp in a CUDA kernel.
 */
static constexpr size_type warp_size{32};

/**
 * @brief A kernel grid configuration construction gadget for simple
 * one-dimensional kernels, with protection against integer overflow.
 */
class grid_1d {
 public:
  thread_index_type const num_threads_per_block;
  thread_index_type const num_blocks;
  /**
   * @param overall_num_elements The number of elements the kernel needs to
   * handle/process, in its main, one-dimensional/linear input (e.g. one or more
   * cuDF columns)
   * @param num_threads_per_block The grid block size, determined according to
   * the kernel's specific features (amount of shared memory necessary, SM
   * functional units use pattern etc.); this can't be determined
   * generically/automatically (as opposed to the number of blocks)
   * @param elements_per_thread Typically, a single kernel thread processes more
   * than a single element; this affects the number of threads the grid must
   * contain
   */
  grid_1d(thread_index_type overall_num_elements,
          thread_index_type num_threads_per_block,
          thread_index_type elements_per_thread = 1)
    : num_threads_per_block(num_threads_per_block),
      num_blocks(util::div_rounding_up_safe(overall_num_elements,
                                            elements_per_thread * num_threads_per_block))
  {
    CUDF_EXPECTS(num_threads_per_block > 0, "num_threads_per_block must be > 0");
    CUDF_EXPECTS(num_blocks > 0, "num_blocks must be > 0");
  }

  /**
   * @brief Returns the global thread index in a 1D grid.
   *
   * The returned index is unique across the entire grid.
   *
   * @param thread_id The thread index within the block
   * @param block_id The block index within the grid
   * @param num_threads_per_block The number of threads per block
   * @return thread_index_type The global thread index
   */
  __device__ static constexpr thread_index_type global_thread_id(
    thread_index_type thread_id,
    thread_index_type block_id,
    thread_index_type num_threads_per_block)
  {
    return thread_id + block_id * num_threads_per_block;
  }

  /**
   * @brief Returns the global thread index of the current thread in a 1D grid.
   *
   * @return thread_index_type The global thread index
   */
  static __device__ thread_index_type global_thread_id()
  {
    return global_thread_id(threadIdx.x, blockIdx.x, blockDim.x);
  }

  /**
   * @brief Returns the global thread index of the current thread in a 1D grid.
   *
   * @tparam num_threads_per_block The number of threads per block
   *
   * @return thread_index_type The global thread index
   */
  template <thread_index_type num_threads_per_block>
  static __device__ thread_index_type global_thread_id()
  {
    return global_thread_id(threadIdx.x, blockIdx.x, num_threads_per_block);
  }

  /**
   * @brief Returns the stride of a 1D grid.
   *
   * The returned stride is the total number of threads in the grid.
   *
   * @param thread_id The thread index within the block
   * @param block_id The block index within the grid
   * @param num_threads_per_block The number of threads per block
   * @return thread_index_type The global thread index
   */
  __device__ static constexpr thread_index_type grid_stride(thread_index_type num_threads_per_block,
                                                            thread_index_type num_blocks_per_grid)
  {
    return num_threads_per_block * num_blocks_per_grid;
  }

  /**
   * @brief Returns the stride of the current 1D grid.
   *
   * @return thread_index_type The number of threads in the grid.
   */
  static __device__ thread_index_type grid_stride() { return grid_stride(blockDim.x, gridDim.x); }

  /**
   * @brief Returns the stride of the current 1D grid.
   *
   * @tparam num_threads_per_block The number of threads per block
   *
   * @return thread_index_type The number of threads in the grid.
   */
  template <thread_index_type num_threads_per_block>
  static __device__ thread_index_type grid_stride()
  {
    return grid_stride(num_threads_per_block, gridDim.x);
  }
};

/**
 * @brief Performs a sum reduction of values from the same lane across all
 * warps in a thread block and returns the result on thread 0 of the block.
 *
 * All threads in a block must call this function, but only values from the
 * threads indicated by `leader_lane` will contribute to the result. Similarly,
 * the returned result is only defined on `threadIdx.x==0`.
 *
 * @tparam block_size The number of threads in the thread block (must be less
 * than or equal to 1024)
 * @tparam leader_lane The id of the lane in the warp whose value contributes to
 * the reduction
 * @tparam T Arithmetic type
 * @param lane_value The value from the lane that contributes to the reduction
 * @return The sum reduction of the values from each lane. Only valid on
 * `threadIdx.x == 0`. The returned value on all other threads is undefined.
 */
template <int32_t block_size, int32_t leader_lane = 0, typename T>
__device__ T single_lane_block_sum_reduce(T lane_value)
{
  static_assert(block_size <= 1024, "Invalid block size.");
  static_assert(cuda::std::is_arithmetic_v<T>, "Invalid non-arithmetic type.");
  constexpr auto warps_per_block{block_size / warp_size};
  auto const lane_id{threadIdx.x % warp_size};
  auto const warp_id{threadIdx.x / warp_size};
  __shared__ T lane_values[warp_size];

  // Load each lane's value into a shared memory array
  if (lane_id == leader_lane) { lane_values[warp_id] = lane_value; }
  __syncthreads();

  // Use a single warp to do the reduction, result is only defined on
  // threadId.x == 0
  T result{0};
  if (warp_id == 0) {
    __shared__ typename cub::WarpReduce<T>::TempStorage temp;
    lane_value = (lane_id < warps_per_block) ? lane_values[lane_id] : T{0};
    result     = cub::WarpReduce<T>(temp).Sum(lane_value);
  }
  // Shared memory has block scope, so sync here to ensure no data
  // races between successive calls to this function in the same
  // kernel.
  __syncthreads();
  return result;
}

/**
 * @brief Finds the smallest value not less than `number_to_round` and modulo `modulus` is
 * zero. Expects modulus to be a power of 2.
 *
 * @note Does not throw or otherwise verify the user has passed in a modulus that is a
 * power of 2.
 *
 * @param[in] number_to_round The value to be rounded up
 * @param[in] modulus The modulus to be rounded up to.  Must be a power of 2.
 *
 * @return cudf::size_type Elements per thread that can be processed for given specification.
 */
template <typename T>
__device__ inline T round_up_pow2(T number_to_round, T modulus)
{
  return (number_to_round + (modulus - 1)) & -modulus;
}

template <class F>
CUDF_KERNEL void single_thread_kernel(F f)
{
  f();
}

/**
 * @brief single thread cuda kernel
 *
 * @tparam Functor Device functor type
 * @param functor device functor object or device lambda function
 * @param stream CUDA stream used for the kernel launch
 */
template <class Functor>
void device_single_thread(Functor functor, rmm::cuda_stream_view stream)
{
  single_thread_kernel<<<1, 1, 0, stream.value()>>>(functor);
}

}  // namespace detail
}  // namespace cudf
