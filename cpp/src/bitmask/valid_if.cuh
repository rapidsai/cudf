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

#pragma once

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/device_scalar.hpp>

namespace cudf {
namespace detail {
constexpr size_type warp_size{32};

template <std::size_t block_size, std::size_t lane = 0, typename T>
__device__ T single_lane_block_sum_reduce(T warp_sum) {
  static_assert(block_size <= 1024, "Invalid block size.");
  constexpr auto warps_per_block{block_size / warp_size};
  auto const lane_id{threadIdx.x % warp_size};
  auto const warp_id{threadIdx.x / warp_size};
  __shared__ T warp_sums[warp_size];

  if (lane_id == lane) {
    warp_sums[warp_id] = (warp_id < warps_per_block) ? warp_sum : 0;
  }

  __syncthreads();

  T result{0};
  if (warp_id == 0) {
    __shared__ typename cub::WarpReduce<T>::TempStorage temp;
    result = cub::WarpReduce<T>(temp).sum(warp_sums[lane_id]);
  }

  return result;
}

template <size_type block_size, typename InputIterator, typename Predicate>
valid_if_kernel(bitmask_type* output, InputIterator begin, InputIterator end,
                Predicate p, size_type* valid_count) {
  auto const lane_id{threadIdx.x % warp_size};
  auto const warp_id{threadIdx.x / warp_size};
  auto const tid = threadIdx.x + blockIdx.x * gridDim.x;
  auto i = begin + tid;
  size_type warp_valid_count{0};

  auto active_mask = __ballot_sync(0xFFFF'FFFF, i < end);
  while (i < end) {
    bitmask_type ballot = __ballot_sync(active_mask, p(*i));
    if (lane_id == 0) {
      auto bit_index = thrust::distance(begin, i);
      output[cudf::detail::word_index(bit_index)] = ballot;
      warp_valid_count += __popc(ballot);
    }
    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < end);
  }

  if (warp_id == 0) {
    atomicAdd(valid_count,
              single_lane_block_sum_reduce<block_size>(warp_valid_count));
  }
}  // namespace detail

/**
 * @brief Generate a new bitmask where every bit is set for which a predicate is
 * `true` over the elements in `[begin,end)`.
 *
 * Bit `i` in the output mask will be set if `p(*(begin+i)) == true`.
 *
 * @param begin The beginning of the sequence
 * @param end The end of the sequence
 * @param p The predicate
 * @param stream Stream on which to execute all GPU activity and device memory
 * allocations.
 * @return A `device_buffer` containing the new bitmask and it's null count
 */
template <typename InputIterator, typename Predicate>
std::pair<rmm::device_buffer, size_type> valid_if(
    InputIterator begin, InputIterator end, Predicate&& p,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  auto size = thrust::distance(begin, end);

  auto null_mask =
      create_null_mask(size, mask_state::UNINITIALIZED, stream, mr);

  rmm::device_scalar<size_type> valid_count{0, stream, mr};
}
}  // namespace detail
   // namespace cudf