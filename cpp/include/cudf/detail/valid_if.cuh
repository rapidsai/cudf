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
#include <cudf/utilities/cuda.cuh>

#include <rmm/device_scalar.hpp>

namespace cudf {
namespace detail {

template <size_type block_size, typename InputIterator, typename Predicate>
valid_if_kernel(bitmask_type* output, InputIterator begin, InputIterator end,
                Predicate p, size_type* valid_count) {
  constexpr size_type leader_lane{0};
  auto const lane_id{threadIdx.x % warp_size};
  auto const warp_id{threadIdx.x / warp_size};
  auto const tid = threadIdx.x + blockIdx.x * gridDim.x;
  auto i = begin + tid;
  size_type warp_valid_count{0};

  auto active_mask = __ballot_sync(0xFFFF'FFFF, i < end);
  while (i < end) {
    bitmask_type ballot = __ballot_sync(active_mask, p(*i));
    if (lane_id == leader_lane) {
      auto bit_index = thrust::distance(begin, i);
      output[cudf::detail::word_index(bit_index)] = ballot;
      warp_valid_count += __popc(ballot);
    }
    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < end);
  }

  auto block_count =
      single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
  if (threadIdx.x == 0) {
    atomicAdd(valid_count, block_count);
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
 * @return A pair containing a `device_buffer` with the new bitmask and it's
 * null count
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

  return std::make_pair(null_mask, valid_count.value(stream));
}
}  // namespace detail
   // namespace cudf