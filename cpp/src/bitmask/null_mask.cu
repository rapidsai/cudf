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

#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.cuh>
#include <utilities/cuda_utils.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/integer_utils.hpp>

#include <cub/cub.cuh>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device_memory_resource.hpp>

namespace cudf {

size_type state_null_count(mask_state state, size_type size) {
  switch (state) {
    case UNALLOCATED:
      return 0;
    case UNINITIALIZED:
      return UNKNOWN_NULL_COUNT;
    case ALL_NULL:
      return size;
    case ALL_VALID:
      return 0;
    default:
      CUDF_FAIL("Invalid null mask state.");
  }
}

// Computes required allocation size of a bitmask
std::size_t bitmask_allocation_size_bytes(size_type number_of_bits,
                                          std::size_t padding_boundary) {
  CUDF_EXPECTS(padding_boundary > 0, "Invalid padding boundary");
  auto necessary_bytes =
      cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes =
      padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                             necessary_bytes, padding_boundary);
  return padded_bytes;
}

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size, mask_state state,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource *mr) {
  size_type mask_size{0};

  if (state != UNALLOCATED) {
    mask_size = bitmask_allocation_size_bytes(size);
  }

  rmm::device_buffer mask(mask_size, stream, mr);

  if (state != UNINITIALIZED) {
    uint8_t fill_value = (state == ALL_VALID) ? 0xff : 0x00;
    CUDA_TRY(cudaMemsetAsync(static_cast<bitmask_type *>(mask.data()),
                             fill_value, mask_size, stream));
  }

  return mask;
}

namespace {

/**---------------------------------------------------------------------------*
 * @brief Counts the number of non-zero bits in a bitmask in the range
 * `[first_bit_index, last_bit_index]`.
 *
 * Expects `0 <= first_bit_index <= last_bit_index`.
 *
 * @param[in] bitmask The bitmask whose non-zero bits will be counted.
 * @param[in] first_bit_index The index (inclusive) of the first bit to count
 * @param[in] last_bit_index The index (inclusive) of the last bit to count
 * @param[out] count The number of non-zero bits in the specified range
 *---------------------------------------------------------------------------**/
template <size_type block_size>
__global__ void count_set_bits_kernel(bitmask_type *const bitmask,
                                      size_type first_bit_index,
                                      size_type last_bit_index,
                                      size_type *global_count) {
  size_type first_word_index = word_index(first_bit_index);
  size_type last_word_index = word_index(last_bit_index);

  size_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  // thread index shifted by the index of the first counted word
  size_type thread_word_index = tid + first_word_index;

  size_type thread_count{0};

  // The number of uncounted bits in the word before `first_bit_index`
  auto pre_slack_bits = first_bit_index % detail::size_in_bits<bitmask_type>();
  // The number of uncounted bits in the word after `last_bit_index`
  auto post_slack_bits =
      detail::size_in_bits<bitmask_type>() -
      (last_bit_index % detail::size_in_bits<bitmask_type>());

  // Start/Stop lie within the same word
  if (first_word_index == last_word_index) {
    if (thread_word_index == first_word_index) {
      bitmask_type word = bitmask[first_word_index];

      // Mask off the pre slack bits
      word = word & ~((1 << pre_slack_bits) - 1);

      // Shift off the post slack bits
      word = word << post_slack_bits;

      thread_count = __popc(word);
    }
  } else {
    if (thread_word_index == first_word_index) {
      bitmask_type first_word = bitmask[first_word_index];
      // Mask off the pre slack bits
      first_word = first_word & ~((bitmask_type{1} << pre_slack_bits) - 1);
      thread_count += __popc(first_word);
    }

    while (first_word_index < thread_word_index < last_word_index) {
      thread_count += __popc(bitmask[thread_word_index]);
      thread_word_index += blockDim.x * gridDim.x;
    }

    if (thread_word_index == last_word_index) {
      bitmask_type last_word = bitmask[last_word_index];
      // Shift off the post slack bits
      last_word = last_word << post_slack_bits;
      thread_count += __popc(last_word);
    }
  }

  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type block_count{BlockReduce(temp_storage).Sum(thread_count)};

  if (threadIdx.x == 0) {
    atomicAdd(global_count, block_count);
  }
}
}  // namespace

namespace detail {
cudf::size_type count_set_bits(bitmask_type *const bitmask, size_type start,
                               size_type stop, cudaStream_t stream = 0) {
  if (nullptr == bitmask) {
    return 0;
  }

  CUDF_EXPECTS(start <= stop, "Invalid bit range.");

  std::size_t num_bits_to_count = stop - start;

  if (num_bits_to_count == 0) {
    return 0;
  }

  auto num_words = cudf::util::div_rounding_up_safe(
      num_bits_to_count, detail::size_in_bits<bitmask_type>());

  constexpr size_type block_size{256};

  cudf::util::cuda::grid_config_1d grid(num_words, block_size);

  rmm::device_scalar<size_type> non_zero_count(0, stream);

  count_set_bits_kernel<block_size><<<grid.num_blocks, grid.num_threads_per_block, 0,
                          stream>>>(bitmask, start, (stop - 1), non_zero_count.get());

  return non_zero_count.value();
}
}  // namespace detail

// Count non-zero bits in the specified range
cudf::size_type count_set_bits(bitmask_type *const bitmask, size_type start,
                               size_type stop) {
  return detail::count_set_bits(bitmask, start, stop);
}

}  // namespace cudf
