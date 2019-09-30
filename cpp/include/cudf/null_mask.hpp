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

#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device_memory_resource.hpp>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Returns the null count for a null mask of the specified `state`
 * representing `size` elements.
 *
 * @param state The state of the null mask
 * @param size The number of elements represented by the mask
 * @return size_type The count of null elements
 *---------------------------------------------------------------------------**/
size_type state_null_count(mask_state state, size_type size);

/**---------------------------------------------------------------------------*
 * @brief Computes the required bytes necessary to represent the specified
 * number of bits with a given padding boundary.
 *
 * @note The Arrow specification for the null bitmask requires a 64B padding
 * boundary.
 *
 * @param number_of_bits The number of bits that need to be represented
 * @param padding_boundary The value returned will be rounded up to a multiple
 * of this value
 * @return std::size_t The necessary number of bytes
 *---------------------------------------------------------------------------**/
std::size_t bitmask_allocation_size_bytes(size_type number_of_bits,
                                          std::size_t padding_boundary = 64);

/**---------------------------------------------------------------------------*
 * @brief Creates a `device_buffer` for use as a null value indicator bitmask of
 * a `column`.
 *
 * @param size The number of elements to be represented by the mask
 * @param state The desired state of the mask
 * @param stream Optional, stream on which all memory allocations/operations
 * will be submitted
 * @param mr Device memory resource to use for device memory allocation
 * @return rmm::device_buffer A `device_buffer` for use as a null bitmask
 * satisfying the desired size and state
 *---------------------------------------------------------------------------**/
rmm::device_buffer create_null_mask(
    size_type size, mask_state state, cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Given a bitmask, counts the number of set (1) bits in the range
 * `[start, stop)`
 *
 * Returns `0` if `bitmask == nullptr`.
 *
 * @throws `cudf::logic_error` if `start > stop`
 * @throws `cudf::logic_error` if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted
 * @param start_bit Index of the first bit to count (inclusive)
 * @param stop_bit Index of the last bit to count (exclusive)
 * @return The number of non-zero bits in the specified range
 *---------------------------------------------------------------------------**/
cudf::size_type count_set_bits(bitmask_type const* bitmask, size_type start,
                               size_type stop);

/**---------------------------------------------------------------------------*
 * @brief Given a bitmask, counts the number of unset (0) bits  in the range
 *`[start, stop)`.
 *
 * Returns `0` if `bitmask == nullptr`.
 *
 * @throws `cudf::logic_error` if `start > stop`
 * @throws `cudf::logic_error` if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted
 * @param start_bit Index of the first bit to count (inclusive)
 * @param stop_bit Index of the last bit to count (exclusive)
 * @return The number of zero bits in the specified range
 *---------------------------------------------------------------------------**/
cudf::size_type count_unset_bits(bitmask_type const* bitmask, size_type start,
                                 size_type stop);

}  // namespace cudf
