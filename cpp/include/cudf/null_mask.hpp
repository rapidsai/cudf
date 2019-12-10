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

#include <cudf/column/column_view.hpp>
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

/**
 * @brief Returns the number of `bitmask_type` words required to represent the
 * specified number of bits.
 *
 * Unlike `bitmask_allocation_size_bytes`, which returns the number of *bytes*
 * needed for a bitmask allocation (including padding), this function returns
 * the *actual* number `bitmask_type` elements necessary to represent
 * `number_of_bits`. This is useful when one wishes to process all of the bits
 * in a bitmask and ignore the padding/slack bits.
 *
 * @param number_of_bits The number of bits that need to be represented
 * @return size_type The necessary number of `bitmask_type` elements
 */
size_type num_bitmask_words(size_type number_of_bits);

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
 * @brief Sets a pre-allocated bitmask buffer to a given state
 *
 * @param bitmask Pointer to bitmask (e.g. returned by `column_view.null_mask()`)
 * @param size The number of elements represented by the mask (e.g.,
   number of rows in a column)
 * @param valid If true set all entries to valid; otherwise, set all to null.
 * @param stream Optional, stream on which all memory allocations/operations
 * will be submitted
 *---------------------------------------------------------------------------**/
  void set_null_mask(bitmask_type* bitmask,
                     size_type size, bool valid, cudaStream_t stream = 0);
  
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

/**---------------------------------------------------------------------------*
 * @brief Creates a `device_buffer` from a slice of bitmask defined by a range
 * of indices `[begin_bit, end_bit)`.
 *
 * Returns empty `device_buffer` if `bitmask == nullptr`.
 *
 * @throws `cudf::logic_error` if `begin_bit > end_bit`
 * @throws `cudf::logic_error` if `begin_bit < 0`
 *
 * @param mask Bitmask residing in device memory whose bits will be copied
 * @param begin_bit Index of the first bit to be copied (inclusive)
 * @param end_bit Index of the last bit to be copied (exclusive)
 * @param stream Optional, stream on which all memory allocations and copies
 * will be performed
 * @param mr Optional, the memory resource that will be used for allocating
 * the device memory for the new device_buffer
 * @return rmm::device_buffer A `device_buffer` containing the bits
 * `[begin_bit, end_bit)` from `mask`.
 *---------------------------------------------------------------------------**/
rmm::device_buffer copy_bitmask(
    bitmask_type const* mask, size_type begin_bit, size_type end_bit,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Copies `view`'s bitmask from the bits
 * `[view.offset(), view.offset() + view.size())` into a `device_buffer`
 *
 * Returns empty `device_buffer` if the column is not nullable
 *
 * @param view Column view whose bitmask needs to be copied
 * @param stream Optional, stream on which all memory allocations and copies
 * will be performed
 * @param mr Optional, the memory resource that will be used for allocating
 * the device memory for the new device_buffer
 * @return rmm::device_buffer A `device_buffer` containing the bits
 * `[view.offset(), view.offset() + view.size())` from `view`'s bitmask.
 *---------------------------------------------------------------------------**/
rmm::device_buffer copy_bitmask(column_view const& view,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Concatenates `views[i]`'s bitmask from the bits
 * `[views[i].offset(), views[i].offset() + views[i].size())` for all elements
 * views[i] in views into a `device_buffer`
 *
 * Returns empty `device_buffer` if the column is not nullable
 *
 * @param views Vector of column views whose bitmask will to be concatenated
 * @param mr Optional, the memory resource that will be used for allocating
 * the device memory for the new device_buffer
 * @param stream Optional, stream on which all memory allocations and copies
 * will be performed
 * @return rmm::device_buffer A `device_buffer` containing the bitmasks of all
 * the column views in the views vector
 *---------------------------------------------------------------------------**/
rmm::device_buffer concatenate_masks(std::vector<column_view> const &views,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Returns a bitwise AND of the bitmasks of two columns
 * 
 * If either of the column isn't nullable, it is considered all valid and the
 * bitmask of the other column is copied and returned. If both columns are not
 * nullable, an empty bitmask is returned
 * 
 * @note The sizes of the two columns should be the same
 * 
 * @param view1 The first column
 * @param view2 The second column
 * @param stream CUDA stream on which to execute kernels 
 * @param mr Memory resource for allocating output bitmask
 * @return rmm::device_buffer Output bitmask
 */
rmm::device_buffer bitmask_and(column_view const& view1,
    column_view const& view2,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cudf
