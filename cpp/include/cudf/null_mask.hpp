/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <vector>

namespace cudf {

/**
 * @addtogroup column_nullmask
 * @{
 * @file
 * @brief APIs for managing validity bitmasks
 */

/**
 * @brief Returns the null count for a null mask of the specified `state`
 * representing `size` elements.
 *
 * @param state The state of the null mask
 * @param size The number of elements represented by the mask
 * @return size_type The count of null elements
 */
size_type state_null_count(mask_state state, size_type size);

/**
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
 */
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

/**
 * @brief Creates a `device_buffer` for use as a null value indicator bitmask of
 * a `column`.
 *
 * @param size The number of elements to be represented by the mask
 * @param state The desired state of the mask
 * @param mr Device memory resource used to allocate the returned device_buffer.
 * @return rmm::device_buffer A `device_buffer` for use as a null bitmask
 * satisfying the desired size and state
 */
rmm::device_buffer create_null_mask(
  size_type size,
  mask_state state,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Sets a pre-allocated bitmask buffer to a given state in the range
 *  `[begin_bit, end_bit)`
 *
 * Sets `[begin_bit, end_bit)` bits of bitmask to valid if `valid==true`
 * or null otherwise.
 *
 * @param bitmask Pointer to bitmask (e.g. returned by `column_view.null_mask()`)
 * @param begin_bit Index of the first bit to set (inclusive)
 * @param end_bit Index of the last bit to set (exclusive)
 * @param valid If true set all entries to valid; otherwise, set all to null.
 */
void set_null_mask(bitmask_type* bitmask, size_type begin_bit, size_type end_bit, bool valid);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in the range
 * `[start, stop)`
 *
 * Returns `0` if `bitmask == nullptr`.
 *
 * @throws cudf::logic_error if `start > stop`
 * @throws cudf::logic_error if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted
 * @param start_bit Index of the first bit to count (inclusive)
 * @param stop_bit Index of the last bit to count (exclusive)
 * @return The number of non-zero bits in the specified range
 */
cudf::size_type count_set_bits(bitmask_type const* bitmask, size_type start, size_type stop);

/**
 * @brief Given a bitmask, counts the number of unset (0) bits  in the range
 *`[start, stop)`.
 *
 * Returns `0` if `bitmask == nullptr`.
 *
 * @throws cudf::logic_error if `start > stop`
 * @throws cudf::logic_error if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted
 * @param start_bit Index of the first bit to count (inclusive)
 * @param stop_bit Index of the last bit to count (exclusive)
 * @return The number of zero bits in the specified range
 */
cudf::size_type count_unset_bits(bitmask_type const* bitmask, size_type start, size_type stop);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in every range
 * `[indices[2*i], indices[(2*i)+1])` (where 0 <= i < indices.size() / 2).
 *
 * Returns an empty vector if `bitmask == nullptr`.
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or
 * indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Bitmask residing in device memory whose bits will be
 * counted
 * @param[in] indices A vector of indices used to specify ranges to count the
 * number of set bits
 * @return std::vector<size_type> A vector storing the number of non-zero bits
 * in the specified ranges
 */
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                std::vector<cudf::size_type> const& indices);

/**
 * @brief Given a bitmask, counts the number of unset (0) bits in every range
 * `[indices[2*i], indices[(2*i)+1])` (where 0 <= i < indices.size() / 2).
 *
 * Returns an empty vector if `bitmask == nullptr`.
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or
 * indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Bitmask residing in device memory whose bits will be
 * counted
 * @param[in] indices A vector of indices used to specify ranges to count the
 * number of unset bits
 * @return std::vector<size_type> A vector storing the number of zero bits in
 * the specified ranges
 */
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  std::vector<cudf::size_type> const& indices);

/**
 * @brief Creates a `device_buffer` from a slice of bitmask defined by a range
 * of indices `[begin_bit, end_bit)`.
 *
 * Returns empty `device_buffer` if `bitmask == nullptr`.
 *
 * @throws cudf::logic_error if `begin_bit > end_bit`
 * @throws cudf::logic_error if `begin_bit < 0`
 *
 * @param mask Bitmask residing in device memory whose bits will be copied
 * @param begin_bit Index of the first bit to be copied (inclusive)
 * @param end_bit Index of the last bit to be copied (exclusive)
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer A `device_buffer` containing the bits
 * `[begin_bit, end_bit)` from `mask`.
 */
rmm::device_buffer copy_bitmask(
  bitmask_type const* mask,
  size_type begin_bit,
  size_type end_bit,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Copies `view`'s bitmask from the bits
 * `[view.offset(), view.offset() + view.size())` into a `device_buffer`
 *
 * Returns empty `device_buffer` if the column is not nullable
 *
 * @param view Column view whose bitmask needs to be copied
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer A `device_buffer` containing the bits
 * `[view.offset(), view.offset() + view.size())` from `view`'s bitmask.
 */
rmm::device_buffer copy_bitmask(
  column_view const& view,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a bitwise AND of the bitmasks of columns of a table
 *
 * If any of the columns isn't nullable, it is considered all valid.
 * If no column in the table is nullable, an empty bitmask is returned.
 *
 * @param view The table of columns
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer Output bitmask
 */
rmm::device_buffer bitmask_and(
  table_view const& view,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
