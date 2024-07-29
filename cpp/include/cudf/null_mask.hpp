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

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <vector>

namespace CUDF_EXPORT cudf {

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
 * @throw std::invalid_argument if state is UNINITIALIZED
 *
 * @param state The state of the null mask
 * @param size The number of elements represented by the mask
 * @return The count of null elements
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
 * @return The necessary number of bytes
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
 * @return The necessary number of `bitmask_type` elements
 */
size_type num_bitmask_words(size_type number_of_bits);

/**
 * @brief Creates a `device_buffer` for use as a null value indicator bitmask of
 * a `column`.
 *
 * @param size The number of elements to be represented by the mask
 * @param state The desired state of the mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A `device_buffer` for use as a null bitmask
 * satisfying the desired size and state
 */
rmm::device_buffer create_null_mask(
  size_type size,
  mask_state state,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Sets a pre-allocated bitmask buffer to a given state in the range
 *  `[begin_bit, end_bit)`
 *
 * Sets `[begin_bit, end_bit)` bits of bitmask to valid if `valid==true`
 * or null otherwise.
 *
 * @param bitmask Pointer to bitmask (e.g. returned by `column_viewnull_mask()`)
 * @param begin_bit Index of the first bit to set (inclusive)
 * @param end_bit Index of the last bit to set (exclusive)
 * @param valid If true set all entries to valid; otherwise, set all to null
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   rmm::cuda_stream_view stream = cudf::get_default_stream());

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
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A `device_buffer` containing the bits
 * `[begin_bit, end_bit)` from `mask`.
 */
rmm::device_buffer copy_bitmask(
  bitmask_type const* mask,
  size_type begin_bit,
  size_type end_bit,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Copies `view`'s bitmask from the bits
 * `[view.offset(), view.offset() + view.size())` into a `device_buffer`
 *
 * Returns empty `device_buffer` if the column is not nullable
 *
 * @param view Column view whose bitmask needs to be copied
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A `device_buffer` containing the bits
 * `[view.offset(), view.offset() + view.size())` from `view`'s bitmask.
 */
rmm::device_buffer copy_bitmask(
  column_view const& view,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs bitwise AND of the bitmasks of columns of a table. Returns
 * a pair of resulting mask and count of unset bits.
 *
 * If any of the columns isn't nullable, it is considered all valid.
 * If no column in the table is nullable, an empty bitmask is returned.
 *
 * @param view The table of columns
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A pair of resulting bitmask and count of unset bits
 */
std::pair<rmm::device_buffer, size_type> bitmask_and(
  table_view const& view,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs bitwise OR of the bitmasks of columns of a table. Returns
 * a pair of resulting mask and count of unset bits.
 *
 * If any of the columns isn't nullable, it is considered all valid.
 * If no column in the table is nullable, an empty bitmask is returned.
 *
 * @param view The table of columns
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return A pair of resulting bitmask and count of unset bits
 */
std::pair<rmm::device_buffer, size_type> bitmask_or(
  table_view const& view,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Given a validity bitmask, counts the number of null elements (unset bits)
 * in the range `[start, stop)`.
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and the
 * function returns ``.
 *
 * @throws cudf::logic_error if `start > stop`
 * @throws cudf::logic_error if `start < 0`
 *
 * @param bitmask Validity bitmask residing in device memory.
 * @param start Index of the first bit to count (inclusive).
 * @param stop Index of the last bit to count (exclusive).
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The number of null elements in the specified range.
 */
cudf::size_type null_count(bitmask_type const* bitmask,
                           size_type start,
                           size_type stop,
                           rmm::cuda_stream_view stream = cudf::get_default_stream());
/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
