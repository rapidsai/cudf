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
#include "column.hpp"

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
/**---------------------------------------------------------------------------*
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified numeric `data_type` with an optional
 * null mask.
 * 
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param[in] type The desired numeric element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream Optional stream on which to issue all memory allocation and device
 * kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the column's `data` and `null_mask`.
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> make_numeric_column(
    data_type type, size_type size, mask_state state = UNALLOCATED,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Construct STRING type column given an array of pointer/size pairs.
 * The total number of char bytes must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char bytes.
 * Use the strings_column_view class to perform strings operations on this type
 * of column.
 *
 * @note `null_count()` and `null_bitmask` are determined if a pair contains
 * a null string. That is, for each pair, if `.first` is null, that string
 * is considered null. Likewise, a string is considered empty (not null)
 * if `.first` is not null and `.second` is 0. Otherwise the `.first` member
 * must be a valid device address pointing to `.second` consecutive bytes.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param strings The pointer/size pair arrays.
 *                Each pointer must be a device memory address or `nullptr` (indicating a null string).
 *                The size must be the number of bytes.
 * @param stream Optional stream for use with all memory allocation
 *               and device kernels
 * @param mr Optional resource to use for device memory
 *           allocation of the column's `null_mask` and children.
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> make_strings_column(
    const rmm::device_vector<thrust::pair<const char*,size_type>>& strings,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Construct STRING type column given an contiguous array of chars
 * encoded as UTF-8, an array of byte offsets identifying individual strings
 * within the char array, and a null bitmask.
 * The total number of char bytes must not exceed the maximum size of size_type.
 * Use the strings_column_view class to perform strings operations on this type
 * of column.
 * This function makes a deep copy of the strings, offsets, null_mask to create
 * a new column.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param strings The contiguous array of chars in device memory.
 *                This char array is expected to be UTF-8 encoded characters.
 * @param offsets The array of byte offsets in device memory.
 *                The number of elements is one more than the total number
 *                of strings so the offset[last] - offset[0] is the total
 *                number of bytes in the strings array.
 * @param null_mask The array of bits specifying the null strings.
 *                  This array must be in device memory.
 *                  Arrow format for nulls is used for interpeting this bitmask.
 * @param null_count The number of null string entries.
 * @param stream Optional stream for use with all memory allocation
 *               and device kernels
 * @param mr Optional resource to use for device memory
 *           allocation of the column's `null_mask` and children.
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> make_strings_column(
    const rmm::device_vector<char>& strings,
    const rmm::device_vector<size_type>& offsets,
    const rmm::device_vector<bitmask_type>& null_mask,
    size_type null_count,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cudf
