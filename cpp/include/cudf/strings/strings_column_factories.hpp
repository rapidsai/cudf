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

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column.hpp>

namespace cudf {
/**---------------------------------------------------------------------------*
 * @brief Construct strings column given an array of pointer/size pairs.
 * Use the strings_column_handler class to perform strings operations on
 * this type of column.
 * 
 * @note `null_count()` and `null_bitmask` are determined if a pair contains
 * a null pointer. Otherwise, it is considered an empty string and not null.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if pointers are invalid
 *
 * @param[in] strs The pointer/size pair arrays.
 *                 Each pointer must be valid device memory address.
 *                 The size must be the number of bytes.
 * @param[in] count The number of elements in the strs array.
 * @param[in] stream Optional stream on which to issue all memory allocation and device
 * kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the column's `data` and `null_mask`.
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> make_strings_column(
    std::pair<const char*,size_t>* strs, size_type count,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
    
}  // namespace cudf
