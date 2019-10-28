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
#include "scalar.hpp"

#include <rmm/thrust_rmm_allocator.h>

// TODO: make use of stream and memory resource
// TODO: update documentation

namespace cudf {
/**---------------------------------------------------------------------------*
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified numeric `data_type` and a null mask.
 * 
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param[in] type The desired numeric element type
 * @param[in] stream Optional stream on which to issue all memory allocation
 * and device kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the scalar's `data` and `is_valid` bool.
 *---------------------------------------------------------------------------**/
std::unique_ptr<scalar> make_numeric_scalar(
    data_type type,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Construct scalar with sufficient uninitialized storage
 * to hold `size` elements of the specified timestamp `data_type` with an optional
 * null mask.
 * 
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 *
 * @param[in] type The desired timestamp element type
 * @param[in] stream Optional stream on which to issue all memory allocation 
 * and device kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the scalar's `data` and `is_valid` bool.
 *---------------------------------------------------------------------------**/
std::unique_ptr<scalar> make_timestamp_scalar(
    data_type type,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Construct STRING type scalar given a `std::string`.
 * The size of the `std::string` must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char bytes.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param string The `std::string` to copy to device
 * @param stream Optional stream for use with all memory allocation
 *               and device kernels
 * @param mr Optional resource to use for device memory
 *           allocation of the scalar's `is_valid`.
 *---------------------------------------------------------------------------**/
std::unique_ptr<scalar> make_string_scalar(
    std::string const& string,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cudf
