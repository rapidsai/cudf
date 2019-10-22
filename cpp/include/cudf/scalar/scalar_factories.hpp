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
 * @brief Construct scalar with sufficient uninitialized storage
 * to hold `size` elements of the specified numeric `data_type` with an optional
 * null mask.
 * 
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param[in] type The desired numeric element type
 * @param[in] size The number of elements in the scalar
 * @param[in] state Optional, controls allocation/initialization of the
 * scalar's null mask. By default, no null mask is allocated.
 * @param[in] stream Optional stream on which to issue all memory allocation and device
 * kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the scalar's `data` and `null_mask`.
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
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 *
 * @param[in] type The desired timestamp element type
 * @param[in] size The number of elements in the scalar
 * @param[in] state Optional, controls allocation/initialization of the
 * scalar's null mask. By default, no null mask is allocated.
 * @param[in] stream Optional stream on which to issue all memory allocation and device
 * kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the scalar's `data` and `null_mask`.
 *---------------------------------------------------------------------------**/
std::unique_ptr<scalar> make_timestamp_scalar(
    data_type type,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

