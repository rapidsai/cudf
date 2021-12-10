/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

namespace cudf {
/**
 * @addtogroup complement_generation
 * @{
 * @file
 * @brief APIs for generating complement values of the input.
 */

/**
 * @brief Generate an array containing complement of the values given in the input.
 *
 * For a given input array of integer values, generate an output array containing the values in the
 * range of [0, size) such that they do not appear in the input.
 *
 * The input array is allowed to have duplicates values, which will not affect the result.
 * In addition, values that are outside of the given range [0, @p size) will be ignored from the
 * operation.
 *
 * @throws cudf::logic_error if @p size is < 0.
 *
 * @param input The input values to find complement.
 * @param size Size that defines the range for output values.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Array containing the generated complement values.
 */
rmm::device_uvector<size_type> complement(
  device_span<size_type const> const& input,
  size_type size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
