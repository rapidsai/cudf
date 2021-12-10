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
 * @brief APIs for generating complement values of the input
 */

/**
 * @brief Generate an array containing complement of the values given from the input.
 *
 * For a number of given columns containing indices in the range of [0, @p size) that represent some
 * gather/scatter maps, generate an output column containing the indices in the same range which do
 * not appear in any of the given maps.
 *
 * Duplicates indices have no affect on the outcome. Invalid indices (i.e., the indices that are
 * outside of the given range [0, @p size)) are ignored during generating the output.
 *
 * @throws cudf::logic_error if any of the given @p maps column is not of integer types.
 * @throws cudf::logic_error if @p size is < 0.
 *
 * @param maps The columns containing input indices.
 * @param size Size that defines the range of indices in both input and output.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The result column containing the generated complement indices.
 */
rmm::device_uvector<size_type> complement(
  std::vector<device_span<size_type>> const& maps,
  size_type size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
