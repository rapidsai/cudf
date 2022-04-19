/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/udf/dstring.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cudf {
namespace strings {
//! Strings UDF support
namespace udf {

/**
 * @addtogroup strings_udfs
 * @{
 * @file
 * @brief Strings APIs for supporting user-defined functions
 */

/**
 * @brief Return a vector of cudf::string_view for the given strings column
 *
 * @param input Strings column
 * @param mr    Device memory resource used to allocate the returned vector
 * @return Device vector of cudf::string_view objects
 */
rmm::device_uvector<string_view> create_string_view_array(
  cudf::strings_column_view const input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Return an empty dstring array
 *
 * Once finished with the array call free_dstring_array to deallocate the dstring objects
 * before destroying the return memory buffer.
 *
 * @param size Number of empty dstring elements
 * @param mr   Device memory resource used to allocate the returned vector
 * @return Device buffer containing the empty dstring objects
 */
std::unique_ptr<rmm::device_buffer> create_dstring_array(
  size_type size, rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Return a cudf::column given an array of dstring objects
 *
 * @param input dstring array
 * @param mr    Device memory resource used to allocate the returned vector
 * @return A strings column copy of the dstring objects
 */
std::unique_ptr<cudf::column> make_strings_column(
  device_span<dstring const> input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Free all the dstring objects in the given array
 *
 * Call this to free the internal memory within individual dstring objects.
 * The input dstrings are modified (emptied) and can be reused.
 *
 * @param input dstring array
 */
void free_dstring_array(device_span<dstring> input);

/** @} */  // end of group
}  // namespace udf
}  // namespace strings
}  // namespace cudf
