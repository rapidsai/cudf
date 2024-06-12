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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_case
 * @{
 * @file
 */

/**
 * @brief Converts a column of strings to lower case.
 *
 * Only upper case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> to_lower(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Converts a column of strings to upper case.
 *
 * Only lower case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> to_upper(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of strings converting lower case characters to
 * upper case and vice versa.
 *
 * Only upper or lower case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> swapcase(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
