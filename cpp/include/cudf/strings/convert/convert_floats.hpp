/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 * @file
 */

/**
 * @brief Returns a new numeric column by parsing float values from each string
 * in the provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Only characters [0-9] plus a prefix '-' and '+' and decimal '.' are recognized.
 * Additionally, scientific notation is also supported (e.g. "-1.78e+5").
 *
 * @throw cudf::logic_error if output_type is not float type.
 *
 * @param strings Strings instance for this operation
 * @param output_type Type of float numeric column to return
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with floats converted from strings
 */
std::unique_ptr<column> to_floats(
  strings_column_view const& strings,
  data_type output_type,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting the float values from the
 * provided column into strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * For each float, a string is created in base-10 decimal.
 * Negative numbers will include a '-' prefix.
 * Numbers producing more than 10 significant digits will produce a string that
 * includes scientific notation (e.g. "-1.78e+15").
 *
 * @throw cudf::logic_error if floats column is not float type.
 *
 * @param floats Numeric column to convert
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with floats as strings
 */
std::unique_ptr<column> from_floats(
  column_view const& floats,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to floats.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [-+0-9eE.].
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', 'A', '+7', '8.9' '3.7e+5']
 * b = s.is_float(s)
 * b is [true, true, false, false, true, true, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of boolean results for each string
 */
std::unique_ptr<column> is_float(
  strings_column_view const& input,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
