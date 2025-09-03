/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 * @file
 */

/**
 * @brief Configures whether casting also byte swaps
 */
enum class endian : bool { BIG, LITTLE };

/**
 * @brief Returns a new integer numeric column encoded to represent the
 * strings in the input column
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * If the input column contains only strings less than or equal to 8 bytes,
 * the column can be converted to an appropriate integer type INT8, INT16, INT32, or INT64.
 * This is useful in libcudf functions that only require relational algebra operations
 * like groupby, join, or sort.
 *
 * @code{.pseudo}
 * Example:
 * s = ['a', 'b', '', 'c', 'd']
 * b = cast_to_integer(s)
 * b is [97, 98, 0, 99, 100]
 * @endcode
 *
 * Multi-byte UTF-8 characters are encoded as multiple bytes in the integer result.
 *
 * If the input column contains strings greater than the number of bytes supported
 * by the output type, the result is undefined.
 *
 * If only equals logic is needed, use `swap==BIG` is sufficient. Otherwise
 * use `swap==LITTLE` to ensure comparison operations work correctly.
 *
 * @throw cudf::logic_error if output_type is not integral type.
 *
 * @param input Strings instance for this operation
 * @param output_type Type of integer numeric column to return
 * @param swap Whether to swap bytes on the output. Default is to swap.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with encoded integers converted from strings
 */
std::unique_ptr<column> cast_to_integer(
  strings_column_view const& input,
  data_type output_type,
  endian swap                       = endian::LITTLE,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting the encoded integer values from the
 * provided column into strings.
 *
 * This performs the inverse of `cast_to_integer`. The individual bytes of the integer
 * are reconverted into UTF-8 strings.
 *
 * @code{.pseudo}
 * Example:
 * b is [97, 98, 0, 99, 100]
 * s = cast_from_integer(b)
 * s is ['a', 'b', '', 'c', 'd']
 * @endcode *
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Ensure the `swap` parameter matches is the same value used in the `cast_to_integer` API.
 *
 * @throw cudf::logic_error if integers column is not integral type.
 *
 * @param integers Encoded integer column to convert
 * @param swap Whether the input is stored as big or little endian
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with integers as strings
 */
std::unique_ptr<column> cast_from_integer(
  column_view const& integers,
  endian swap                       = endian::LITTLE,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minimum integer type required to encode the input column.
 *
 * Use this type to with `cast_to_integer` to encode the input column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The minimum integer type required to encode the input column, or std::nullopt
 * if the input column is not castable to an integer.
 */
std::optional<cudf::data_type> integer_cast_type(
  strings_column_view const& input, rmm::cuda_stream_view stream = cudf::get_default_stream());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
