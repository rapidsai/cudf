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
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {

//! Strings column APIs
namespace strings {
/**
 * @addtogroup strings_apis
 * @{
 * @file strings/attributes.hpp
 * @brief Read attributes of strings column
 */

/**
 * @brief Returns a column containing character lengths
 * of each string in the given column
 *
 * The output column will have the same number of rows as the
 * specified strings column. Each row value will be the number of
 * characters in the corresponding string.
 *
 * Any null string will result in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with lengths for each string
 */
std::unique_ptr<column> count_characters(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column containing byte lengths
 * of each string in the given column
 *
 * The output column will have the same number of rows as the
 * specified strings column. Each row value will be the number of
 * bytes in the corresponding string.
 *
 * Any null string will result in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with the number of bytes for each string
 */
std::unique_ptr<column> count_bytes(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a numeric column with code point values (integers) for each
 * character of each string
 *
 * A code point is the integer value representation of a character.
 * For example, the code point value for the character 'A' in UTF-8 is 65.
 *
 * The size of the output column will be the total number of characters in the
 * strings column.
 *
 * Any null string is ignored. No null entries will appear in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New INT32 column with code point integer values for each character
 */
std::unique_ptr<column> code_points(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of strings_apis group

}  // namespace strings
}  // namespace CUDF_EXPORT cudf
