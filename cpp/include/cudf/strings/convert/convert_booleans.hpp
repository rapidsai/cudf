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
#include <cudf/scalar/scalar.hpp>
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
 * @brief Returns a new BOOL8 column by parsing boolean values from the strings
 * in the provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param input Strings instance for this operation
 * @param true_string String to expect for true. Non-matching strings are false
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column converted from strings
 */
std::unique_ptr<column> to_booleans(
  strings_column_view const& input,
  string_scalar const& true_string,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting the boolean values from the
 * provided column into strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @throw cudf::logic_error if the input column is not BOOL8 type.
 *
 * @param booleans Boolean column to convert
 * @param true_string String to use for true in the output column
 * @param false_string String to use for false in the output column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> from_booleans(
  column_view const& booleans,
  string_scalar const& true_string,
  string_scalar const& false_string,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
