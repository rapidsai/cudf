/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>

namespace cudf {
namespace strings {
namespace udf {

class udf_string;

/**
 * @brief Return a cudf::string_view array for the given strings column
 *
 * No string data is copied so the input column controls the lifetime of the
 * underlying strings.
 *
 * New device memory is allocated and returned to hold just the string_view instances.
 *
 * @param input Strings column to convert to a string_view array.
 * @return Array of string_view objects in device memory
 */
std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input);

/**
 * @brief Return a STRINGS column given an array of udf_string objects
 *
 * This will make a copy of the strings in d_string in order to build
 * the output column.
 * The individual udf_strings are also cleared freeing each of their internal
 * device memory buffers.
 *
 * @param d_strings Pointer to device memory of udf_string objects
 * @param size The number of elements in the d_strings array
 * @return A strings column copy of the udf_string objects
 */
std::unique_ptr<cudf::column> column_from_udf_string_array(udf_string* d_strings,
                                                           cudf::size_type size);

/**
 * @brief Frees a vector of udf_string objects
 *
 * The individual udf_strings are cleared freeing each of their internal
 * device memory buffers.
 *
 * @param d_strings Pointer to device memory of udf_string objects
 * @param size The number of elements in the d_strings array
 */
void free_udf_string_array(udf_string* d_strings, cudf::size_type size);

}  // namespace udf
}  // namespace strings
}  // namespace cudf
