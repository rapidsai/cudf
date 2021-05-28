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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_copy
 * @{
 * @file strings/copy.hpp
 * @brief Strings APIs for copying
 */

/**
 * @brief Row-wise concatenates the given list of strings columns and
 * returns a single strings column result.
 *
 * @code{.pseudo}
 * Example:
 * s1 = ['aa', null, '', 'dd']
 * out = concatenate({s1, s2})
 * out is ['aa', null, 'cc', null]
 * @endcode
 *
 * @throw cudf::logic_error if input columns are not all strings columns.
 * @throw cudf::logic_error if separator is not valid.
 * @throw cudf::logic_error if only one column is specified
 *
 * @param strings_columns List of string columns to concatenate.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> repeat(
  string_scalar const& string,
  size_type repeat_times,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief
 *
 * @code{.pseudo}
 * Example:
 * c0   = ['aa', null, '',  'ee',  null, 'ff']
 * out = concatenate({c0, c1, c2}, sep)
 * out is [null, null, null, null, null, null]
 *
 * @endcode
 *
 * @throw cudf::logic_error if no input columns are specified - table view is empty
 * @throw cudf::logic_error if input columns are not all strings columns.
 * @throw cudf::logic_error if the number of rows from @p separators and @p strings_columns
 *                          do not match
 *
 * @param strings_columns List of strings columns to concatenate.
 * @param mr Resource for allocating device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> repeat(
  strings_column_view const& strings_column,
  size_type repeat_times,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
