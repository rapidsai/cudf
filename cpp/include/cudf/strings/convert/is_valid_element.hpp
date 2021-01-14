/*
 * Copyright (c) 2021, Baidu CORPORATION.
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

#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 * @file
 */

/**
 * @brief Returns a boolean column identifying strings in which all characters are valid.
 * 
 * Boolean variable `allow_decimal` indicates that whether we allow the input string data 
 * is decimal, if `allow_decimal` is false, this function will check that the format is 
 * [+-]?[0-9]+ like `is_integer`, or itll should check that it matches [+-]?[0-9]+(.[0-9]+)
 * similar to `is_float` but without some of the special cases for float (E, Inf, -Inf, NaN).
 * 
 * input_type is used to check whether the data overflows, for example, if input_type is 
 * `int8_t` and input string data is `128`, then it will return false ,because it out of ranges
 * [-128, 127] and overflows.
 * 
 * @param strings Strings instance for this operation.
 * @param allow_decimal identification whether we allow the element is decimal or not.
 * @param input_type input data type for check overflow.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_valid_element(
  strings_column_view const& strings,
  bool allow_decimal,
  data_type input_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf

