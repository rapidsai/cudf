/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_case
 * @{
 */

/**
 * @brief Returns a column of capitalized strings.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * Example:
 * ```
 * input_string_tbl = ["tesT1", "a Test", "Another Test"];
 *
 * capt_string_tbl = ["Test1", "A test", "Another test"]
 * ```
 *
 * @param[in] strings String column.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Column of strings capitalized from the input column.
 */
std::unique_ptr<column> capitalize(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Modifies first character after spaces to uppercase and lower-cases the rest.
 *
 * Returns a column of strings where, for each string row in the input,
 * the first character after spaces is modified to upper-case,
 * while all the remaining characters in a word are modified to lower-case.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * Example:
 * ```
 * input_string_tbl = ["   tesT1", "a Test", " Another test "];
 *
 * titled_string_tbl = ["   Test1", "A Test", " Another Test "]
 * ```
 *
 * @param[in] strings String column.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return Column of title strings.
 */
std::unique_ptr<column> title(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
