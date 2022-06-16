/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_contains
 * @{
 * @file
 */

/**
 * @brief Returns a table of strings columns for each matching occurrence of the
 * regex pattern within each string.
 *
 * The number of output columns is determined by the string with the most
 * matches.
 *
 * @code{.pseudo}
 * Example:
 * s = ["bunny","rabbit"]
 * r = findall(s, "[ab]"")
 * r is now a table of 3 columns:
 *   ["b","a"]
 *   [null,"b"]
 *   [null,"b"]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation.
 * @param pattern Regex pattern to match within each string.
 * @param flags Regex flags for interpreting special characters in the pattern.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return New table of strings columns.
 */
std::unique_ptr<table> findall(
  strings_column_view const& input,
  std::string_view pattern,
  regex_flags const flags             = regex_flags::DEFAULT,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a lists column of strings for each matching occurrence of the
 * regex pattern within each string.
 *
 * Each output row includes all the substrings within the corresponding input row
 * that match the given pattern. If no matches are found, the output row is empty.
 *
 * @code{.pseudo}
 * Example:
 * s = ["bunny", "rabbit", "hare", "dog"]
 * r = findall_record(s, "[ab]")
 * r is now a lists column like:
 *  [ ["b"]
 *    ["a","b","b"]
 *    ["a"]
 *    [] ]
 * @endcode
 *
 * A null output row occurs if the corresponding input row is null.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation.
 * @param pattern Regex pattern to match within each string.
 * @param flags Regex flags for interpreting special characters in the pattern.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New lists column of strings.
 */
std::unique_ptr<column> findall_record(
  strings_column_view const& input,
  std::string_view pattern,
  regex_flags const flags             = regex_flags::DEFAULT,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
