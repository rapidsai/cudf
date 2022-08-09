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

#include <cudf/column/column.hpp>
#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_contains
 * @{
 * @file strings/contains.hpp
 * @brief Strings APIs for regex contains, count, matches
 */

/**
 * @brief Returns a boolean column identifying rows which
 * match the given regex pattern.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc","123","def456"]
 * r = contains_re(s,"\\d+")
 * r is now [false, true, true]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param strings Strings instance for this operation.
 * @param pattern Regex pattern to match to each string.
 * @param flags Regex flags for interpreting special characters in the pattern.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> contains_re(
  strings_column_view const& strings,
  std::string_view pattern,
  regex_flags const flags             = regex_flags::DEFAULT,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a boolean column identifying rows which
 * matching the given regex pattern but only at the beginning the string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc","123","def456"]
 * r = matches_re(s,"\\d+")
 * r is now [false, true, false]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param strings Strings instance for this operation.
 * @param pattern Regex pattern to match to each string.
 * @param flags Regex flags for interpreting special characters in the pattern.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> matches_re(
  strings_column_view const& strings,
  std::string_view pattern,
  regex_flags const flags             = regex_flags::DEFAULT,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the number of times the given regex pattern
 * matches in each string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc","123","def45"]
 * r = count_re(s,"\\d")
 * r is now [0, 3, 2]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param strings Strings instance for this operation.
 * @param pattern Regex pattern to match within each string.
 * @param flags Regex flags for interpreting special characters in the pattern.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New INT32 column with counts for each string.
 */
std::unique_ptr<column> count_re(
  strings_column_view const& strings,
  std::string_view pattern,
  regex_flags const flags             = regex_flags::DEFAULT,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
