/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
 * @param strings Strings instance for this operation.
 * @param pattern Regex pattern to match within each string.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return New table of strings columns.
 */
std::unique_ptr<table> findall_re(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
