/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf {
namespace strings {

struct regex_program;

/**
 * @addtogroup strings_contains
 * @{
 * @file
 */

/**
 * @brief Returns a lists column of strings for each matching occurrence using
 * the regex_program pattern within each string
 *
 * Each output row includes all the substrings within the corresponding input row
 * that match the given pattern. If no matches are found, the output row is empty.
 *
 * @code{.pseudo}
 * Example:
 * s = ["bunny", "rabbit", "hare", "dog"]
 * p = regex_program::create("[ab]")
 * r = findall(s, p)
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
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column of strings
 */
std::unique_ptr<column> findall(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
