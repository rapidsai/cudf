/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace cudf::strings {

/**
 * @addtogroup strings_regex
 * @{
 */

/**
 * @brief Compute the working memory size for evaluating a regex pattern
 * on a given strings column.
 *
 * This function returns the size in bytes of the memory needed to evaluate
 * the given regex pattern in parallel over the returned output rows.
 * The number of output rows will be less than or equal to the size of the
 * input column.
 *
 * This function computes only the state data memory size required to process
 * a regex pattern over the output row count.
 * Specific functions that use regex may require additional working memory
 * unrelated to the regex processing.
 *
 * @param input Strings instance
 * @param pattern Regex pattern to be used
 * @param flags Regex flags for interpreting special characters in the pattern
 * @return Size of the state memory in bytes required for processing `pattern` on `strings`
 *         and the number of concurrent rows this memory will support
 */
std::pair<std::size_t, size_type> compute_regex_state_memory(
  strings_column_view const& input,
  std::string_view pattern,
  regex_flags const flags = regex_flags::DEFAULT);

/** @} */  // end of doxygen group

}  // namespace cudf::strings
