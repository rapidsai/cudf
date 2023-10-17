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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <optional>

namespace cudf {
namespace strings {

struct regex_program;

/**
 * @addtogroup strings_replace
 * @{
 * @file
 */

/**
 * @brief For each string, replaces any character sequence matching the given regex
 * with the provided replacement string.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param replacement The string used to replace the matched sequence in each string.
 *        Default is an empty string.
 * @param max_replace_count The maximum number of times to replace the matched pattern
 *        within each string. Default replaces every substring that is matched.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace_re(
  strings_column_view const& input,
  regex_program const& prog,
  string_scalar const& replacement           = string_scalar(""),
  std::optional<size_type> max_replace_count = std::nullopt,
  rmm::cuda_stream_view stream               = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr        = rmm::mr::get_current_device_resource());

/**
 * @brief For each string, replaces any character sequence matching the given patterns
 * with the corresponding string in the `replacements` column.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param patterns The regular expression patterns to search within each string
 * @param replacements The strings used for replacement
 * @param flags Regex flags for interpreting special characters in the patterns
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace_re(
  strings_column_view const& input,
  std::vector<std::string> const& patterns,
  strings_column_view const& replacements,
  regex_flags const flags             = regex_flags::DEFAULT,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief For each string, replaces any character sequence matching the given regex
 * using the replacement template for back-references.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @throw cudf::logic_error if capture index values in `replacement` are not in range 0-99, and also
 * if the index exceeds the group count specified in the pattern
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param replacement The replacement template for creating the output string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace_with_backrefs(
  strings_column_view const& input,
  regex_program const& prog,
  std::string_view replacement,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace strings
}  // namespace cudf
