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

//! NVText APIs
namespace nvtext {
/**
 * @addtogroup nvtext_normalize
 * @{
 * @file
 */

/**
 * @brief Returns a new strings column by normalizing the whitespace in each
 * string in the input column.
 *
 * Normalizing a string replaces any number of whitespace character
 * (character code-point <= ' ') runs with a single space ' ' and
 * trims whitespace from the beginning and end of the string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a b", "  c  d\n", "e \t f "]
 * t = normalize_spaces(s)
 * t is now ["a b","c d","e f"]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * @param strings Strings column to normalize.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of normalized strings.
 */
std::unique_ptr<cudf::column> normalize_spaces(
  cudf::strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Normalizes strings characters for tokenizing.
 *
 * This uses the normalizer that is built into the nvtext::subword_tokenize function
 * which includes:
 *
 * - adding padding around punctuation (unicode category starts with "P")
 *   as well as certain ASCII symbols like "^" and "$"
 * - adding padding around the [CJK Unicode block
 * characters](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block))
 * - changing whitespace (e.g. `"\t", "\n", "\r"`) to just space `" "`
 * - removing control characters (unicode categories "Cc" and "Cf")
 *
 * The padding process here adds a single space before and after the character.
 * Details on _unicode category_ can be found here:
 * https://unicodebook.readthedocs.io/unicode.html#categories
 *
 * If `do_lower_case = true`, lower-casing also removes the accents. The
 * accents cannot be removed from upper-case characters without lower-casing
 * and lower-casing cannot be performed without also removing accents.
 * However, if the accented character is already lower-case, then only the
 * accent is removed.
 *
 * @code{.pseudo}
 * s = ["éâîô\teaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]"]
 * s1 = normalize_characters(s,true)
 * s1 is now ["eaio eaio", "acenu", "acenu", " $ 24 . 08", " [ a , bb ] "]
 * s2 = normalize_characters(s,false)
 * s2 is now ["éâîô eaio", "ĂĆĖÑÜ", "ACENU", " $ 24 . 08", " [ a , bb ] "]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * This function requires about 16x the number of character bytes in the input
 * strings column as working memory.
 *
 * @param strings The input strings to normalize.
 * @param do_lower_case If true, upper-case characters are converted to
 *        lower-case and accents are stripped from those characters.
 *        If false, accented and upper-case characters are not transformed.
 * @param mr Memory resource to allocate any returned objects.
 * @return Normalized strings column
 */
std::unique_ptr<cudf::column> normalize_characters(
  cudf::strings_column_view const& strings,
  bool do_lower_case,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
