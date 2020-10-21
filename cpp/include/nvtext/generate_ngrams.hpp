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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace nvtext {
/**
 * @addtogroup nvtext_ngrams
 * @{
 * @file
 */

/**
 * @brief Returns a single column of strings by generating ngrams from
 * a strings column.
 *
 * An ngram is a grouping of 2 or more strings with a separator. For example,
 * generating bigrams groups all adjacent pairs of strings.
 *
 * ```
 * ["a", "bb", "ccc"] would generate bigrams as ["a_bb", "bb_ccc"]
 * and trigrams as ["a_bb_ccc"]
 * ```
 *
 * The size of the output column will be the total number of ngrams generated from
 * the input strings column.
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @throw cudf::logic_error if `ngrams < 2`
 * @throw cudf::logic_error if `separator` is invalid
 * @throw cudf::logic_error if there are not enough strings to generate any ngrams
 *
 * @param strings Strings column to tokenize and produce ngrams from.
 * @param ngrams The ngram number to generate.
 *               Default is 2 = bigram.
 * @param separator The string to use for separating ngram tokens.
 *                  Default is "_" character.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> generate_ngrams(
  cudf::strings_column_view const& strings,
  cudf::size_type ngrams               = 2,
  cudf::string_scalar const& separator = cudf::string_scalar{"_"},
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @brief Generates ngrams of characters within each string.
 *
 * Each character of a string used to build ngrams.
 * Ngrams are not created across strings.
 *
 * ```
 * ["ab", "cde", "fgh"] would generate bigrams as ["ab", "cd", "de", "fg", "gh"]
 * ```
 *
 * The size of the output column will be the total number of ngrams generated from
 * the input strings column.
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @throw cudf::logic_error if `ngrams < 2`
 * @throw cudf::logic_error if there are not enough characters to generate any ngrams
 *
 * @param strings Strings column to produce ngrams from.
 * @param ngrams The ngram number to generate.
 *               Default is 2 = bigram.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> generate_character_ngrams(
  cudf::strings_column_view const& strings,
  cudf::size_type ngrams              = 2,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
