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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace nvtext {

/**
 * @addtogroup nvtext_tokenize
 * @{
 * @file
 */

/**
 * @brief The table of merge pairs for the BPE encoder.
 *
 * To create an instance, call nvtext::load_merges_table
 */
struct bpe_merge_pairs {
  std::unique_ptr<cudf::column> merge_pairs;  // strings
};

/**
 * @brief Create a nvtext::bpe_merge_pairs from an input file.
 *
 * The file should contain a pair of strings per line separated by
 * a single space.
 *
 * Example:
 * @code{.txt}
 * e n
 * i t
 * i s
 * ...
 * @endcode
 *
 * The pairs are expected to be ordered in the file by their rank
 * relative to each other. A pair will be is in priority over
 * any pairs below it.
 *
 * @param filename_merges Local file path of pairs encoded in UTF-8
 * @param mr Memory resource to allocate any returned objects.
 */
std::unique_ptr<bpe_merge_pairs> load_merge_pairs_file(
  std::string const& filename_merges,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Byte pair encode the input strings.
 *
 * This will split each string on whitespace, perform the encoding,
 * and then build the output using the given `separator`.
 *
 * The encoding algorithm rebuilds each string by matching substrings
 * in the `merge_pairs` table and iteratively removing the minimum ranked pair
 * until no pairs are left. Then, a space is inserted between the remaining
 * pairs before the result is joined to make the output string.
 *
 * @code{.pseudo}
 * mps = load_merges_file("merges.txt")
 * input = ["test sentence", "this is test"]
 * result = byte_pair_encoding(input, mps)
 * result is now ["test Ġsent tence", "this Ġis Ġtest"]
 * @endcode
 *
 * @throw cudf::logic_error if `merge_pairs` is empty
 * @throw cudf::logic_error if `separator` is invalid
 *
 * @param input Strings to encode.
 * @param merge_pairs Created by a call to nvtext::load_merges_file.
 * @param separator String used to build the output after encoding.
 *                  Default is a space followed by `Ġ`.
 * @param mr Memory resource to allocate any returned objects.
 */
std::unique_ptr<cudf::column> byte_pair_encoding(
  cudf::strings_column_view const& input,
  bpe_merge_pairs const& merges_pairs,
  cudf::string_scalar const& separator = cudf::string_scalar(" Ġ"),
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
