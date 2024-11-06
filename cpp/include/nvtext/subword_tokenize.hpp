/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT nvtext {

/**
 * @addtogroup nvtext_tokenize
 * @{
 * @file
 */

/**
 * @brief The vocabulary data for use with the subword_tokenize function.
 */
struct hashed_vocabulary {
  uint16_t first_token_id{};            ///< The first token id in the vocabulary
  uint16_t separator_token_id{};        ///< The separator token id in the vocabulary
  uint16_t unknown_token_id{};          ///< The unknown token id in the vocabulary
  uint32_t outer_hash_a{};              ///< The a parameter for the outer hash
  uint32_t outer_hash_b{};              ///< The b parameter for the outer hash
  uint16_t num_bins{};                  ///< Number of bins
  std::unique_ptr<cudf::column> table;  ///< uint64 column, the flattened hash table with key, value
                                        ///< pairs packed in 64-bits
  std::unique_ptr<cudf::column> bin_coefficients;  ///< uint64 column, containing the hashing
                                                   ///< parameters for each hash bin on the GPU
  std::unique_ptr<cudf::column> bin_offsets;  ///< uint16 column, containing the start index of each
                                              ///< bin in the flattened hash table
  std::unique_ptr<cudf::column>
    cp_metadata;  ///< uint32 column, The code point metadata table to use for normalization
  std::unique_ptr<cudf::column>
    aux_cp_table;  ///< uint64 column, The auxiliary code point table to use for normalization
};

/**
 * @brief Load the hashed vocabulary file into device memory.
 *
 * The object here can be used to call the subword_tokenize without
 * incurring the cost of loading the same file each time.
 *
 * @throw cudf::logic_error if the `filename_hashed_vocabulary` could not be opened.
 *
 * @param filename_hashed_vocabulary A path to the preprocessed vocab.txt file.
 *        Note that this is the file AFTER python/perfect_hash.py has been used
 *        for preprocessing.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resource to allocate any returned objects.
 * @return vocabulary hash-table elements
 */
std::unique_ptr<hashed_vocabulary> load_vocabulary_file(
  std::string const& filename_hashed_vocabulary,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Result object for the subword_tokenize functions.
 */
struct tokenizer_result {
  /**
   * @brief The number of rows for the output token-ids.
   */
  uint32_t nrows_tensor{};
  /**
   * @brief The number of token-ids in each row.
   */
  uint32_t sequence_length{};
  /**
   * @brief A vector of token-ids for each row.
   *
   * The data is a flat matrix (nrows_tensor x sequence_length) of token-ids.
   * This column is of type UINT32 with no null entries.
   */
  std::unique_ptr<cudf::column> tensor_token_ids;
  /**
   * @brief This mask identifies which tensor-token-ids are valid.
   *
   * This column is of type UINT32 with no null entries.
   */
  std::unique_ptr<cudf::column> tensor_attention_mask;
  /**
   * @brief The metadata for each tensor row.
   *
   * There are three elements per tensor row [row-id, start_pos, stop_pos])
   * This column is of type UINT32 with no null entries.
   */
  std::unique_ptr<cudf::column> tensor_metadata;
};

/**
 * @brief Creates a tokenizer that cleans the text, splits it into tokens and
 *        returns token-ids from an input vocabulary.
 *
 * The strings are first normalized by converting to lower-case, removing
 * punctuation, replacing a select set of multi-byte characters and
 * whitespace characters.
 *
 * The strings are then tokenized by using whitespace as a delimiter.
 * Consecutive delimiters are ignored. Each token is then assigned
 * a 4-byte token-id mapped from the provided vocabulary table.
 *
 * Essentially each string is converted into one or more vectors of token-ids
 * in the output column. The total number of these vectors times `max_sequence_length`
 * is the size of the `tensor_token_ids` output column. For `do_truncate==true`:
 * ```
 * size of tensor_token_ids = max_sequence_length * strings.size()
 * size of tensor_attention_mask = max_sequence_length * strings.size()
 * size of tensor_metadata = 3 * strings.size()
 * ```
 *
 * For `do_truncate==false` the number of rows per output string depends on the
 * number of tokens resolved and the `stride` value which may repeat tokens
 * in subsequent overflow rows.
 *
 * This function requires about 21x the number of character bytes in the input
 * strings column as working memory.
 *
 * @throw cudf::logic_error if `stride > max_sequence_length`
 * @throw std::overflow_error if `max_sequence_length * max_rows_tensor`
 *        exceeds the column size limit
 *
 * @param strings The input strings to tokenize.
 * @param vocabulary_table The vocabulary table pre-loaded into this object.
 * @param max_sequence_length Limit of the number of token-ids per row in final tensor
 *        for each string.
 * @param stride Each row in the output token-ids will replicate `max_sequence_length - stride`
 *        the token-ids from the previous row, unless it is the first string.
 * @param do_lower_case If true, the tokenizer will convert uppercase characters in the
 *        input stream to lower-case and strip accents from those characters.
 *        If false, accented and uppercase characters are not transformed.
 * @param do_truncate If true, the tokenizer will discard all the token-ids after
 *        `max_sequence_length` for each input string. If false, it will use a new row
 *        in the output token-ids to continue generating the output.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resource to allocate any returned objects.
 * @return token-ids, attention-mask, and metadata
 */
tokenizer_result subword_tokenize(
  cudf::strings_column_view const& strings,
  hashed_vocabulary const& vocabulary_table,
  uint32_t max_sequence_length,
  uint32_t stride,
  bool do_lower_case,
  bool do_truncate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
