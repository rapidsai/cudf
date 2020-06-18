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

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <stdint.h>
#include <string.h>

namespace nvtext {

struct TokenizerResult {
  uint32_t nrows_tensor{};
  uint32_t* device_tensor_tokenIDS{};  // how are these freed?
  uint32_t* device_attention_mask{};
  uint32_t* device_tensor_metadata{};
};

/**
 * @brief Creates a full tokenizer that cleans the text and splits it into tokens.
 *
 * @param sentences The input sentences to tokenize.
 * @param filename_hashed_vocabulary A path to the preprocessed vocab.txt file.
 *               Note that this is the file AFTER python/perfect_hash.py has been used
 *               for preprocessing. Passing in the default vocab.txt file will cause
 *               undefined behavior.
 * @param max_sequence_length Limit the number of tokenIDs per row in final tensor with tokenIDS
 * @param stride Each row in tensor-tokenIDS will replicate (max_sequence_length - stride) tokenIDs
 *               from previous row, unless it is the first row of sentence/log.
 * @param do_lower_case If true, the tokenizer will convert uppercase characters in the
 *                      input stream to lower case AND strip accents from those characters.
 *                      If false, accented and uppercase characters are not transformed.
 * @param do_truncate If true, tokenizer will discard all the tokenIDs after max_sequence_length
 *                    for each input sentence/log. If false, it will use a new row in the
 *                    tensor-tokenIDS to continue generating the output.
 * @param max_num_sentences Maximum number of input sentences for instantiating the tokenizer.
 *                          Used to allocate memory on device.
 *                          If input contains larger number of sentences, behavior is undefined.
 * @param max_num_chars Maximum number of characters for instantiating the tokenizer.
 *                      Used to allocate memory on device.
 *                      If input contains larger number of characters, behavior is undefined.
 * @param max_rows_tensor Maximum number of rows in tensor_tokenIDS expected by tokenizer.
 *                        Used to allocate memory on device.
 *                        If output contains larger number of rows, behavior is undefined.
 * @param mr Memory resource to allocate any returned objects.
 * @return tokenIDS, mask, metadata
 */
std::unique_ptr<TokenizerResult> subword_tokenize(
  cudf::strings_column_view const& sentences,
  std::string const& filename_hashed_vocabulary,
  uint32_t max_sequence_length,
  uint32_t stride,
  bool do_lower,
  bool do_truncate,
  uint32_t max_num_sentences,
  uint32_t max_num_chars,
  uint32_t max_rows_tensor,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace nvtext
