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

#include "text/subword/detail/data_normalizer.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace nvtext {

struct hashed_vocabulary;

namespace detail {

/**
 * @brief This splits words into tokens contained in the model vocabulary file.
 *
 * The tokenizer first normalizes the character bytes, identifies the words in
 * each string, and then converts each word in to a integer token-id per the
 * provided vocabulary hash table.
 *
 * The `tokenize()` function produces two device vectors `uvector_pair`.
 * The first is the token-ids for each word identified in the input strings.
 * The second is the offsets to identify which ids go with each string.
 *
 * Temporary buffers are created equal to 3 uint32 values plus 1 byte per input byte.
 * Also the normalize step allocates an additional 16x bytes per input byte but 8x
 * of this memory is reused by the `tokenize()` function.
 * This means 13x + 8x = 21x the number bytes of the input strings buffer must be
 * available to call the `tokenize()` function in this class.
 */
class wordpiece_tokenizer {
 public:
  /**
   * @brief Creates a full tokenizer that cleans the text and splits it into tokens.
   *
   * @param vocab_table The preprocessed hashed vocabulary data.
   * @param max_sequence_length Limit the number of token-ids per row in the output
   * @param stride Each row in tensor-token-ids will replicate `max_sequence_length - stride`
   *        token-ids from the previous row, unless it is the first string.
   * @param do_truncate If true, the tokenizer will discard all the token-ids after
   *        `max_sequence_length` for each input string. If false, it will use a
   *        new row in the tensor-token-ids to continue generating the output.
   * @param do_lower_case If true, the tokenizer will convert uppercase characters in the
   *        input stream to lowercase and strip accents from those characters.
   *        If false, accented and uppercase characters are not transformed.
   * @param max_word_length The length of the longest word that will be tokenized. Words
   *        longer than this will simply be replaced by the unknown token
   *        specified in the `vocab_file`.
   */
  wordpiece_tokenizer(hashed_vocabulary const& vocab_table,
                      uint32_t max_sequence_length,
                      uint32_t stride,
                      bool do_truncate,
                      bool do_lower_case,
                      uint32_t max_word_length = 200);

  /**
   * @brief Splits the input text into token ids.
   *
   * This class is simply a wrapper around the basic and word piece tokenizers.
   *
   * @param input Strings to tokenize
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Pointer to token-ids and token-id offsets
   */
  uvector_pair tokenize(cudf::strings_column_view const& input, rmm::cuda_stream_view stream);

 private:
  /**
   * @brief Splits the code points from the normalizer into tokens.
   *
   * @param[in,out] cps_and_offsets The output code points and offsets
   *        from the normalizer.
   *        The data is modified to contain the token ids and token counts
   *        per string.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void tokenize(uvector_pair& cps_and_offsets, rmm::cuda_stream_view stream);

  hashed_vocabulary const& vocab_table;
  data_normalizer normalizer;  // removes punctuation, accents, etc
  uint32_t const max_sequence_length;
  uint32_t const stride;
  bool const do_truncate;
  uint32_t const max_word_length;
};

}  // namespace detail
}  // namespace nvtext
