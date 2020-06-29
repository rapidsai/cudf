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

#include <text/subword/detail/data_normalizer.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>
#include <vector>

namespace nvtext {

struct hashed_vocabulary;

namespace detail {

/**
 * @brief This splits words into tokens contained in the model vocabulary file.
 */
class wordpiece_tokenizer {
 public:
  /**
   * @brief Creates a full tokenizer that cleans the text and splits it into tokens.
   *
   * @param vocab_table The preprocessed hashed vocabulary data.
   * @param max_num_strings Maximum number of input strings for instantiating the tokenizer.
   *        Used to allocate temporary working memory on the GPU.
   *        If the input contains a larger number of strings, behavior is undefined.
   * @param max_num_chars Maximum number of characters for instantiating the tokenizer.
   *        Used to allocate temporary working memory on the GPU.
   *        If the input contains a larger number of characters, behavior is undefined.
   * @param max_rows_final_tensor Maximum number of rows in tensor_token-ids expected by tokenizer.
   *        Used to allocate temporary working memory on the GPU.
   *        If the output contains a larger number of rows, behavior is undefined.
   * @param max_sequence_length Limit the number of token-ids per row in the output
   * @param stride Each row in tensor-token-ids will replicate `max_sequence_length - stride`
   *        token-ids from the previous row, unless it is the first string.
   * @param do_truncate If true, the tokenizer will discard all the token-ids after
   *        `max_sequence_length` for each input string. If false, it will use a
   *        new row in the tensor-token-ids to continue generating the output.
   * @param do_lower_case If true, the tokenizer will convert uppercase characters in the
   *        input stream to lowercase and strip accents from those characters.
   *        If false, accented and uppercase characters are not transformed.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param max_word_length The length of the longest word that will be tokenized. Words
   *        longer than this will simply be replaced by the unknown token
   *        specified in the `vocab_file`.
   */
  wordpiece_tokenizer(hashed_vocabulary const& vocab_table,
                      uint32_t max_num_strings,
                      uint32_t max_num_chars,
                      uint32_t max_rows_final_tensor,
                      uint32_t max_sequence_length,
                      uint32_t stride,
                      bool do_truncate,
                      bool do_lower_case,
                      cudaStream_t stream      = 0,
                      uint32_t max_word_length = 200);

  /**
   * @brief Splits the input text into token ids.
   *
   * This class is simply a wrapper around the basic and word piece tokenizers.
   *
   * @param d_strings A vector of strings which MUST be encoded in the utf8 format.
   * @param d_offsets A vector of byte offsets to the beginning of individual strings in
   *        the `d_strings` parameter.
   * @param num_strings The number of strings in `d_strings`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @return Pointer to token-ids and token-id offsets
   */
  std::pair<uint32_t*, uint32_t*> tokenize(char const* d_strings,
                                           uint32_t const* d_offsets,
                                           uint32_t num_strings,
                                           cudaStream_t stream);

 private:
  /**
   * @brief Splits code points run through the normalizer into tokens.
   *
   * @param cps_and_length[in,out] A gpu pointer to the codepoints.
   *        The data is modified to contain the token ids of each
   *        space separated sequence of code points.
   * @param offsets_and_length[in,out] A gpu pointer to the string offsets.
   *        This is always `num strings + 1`.
   *        The data is modified to contain string lengths in terms
   *        of token ids instead of code points.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void tokenize(ptr_length_pair& cp_and_length,
                ptr_length_pair& offsets_and_length,
                cudaStream_t stream);

  data_normalizer normalizer;  // removes punctuation, accents, etc
  uint32_t const max_sequence_length;
  uint32_t const stride;
  bool const do_truncate;
  uint32_t const max_word_length;

  // hash table parameters
  hashed_vocabulary const& vocab_table;

  // working memory for tokenizing
  rmm::device_uvector<uint32_t> device_token_ids;
  rmm::device_uvector<uint32_t> device_word_indices;
  rmm::device_uvector<uint8_t> device_tokens_per_word;
  rmm::device_uvector<uint32_t> device_num_selected;
  rmm::device_uvector<size_t> cub_temp_storage;
  size_t max_cub_storage_bytes;
};

}  // namespace detail
}  // namespace nvtext
