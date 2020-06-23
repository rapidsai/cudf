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
#include <vector>

namespace nvtext {
namespace detail {

#if 0
struct ptr_length_pair {
  uint32_t* gpu_ptr{};
  size_t length{};
};

/**
 * @brief This splits words into tokens contained in the model vocabulary file.
 */
class word_piece_tokenizer {
 public:
  /**
   * @brief Creates a tokenizer by loading the preprocessed vocabulary from the given file path.
   *
   * @param vocab_file A path to the preprocessed vocabulary text file.
   *        Note that this is the file AFTER python/perfect_hash.py has been used
   *        for preprocessing. Passing in the default vocab.txt file will cause
   *        undefined behavior.
   * @param max_num_chars Maximum number of characters for instantiating the tokenizer.
   *        Used to allocate temporary working memory on the GPU.
   *        If the input contains a larger number of characters, behavior
   *        is undefined.
   * @param max_inp_chars_per_word The length of the longest word that will be tokenized. Words
   *        longer than this will simply be replaced by the unknown token
   *        unk_token which was specified in the python script
   *        python/perfect_hash.py
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  word_piece_tokenizer(std::string const& vocab_file,
                       uint32_t max_num_chars,
                       uint32_t max_inp_chars_per_word = 200,
                       cudaStream_t stream             = 0);

  /**
   * @brief Splits code points run through the basic tokenizer into tokens.
   *
   * @param cps_and_length[in,out] A gpu pointer to the codepoints after being run through the basic
   *        tokenizer along with the number of code points in the length
   *        field. The data is modified to contain the token ids of each
   *        space separated sequence of code points.
   * @param offsets_and_length[in,out] A gpu pointer to the sentence offsets along with the length
   *        of the underlying array. This is always `num sentences + 1`.
   *        The data is modified to contain sentence lengths in terms
   *        of token ids instead of code points.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void tokenize(ptr_length_pair& cp_and_length,
                ptr_length_pair& offsets_and_length,
                cudaStream_t stream);

 private:
  // Hash parameters
  uint32_t outer_hash_a_param;
  uint32_t outer_hash_b_param;
  uint16_t num_outer_bins;

  uint16_t unk_token_id;
  uint16_t first_tok_id;
  uint16_t sep_tok_id;
  unsigned int max_word_length;

  // pointers to device data needed for tokenization
  rmm::device_vector<uint64_t> device_hash_table;
  rmm::device_vector<uint64_t> device_bin_coefficients;
  rmm::device_vector<uint16_t> device_bin_offsets;

  rmm::device_vector<uint32_t> device_token_ids{};
  rmm::device_vector<uint32_t> device_word_indices;
  rmm::device_vector<uint8_t> device_tokens_per_word;

  rmm::device_vector<size_t> cub_temp_storage;
  rmm::device_vector<uint32_t> device_num_selected;
  size_t max_cub_storage_bytes;
};
#endif

class full_tokenizer {
 public:
  /**
   * @brief Creates a full tokenizer that cleans the text and splits it into tokens.
   *
   * @param vocab_file A path to the preprocessed hashed vocabulary file.
   *        Note that this is the file AFTER python/perfect_hash.py has been used
   *        for preprocessing. Passing in the default vocab.txt file will cause
   *        undefined behavior.
   * @param max_num_sentences Maximum number of input sentences for instantiating the tokenizer.
   *        Used to allocate temporary working memory on the GPU.
   *        If the input contains a larger number of sentences, behavior is undefined.
   * @param max_num_chars Maximum number of characters for instantiating the tokenizer.
   *        Used to allocate temporary working memory on the GPU.
   *        If the input contains a larger number of characters, behavior is undefined.
   * @param max_rows_final_tensor Maximum number of rows in tensor_tokenIDS expected by tokenizer.
   *        Used to allocate temporary working memory on the GPU.
   *        If the output contains a larger number of rows, behavior is undefined.
   * @param max_sequence_length Limit the number of tokenIDs per row in final tensor with tokenIDS
   * @param stride Each row in tensor-tokenIDS will replicate `max_sequence_length - stride`
   *        tokenIDs from the previous row, unless it is the first row of sentence/log.
   * @param do_truncate If true, the tokenizer will discard all the tokenIDs after
   *        `max_sequence_length` for each input sentence/log. If false, it will use a
   *        new row in the tensor-tokenIDS to continue generating the output.
   * @param do_lower_case If true, the tokenizer will convert uppercase characters in the
   *        input stream to lower case AND strip accents from those characters.
   *        If false, accented and uppercase characters are not transformed.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param max_inp_chars_per_word The length of the longest word that will be tokenized. Words
   *        longer than this will simply be replaced by the unknown token
   *        unk_token which was specified in the python script
   *        python/perfect_hash.py
   */
  full_tokenizer(std::string const& vocab_file,
                 uint32_t max_num_sentences,
                 uint32_t max_num_chars,
                 uint32_t max_rows_final_tensor,
                 uint32_t max_sequence_length,
                 uint32_t stride,
                 bool do_truncate,
                 bool do_lower_case,
                 cudaStream_t stream        = 0,
                 int max_inp_chars_per_word = 200);

  /**
   * @brief Splits the input text into token ids.
   *
   * This class is simply a wrapper around the basic and word piece tokenizers.
   *
   * @param device_sentences A vector of strings which MUST be encoded in the utf8 format.
   *        If this precondition is not held then the behavior of this
   *        function is undefined.
   * @param offsets A vector of byte offsets to the beginning of individual strings in
   *        the `device_sentences` parameter.
   * @param offset_size The number of strings identified in `device_sentences`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @return token-ids and offsets
   */
  std::pair<uint32_t*, uint32_t*> tokenize(const char* device_sentences,
                                           const uint32_t* offsets,
                                           uint32_t offset_size,
                                           cudaStream_t stream);

  //  // Returns number of rows in the final tensor with tokenIDS
  //  uint32_t get_nrows_tensor_tokenIDS();
  //  // Returns device pointer to final tensor with tokenIDs
  //  uint32_t* get_tensor_tokenIDS();
  //  // Returns device pointer to final attention mask
  //  uint32_t* get_attention_mask();
  //  // Returns device pointer to final tensor's metadata
  //  uint32_t* get_tensor_metadata();

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

  data_normalizer normalizer;
  uint32_t max_sequence_length;
  uint32_t stride;
  bool do_truncate;

  // hash table parameters
  uint32_t outer_hash_a_param;
  uint32_t outer_hash_b_param;
  uint16_t num_outer_bins;
  uint16_t unk_token_id;
  uint16_t first_tok_id;
  uint16_t sep_tok_id;
  unsigned int max_word_length;
  rmm::device_vector<uint64_t> device_hash_table;
  rmm::device_vector<uint64_t> device_bin_coefficients;
  rmm::device_vector<uint16_t> device_bin_offsets;

  // pointers to device data needed for tokenization
  rmm::device_vector<uint32_t> device_token_ids{};
  rmm::device_vector<uint32_t> device_word_indices;
  rmm::device_vector<uint8_t> device_tokens_per_word;

  rmm::device_vector<size_t> cub_temp_storage;
  rmm::device_vector<uint32_t> device_num_selected;
  size_t max_cub_storage_bytes;
};

}  // namespace detail
}  // namespace nvtext
