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

#include <vector>

#include <rmm/thrust_rmm_allocator.h>

namespace nvtext {

template <typename T>
struct ptr_length_pair {
  T gpu_ptr;
  size_t length;

  ptr_length_pair() : gpu_ptr(nullptr), length(0) {}

  ptr_length_pair(T gpu_ptr, size_t length) : gpu_ptr(gpu_ptr), length(length) {}
};

/**
 *  Performs text cleaning for the wordpiece tokenizer.
 *
 *  Every instantiation of this class will transfer the meta data over to the GPU. Thus,
 *  it is advices to create one class and reuse that class as needed.
 *
 *  Converts characters to lowercase, adds spaces around punctuation and chinese
 *  characters, strips accents from letters in the text and standardizes whitespace
 *  characters to all be the code point for the " " literal.
 */
class GpuBasicTokenizer {
 public:
  /**
   *    Initializes the basic tokenizer. This will initiate the transfer of the metadata
   *    needed to tokenize characters from the host to the GPU.
   *
   *    Params:
   *
   *      max_num_sentences: Maximum number of input sentences for instantiating the tokenizer.
   *                         Used to allocate memory on device.
   *                         If input contains larger number of sentences, behavior is undefined.
   *
   *      max_num_chars: Maximum number of characters for instantiating the tokenizer.
   *                     Used to allocate memory on device.
   *                     If input contains larger number of characters, behavior is undefined.
   *
   *      do_lower_case: If true, the tokenizer will convert uppercase characters in the
   *                      input stream to lower case AND strip accents from those characters.
   *                      If false, accented and uppercase characters are not transformed.
   */
  GpuBasicTokenizer(uint32_t max_num_sentences,
                    uint32_t max_num_chars,
                    std::vector<uint32_t> const& cp_metadata,
                    std::vector<uint64_t> const& aux_table,
                    bool do_lower_case = true);

  /*
    Tokenizes a vector of strings.

    Params: sentences - A vector of strings which MUST be encoded in the utf8 format.
                        If this precondition is not held then the behavior of this
                        function is undefined.

    This function does the actual string cleaning in the description of the class. It
    will first flatten the sentences into a single char array on the host then copy
    them to the device. The device will then tokenize the characters as described.

    If do_lower_case is true, this function will convert each character to lowercase
    AND strip accents from the characters. If false it will do all other conversions
    in the class description except lowercasing and punctuation stripping.

    Returns: Two pointers to GPU data along with their lengths. The first is a pointer
              to the flattened code-points array along with its length and the second is
              a pointer to the start offset in the code points array for each sentence.

              Assuming pair is the return value, we have that:

              pair.first.gpu_ptr: A pointer to a contiguous array of unicode code points
                                  corresponding to the characters in the text after running
                                  basic tokenization. All the sentences are in a flattened
                                  array.

              pair.first.length: The number of unicode code points in the tokenized output.

              pair.second.gpu_ptr: A pointer to the start offsets of the sentences in the
                                  flattened code_point array. That is, sentence i starts
                                  at character pair.second.gpu_ptr[i]. Note: This array
                                  will always be of length text_batch.size() + 1 since
                                  we need one entry for each input and a last entry which
                                  has the total number of characters.

              pair.second.length: Always num_sentences + 1.
  */
  std::pair<ptr_length_pair<uint32_t*>, ptr_length_pair<uint32_t*>> tokenize(
    const char* device_sentences, const uint32_t* offsets, uint32_t offset_size);

  // Frees the GPU data associated with the tokenizer.
  ~GpuBasicTokenizer();

 private:
  bool do_lower_case;

  // pointers to device data needed for tokenization
  rmm::device_vector<uint32_t> device_cp_metadata;
  rmm::device_vector<uint64_t> device_aux_table;

  rmm::device_vector<unsigned char> device_sentences;
  rmm::device_vector<uint32_t> device_sentence_offsets;
  rmm::device_vector<uint32_t> device_code_points;
  rmm::device_vector<uint32_t> device_chars_per_thread;

  rmm::device_vector<size_t> cub_temp_storage;
  rmm::device_vector<uint32_t> device_num_selected;
  size_t max_cub_storage_bytes;
};

/*
    This splits words into tokens contained in the model vocabulary file.
*/
class GpuWordPieceTokenizer {
 public:
  /*
    Creates a tokenizer by loading the preprocessed vocabulary from the file path given.

    vocab_file: A path to the preprocessed vocab.txt file.
                Note that this is the file AFTER python/perfect_hash.py has been used
                for preprocessing. Passing in the default vocab.txt file will cause
                undefined behavior.

    max_num_chars: Maximum number of characters for instantiating the tokenizer.
                   Used to allocate memory on device.
                   If input contains larger number of characters, behavior is undefined.

    max_inp_chars_per_word: The length of the longest word that will be tokenized. Words
                            longer than this will simply be replaced by the unknown token
                            unk_token which was specified in the python script
                            python/perfect_hash.py
  */
  GpuWordPieceTokenizer(std::string const& vocab_file,
                        uint32_t max_num_chars,
                        uint32_t max_inp_chars_per_word = 200);

  /*
    Splits code points run through the basic tokenizer into tokens.

    Params (All modified)
    cps_and_length: A gpu pointer to the codepoints after being run through the basic
                    tokenizer along with the number of code points in the length field.

                    This is modified to contain the token ids of each space separated
                    sequence of code points.

    offsets_and_length: A gpu pointer to the sentence offsets along with the length of
                        the underlying array. This is always num sentences + 1.

                        Note that the data in this pointer is modified after the function call to
                        contain sentence length in terms of token ids instead of code points.

    Returns: A pointer length pair pointing to the token ids and with the number of token ids

              cp_and_length is overwrite to generate output
              offsets_and_length is updated within this function

              Assuming ret is the return value, we have that:

              ret.gpu_ptr: A pointer to a contiguous array of token id of each token
                          found after performing word piece tokenization

              ret.length: The total number of token ids found.


  */
  void tokenize(ptr_length_pair<uint32_t*>& cp_and_length,
                ptr_length_pair<uint32_t*>& offsets_and_length);

  // Frees GPU data associated with the tokenizer
  ~GpuWordPieceTokenizer();

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

class GpuFullTokenizer {
 public:
  /*
    Creates a full tokenizer that cleans the text and splits it into tokens

    vocab_file: A path to the preprocessed vocab.txt file.
                Note that this is the file AFTER python/perfect_hash.py has been used
                for preprocessing. Passing in the default vocab.txt file will cause
                undefined behavior.

    max_num_sentences: Maximum number of input sentences for instantiating the tokenizer.
                       Used to allocate memory on device.
                       If input contains larger number of sentences, behavior is undefined.

    max_num_chars: Maximum number of characters for instantiating the tokenizer.
                   Used to allocate memory on device.
                   If input contains larger number of characters, behavior is undefined.

    max_rows_final_tensor: Maximum number of rows in tensor_tokenIDS expected by tokenizer.
                           Used to allocate memory on device.
                           If output contains larger number of rows, behavior is undefined.

    max_sequence_length: limit the number of tokenIDs per row in final tensor with tokenIDS

    stride: Each row in tensor-tokenIDS will replicate (max_sequence_length - stride) tokenIDs
            from previous row, unless it is the first row of sentence/log.

    do_truncate: If true, tokenizer will discard all the tokenIDs after max_sequence_length
                 for each input sentence/log. If false, it will use a new row in the tensor-tokenIDS
                 to continue generating the output.

    do_lower_case: If true, the tokenizer will convert uppercase characters in the
                    input stream to lower case AND strip accents from those characters.
                    If false, accented and uppercase characters are not transformed.


    max_inp_chars_per_word: The length of the longest word that will be tokenized. Words
                            longer than this will simply be replaced by the unknown token
                            unk_token which was specified in the python script
                            python/perfect_hash.py
  */
  GpuFullTokenizer(std::string const& vocab_file,
                   uint32_t max_num_sentences,
                   uint32_t max_num_chars,
                   uint32_t max_rows_final_tensor,
                   uint32_t max_sequence_length,
                   uint32_t stride,
                   bool do_truncate,
                   bool do_lower_case,
                   int max_inp_chars_per_word = 200);

  /*
    Splits the input text into token ids. This class is simply a wrapper around the basic
    and word piece tokenizers.

    Params: text_batch - A vector of strings which MUST be encoded in the utf8 format.
                          If this precondition is not held then the behavior of this
                          function is undefined.


    Returns: Two pointers to GPU data along with their lengths. The first is a pointer
              to the flattened token ids array along with its length and the second is
              a pointer to the start offset in the token id array for each sentence.

              Assuming pair is the return value, we have that:

              pair.first.gpu_ptr: A pointer to a contiguous array of token id of each token
                                  found after performing word piece tokenization

              pair.first.length: The total number of token ids found.

              pair.second.gpu_ptr: A pointer to the start offsets of the sentences in the
                                  flattened token id array. That is, sentence i starts
                                  at token id pair.second.gpu_ptr[i]. Note: This array
                                  will always be of length text_batch.size() + 1 since
                                  we need one entry for each input and a last entry which
                                  has the total number of token ids.

              pair.second.length: Always num_sentences + 1.

  */
  void tokenize(const char* device_sentences, const uint32_t* offsets, uint32_t offset_size);

  // Returns number of rows in the final tensor with tokenIDS
  uint32_t get_nrows_tensor_tokenIDS();

  // Returns device pointer to final tensor with tokenIDs
  uint32_t* get_tensor_tokenIDS();

  // Returns device pointer to final attention mask
  uint32_t* get_attention_mask();

  // Returns device pointer to final tensor's metadata
  uint32_t* get_tensor_metadata();

  // Frees GPU data associated with the tokenizer
  ~GpuFullTokenizer();

 private:
  GpuBasicTokenizer basic_tokenizer;
  GpuWordPieceTokenizer word_piece_tokenizer;

  uint32_t nrows_tensor_tokenIDS;
  rmm::device_vector<uint32_t> tensor_tokenIDS;
  rmm::device_vector<uint32_t> attention_mask;  // on device
  rmm::device_vector<uint32_t> metadata;  // on device (one row per tensor row, with 3 elements
                                          // [rowID, starting_pos, stop_pos])

  // correspondence between each row of tensor_tokenIDS and log_id
  rmm::device_vector<uint32_t> device_row2log;
  // correspondence between each row of tensor_tokenIDS and row number within s specific log
  rmm::device_vector<uint32_t> device_row2row_within_log;
  uint32_t max_sequence_length;
  uint32_t stride;
  bool do_truncate;
};

}  // namespace nvtext
