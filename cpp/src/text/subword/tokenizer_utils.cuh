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

#include <stdint.h>
#include <vector>

#define SPACE_CODE_POINT 32
#define MAX_NEW_CHARS 3

// A selection op for cub to get elements from an array not equal to a certain value. See
// https://nvlabs.github.io/cub/structcub_1_1_device_partition.html for an example of this
// struct.
struct NotEqual {
  uint32_t val_to_omit;

  __host__ __device__ __forceinline__ NotEqual(uint32_t val_to_omit) : val_to_omit(val_to_omit) {}

  __host__ __device__ __forceinline__ bool operator()(const uint32_t& a) const
  {
    return (a != val_to_omit);
  }
};

/*

*/
static __global__ void update_sentence_lengths(uint32_t* old_lengths,
                                               uint32_t* chars_up_to_idx,
                                               size_t num_sentences)
{
  uint32_t sen_for_thread = threadIdx.x + blockDim.x * blockIdx.x + 1;

  if (sen_for_thread <= num_sentences) {
    old_lengths[sen_for_thread] = chars_up_to_idx[old_lengths[sen_for_thread] - 1];
  }
}

void gpu_basic_tokenize(const std::vector<std::string>& sentences,
                        bool do_lower_case,
                        uint32_t*& device_result_code_points,
                        uint32_t*& device_cleaned_sentence_lengths,
                        uint32_t* device_cp_metadata,
                        uint64_t* device_aux_data,
                        uint32_t* device_sentence_offsets);

/*
  Runs word piece tokenization on the GPU.

  Params
  -------
  dev_code_points: A pointer to the code points after basic tokenization on the device. The length
  of this array is num_code_points.

  dev_sentence_offsets: A pointer an array containing the index of the starting character of each
  sentence with an extra space at the end containing the total number of characters. As a result,
  this array is of length num_sentences + 1.

  dev_hash_table: A pointer to the flattened hash_table on the device

  dev_bin_coefficients: A pointer to the coefficients for each hash bin

  dev_bin_offsets: A pointer to the start of each bin in the flattened hash table.

  unk_tok_id: The token id to be place for unknown tokens.

  max_word_length: The maximum length of a word. Any word longer than this length is replaced by the
  unknown token.

  num_code_points: The total number of code points after basic tokenization

  num_sentences: The total number of sentences to be tokenized

  outer_table_a: The a parameter for the outer hash

  outer_table_b: The b parameter for the outer hash

  num_bins: The number of bins for the outer hash

  Modifies
  ----------
  dev_code_points: Updates data to be the token IDs tokens in each word. A word is a sequence of
  code points separated by a space code point.

  dev_sentence_offsets - Updates the sentence offsets to be the index of the starting token_id for
  each sentence. The last slot is updated to contain the total number of token_ids.

                         Therefore, assuming  pair is the return value of this function,
                         pair.second == dev_sentence_offsets[num_sentences].


  Returns
  ---------
  The total number of token ids.
*/
// uint32_t gpuWordPieceTokenize(uint32_t* dev_code_points, uint32_t* dev_sentence_offsets,
//                              uint64_t* dev_hash_table, uint64_t* dev_bin_coefficients,
//                              uint16_t* dev_bin_offsets, uint16_t unk_tok_id,
//                              uint16_t max_word_length, size_t num_code_points,
//                              uint32_t num_sentences, uint32_t outer_table_a,
//                              uint32_t outer_table_b, uint16_t num_bins);
