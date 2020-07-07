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

#include <text/subword/detail/cp_data.h>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/error.hpp>
#include <nvtext/subword_tokenize.hpp>
#include <text/subword/detail/hash_utils.cuh>
#include <text/subword/detail/tokenizer_utils.cuh>
#include <text/subword/detail/wordpiece_tokenizer.hpp>

#include <thrust/for_each.h>
#include <thrust/transform_scan.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <iostream>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Initializes the token-ids, word-indices, and token counts vectors.
 *
 * Each thread process a single code point from code_points.
 * This also locates the start and end of each word within the code_points buffer.
 * A word start is identified as a non-space character that appears right after a space.
 * A word end is identified as a space character that appears right after a non-space one.
 * If the code point at this thread does not represent a word start or word end,
 * a max uint32_t value is written to the appropriate vector instead.
 * A post processing step is required to filter the relevant values in these
 * vectors.
 *
 * It is guaranteed that the same number of valid values will be written to both the
 * start and end indices and that after the select step, the two arrays will be aligned.
 *  That is, `start_word_indices[word]` and `end_word_indices[word]` are the start and
 * end for the same word.
 *
 * @param code_points[in] A pointer to the code points in the strings after normalization.
 * @param start_word_indices[out] An array of size `num_code_points` which will contain the
 *        starting index for each word.
 * @param end_word_indices[out] An array of size `num_code_points` which will contain the
 *        ending index for each word.
 * @param num_code_points The total number of code_points.
 * @param token_ids[out] An array of size `num_code_points` which will hold the token ids.
 *        This kernel just sets all the values to max uint32_t.
 * @param tokens_per_word[out] An array of size `num_code_points` which hold the number of
 *        tokens. This kernel just sets all the values to 0.
 */
__global__ void init_data_and_mark_word_start_and_ends(uint32_t const* code_points,
                                                       uint32_t* start_word_indices,
                                                       uint32_t* end_word_indices,
                                                       size_t num_code_points,
                                                       uint32_t* token_ids,
                                                       uint8_t* tokens_per_word)
{
  uint32_t char_for_thread = blockDim.x * blockIdx.x + threadIdx.x;

  // Deal with the start_word_indices array
  if (char_for_thread < num_code_points) {
    uint32_t val_to_write = std::numeric_limits<uint32_t>::max();
    if ((code_points[char_for_thread] != SPACE_CODE_POINT) && (char_for_thread > 0) &&
        (code_points[char_for_thread - 1] == SPACE_CODE_POINT)) {
      val_to_write = char_for_thread;
    }
    start_word_indices[char_for_thread] = val_to_write;

    // Deal with the end_word_indices_array
    val_to_write = std::numeric_limits<uint32_t>::max();
    if ((code_points[char_for_thread] != SPACE_CODE_POINT) &&
        (char_for_thread + 1 < num_code_points) &&
        (code_points[char_for_thread + 1] == SPACE_CODE_POINT)) {
      val_to_write = char_for_thread + 1;
    }
    end_word_indices[char_for_thread] = val_to_write;

    token_ids[char_for_thread]       = std::numeric_limits<uint32_t>::max();
    tokens_per_word[char_for_thread] = 0;
  }
}

/**
 * @brief Resolves the string boundaries for the start and end words.
 *
 * This kernel should be called after `mark_word_start_and_ends` with at
 * least `num_strings` total threads.
 *
 * The start and end indices are updated to honor the string boundaries
 * within the strings array. This corrects any word ranges that span across
 * individual strings.
 *
 * @param code_points A pointer to the code points in the strings.
 * @param strings_offsets An array containing the index of the starting character of each string
 *        with an extra space at the end containing the total number of characters. As a result,
 *        this array is of length num_strings + 1.
 * @param start_word_indices An array which will contain the starting index for each word scattered
 *        throughout. If an index does not represent a word start, the max-uint32_t value is written
 *        to indicate this.
 * @param end_word_indices An array which will contain the one past the end index for each word
 *        scattered throughout. If an index does not represent a word end, the max uint32_t value is
 *        written to indicate this.
 * @param num_strings The total number of strings to be processed.
 */
__global__ void mark_string_start_and_ends(uint32_t const* code_points,
                                           uint32_t const* strings_offsets,
                                           uint32_t* start_word_indices,
                                           uint32_t* end_word_indices,
                                           uint32_t num_strings)
{
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  // Ensure the starting character of each strings is written to the word start array.
  if (idx <= num_strings) {
    auto const offset = strings_offsets[idx];

    if ((idx < num_strings) && (code_points[offset] != SPACE_CODE_POINT)) {
      start_word_indices[offset] = offset;
    }

    if ((idx > 0) && (code_points[offset - 1] != SPACE_CODE_POINT)) {
      end_word_indices[offset - 1] = offset;
    }
  }
}

/**
 * @brief Converts words into token ids.
 *
 * Each thread is assigned a word to convert based on the `hash_table`. Each thread converts
 * its word and writes the number of tokens it found in the `tokens_per_word` array.
 *
 * The `tokens_per_word` array is kept to the length `num_code_points + 1`. This means each thread
 * can write its number of tokens to the `tokens_per_word` corresponding to the starting
 * character of each word. Since strings must start at some word, we can prefix sum this array
 * and use the strings_lengths code point offsets to directly index the number of tokens in each
 * string.
 *
 * The `token_ids` array should be initialized to the max uint32_t before calling this kernel.
 *
 * @param code_points An array containing all of the code points to be processed
 * @param hash_table An array containing the flattened hash table with key, value pairs
 *        packed in 64-bits
 * @param bin_coefficients A pointer to the GPU pointer containing the hashing parameters for
 *        each hash bin on the GPU.
 * @param bin_offsets: A pointer to the GPU pointer containing the start index of each bin in
 *        the flattened hash table.
 * @param token_ids The index for each token found during tokenization. This is of length
 *        num_code_points. In most cases, multiple characters will collapse to one token. In these
 *        cases, the max uint32_t will be in place. Cub will be used later to filter out these
 *        invalid ids later.
 * @param word_starts An array of length `num_code_points`. The first total word elements contains
 *        the index of the first character for each word.
 * @param word_ends An array of length num_code_points. The first total_words elements contains the
 *        past the end index for each word. This array is kept aligned with the initial
 *        token_ids array containing the word start code points.
 *        `word_ends[word] - filtered_start_indices[word] = word_length`
 * @param tokens_per_word An array of size num_code_points that will contain the number of tokens in
 *        each word in a string. This array can be exclusive summed and the result used in
 *        conjunction with the strings lengths array to find the tokens in each string. This is
 *        possible since the number of tokens in each word will be placed at the index corresponding
 *        to the start character of a word. If we assume prefix_summed is the prefix sum of the
 *        tokens_per_word array, then `prefix_summed[strings_lengths[string_idx] - 1]` is the number
 *        of tokens found before the start of string.
 * @param unk_token_id The token id to be place for unknown tokens
 * @param max_word_length The maximum length of a word. Any word longer than this length is
 *        replaced by the unknown token.
 * @param total_words The total number of white space separated words
 * @param outer_hash_a_param The a parameter for the outer hash
 * @param outer_hash_b_param: The b parameter for the outer hash
 * @param num_outer_bins: The number of bins for the outer hash
 */
__global__ void kernel_wordpiece_tokenizer(uint32_t const* code_points,
                                           uint64_t const* hash_table,
                                           uint64_t const* bin_coefficients,
                                           uint16_t const* bin_offsets,
                                           uint16_t unk_token_id,
                                           uint32_t outer_hash_a_param,
                                           uint32_t outer_hash_b_param,
                                           uint16_t num_outer_bins,
                                           uint32_t const* word_starts,
                                           uint32_t const* word_ends,
                                           uint32_t max_word_length,
                                           uint32_t total_words,
                                           uint32_t* token_ids,
                                           uint8_t* tokens_per_word)
{
  uint32_t const word_to_tokenize = blockDim.x * blockIdx.x + threadIdx.x;

  if (word_to_tokenize >= total_words) return;
  // Each thread gets the start code_point offset for each word and resets the token_id memory to
  // the default value. In a post processing step, all of these values will be removed.
  auto const token_start = word_starts[word_to_tokenize];
  auto const token_end   = word_ends[word_to_tokenize];
  auto const word_length = token_end - token_start;

  // The sdbm hash of "##"
  constexpr uint32_t hashtag_hash = 2296000;
  uint16_t num_values_tokenized   = 0;
  // initialize start, end
  uint32_t start = token_start;
  uint32_t end   = token_end;

  if (word_length > max_word_length) {
    start                        = token_end;
    num_values_tokenized         = 1;
    token_ids[token_start]       = unk_token_id;
    tokens_per_word[token_start] = num_values_tokenized;
  }

  while (start < token_end) {
    end = token_end;
    // init token_id to no token
    int token_id = -1;
    // compute current length
    uint32_t const length = token_end - start;
    uint64_t substr_hash =
      sdbm_hash(code_points + start, length, start == token_start ? 0 : hashtag_hash);
    while (start < end) {
      token_id = retrieve(substr_hash,
                          outer_hash_a_param,
                          outer_hash_b_param,
                          num_outer_bins,
                          hash_table,
                          bin_coefficients,
                          bin_offsets);
      if (token_id != -1) { break; }
      --end;
      // Pop off the last value from the substr hash
      substr_hash = prev_sdbm_hash(substr_hash, code_points[end]);
    }

    if (token_id == -1) {
      end      = token_end;
      token_id = unk_token_id;
      // We need to clean up the global array. This case is very uncommon.
      //  Only 0.016% of words cannot be resolved to a token from the squad dev set.
      for (uint32_t i = 1; i < num_values_tokenized; ++i) {
        token_ids[token_start + i] = std::numeric_limits<uint32_t>::max();
      }
      num_values_tokenized = 0;
    }

    token_ids[token_start + num_values_tokenized] = token_id;
    ++num_values_tokenized;
    start = end;
  }

  tokens_per_word[token_start] = num_values_tokenized;
}

}  // namespace

wordpiece_tokenizer::wordpiece_tokenizer(hashed_vocabulary const& vocab_table,
                                         uint32_t max_num_strings,
                                         uint32_t max_num_chars,
                                         uint32_t max_rows_final_tensor,
                                         uint32_t max_sequence_length,
                                         uint32_t stride,
                                         bool do_truncate,
                                         bool do_lower_case,
                                         cudaStream_t stream,
                                         uint32_t max_word_length)
  : vocab_table(vocab_table),
    max_sequence_length{max_sequence_length},
    max_word_length{max_word_length},
    stride(stride),
    do_truncate(do_truncate),
    normalizer(max_num_strings, max_num_chars, stream, do_lower_case),
    device_token_ids(MAX_NEW_CHARS * max_num_chars, stream),
    device_word_indices(2 * MAX_NEW_CHARS * max_num_chars, stream),
    device_tokens_per_word(0, stream),
    device_num_selected(1, stream),
    cub_temp_storage(0, stream)
{
  const size_t max_new_char_total        = MAX_NEW_CHARS * max_num_chars;
  const size_t device_word_indices_count = 2 * max_new_char_total;

  const size_t four_byte_cp_chunks = 1 + (max_new_char_total - 1) / sizeof(uint32_t);
  const size_t rounded_num_cps     = sizeof(uint32_t) * four_byte_cp_chunks;
  device_tokens_per_word.resize(rounded_num_cps, stream);

  // Determine temporary device storage requirements for cub
  static NotEqual select_op(std::numeric_limits<uint32_t>::max());
  size_t temp_storage_bytes = 0, temp_storage_bytes_2 = 0;
  cub::DeviceSelect::If(nullptr,
                        temp_storage_bytes,
                        device_word_indices.data(),
                        device_word_indices.data(),
                        device_num_selected.data(),
                        device_word_indices_count,
                        select_op,
                        stream);
  cub::DeviceScan::InclusiveSum(nullptr,
                                temp_storage_bytes_2,
                                device_tokens_per_word.data(),
                                device_word_indices.data(),
                                max_new_char_total,
                                stream);
  max_cub_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_2);
  cub_temp_storage.resize(max_cub_storage_bytes, stream);
}

std::pair<uint32_t*, uint32_t*> wordpiece_tokenizer::tokenize(char const* d_strings,
                                                              uint32_t const* d_offsets,
                                                              uint32_t num_strings,
                                                              cudaStream_t stream)
{
  auto cps_and_offsets = normalizer.normalize(d_strings, d_offsets, num_strings, stream);
  tokenize(cps_and_offsets.first, cps_and_offsets.second, stream);
  return std::make_pair(cps_and_offsets.first.gpu_ptr, cps_and_offsets.second.gpu_ptr);
}

void wordpiece_tokenizer::tokenize(ptr_length_pair& cp_and_length,
                                   ptr_length_pair& offsets_and_length,
                                   cudaStream_t stream)
{
  uint32_t* device_code_points     = cp_and_length.gpu_ptr;
  size_t const num_code_points     = cp_and_length.length;
  uint32_t* device_strings_offsets = offsets_and_length.gpu_ptr;
  uint32_t const num_strings       = offsets_and_length.length - 1;

  // make device_start_word_indices and device_end_word_indices contiguous
  uint32_t* device_start_word_indices = device_word_indices.data();
  uint32_t* device_end_word_indices   = device_start_word_indices + num_code_points;

  cudf::detail::grid_1d const grid_init{static_cast<cudf::size_type>(num_code_points),
                                        THREADS_PER_BLOCK};
  detail::init_data_and_mark_word_start_and_ends<<<grid_init.num_blocks,
                                                   grid_init.num_threads_per_block,
                                                   0,
                                                   stream>>>(device_code_points,
                                                             device_start_word_indices,
                                                             device_end_word_indices,
                                                             num_code_points,
                                                             device_token_ids.data(),
                                                             device_tokens_per_word.data());
  CHECK_CUDA(stream);

  cudf::detail::grid_1d const grid_mark{static_cast<cudf::size_type>(num_strings),
                                        THREADS_PER_BLOCK};
  detail::mark_string_start_and_ends<<<grid_mark.num_blocks,
                                       grid_mark.num_threads_per_block,
                                       0,
                                       stream>>>(device_code_points,
                                                 device_strings_offsets,
                                                 device_start_word_indices,
                                                 device_end_word_indices,
                                                 num_strings);
  CHECK_CUDA(stream);

  // Now start_word_indices has the word starts scattered throughout the array. We need to select
  // all values not equal to the max uint32_t and place them at the start of the array. We leverage
  // the fact that the start_word_indices and the end_word indices are contiguous to only launch one
  // device select kernel.
  // Create a selection op for all device selects
  NotEqual const select_op(std::numeric_limits<uint32_t>::max());
  cub::DeviceSelect::If(cub_temp_storage.data(),
                        max_cub_storage_bytes,
                        device_start_word_indices,  // input
                        device_start_word_indices,  // output
                        device_num_selected.data(),
                        2 * num_code_points,
                        select_op,
                        stream);
  CHECK_CUDA(stream);

  // Grab the number of words which is the number of threads needed for the main word piece
  // tokenizer kernel. The number of tokens selected will be double the number of words since we
  // select from both the start and end index arrays.
  uint32_t num_words = 0;
  CUDA_TRY(cudaMemcpyAsync(
    &num_words, device_num_selected.data(), sizeof(num_words), cudaMemcpyDeviceToHost, stream));
  num_words /= 2;

  // We need to change the end_word_indices pointer after the selection is complete
  device_end_word_indices = device_start_word_indices + num_words;

  cudf::detail::grid_1d const grid{static_cast<cudf::size_type>(num_words), THREADS_PER_BLOCK};
  detail::kernel_wordpiece_tokenizer<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
    device_code_points,
    vocab_table.table->view().data<uint64_t>(),
    vocab_table.bin_coefficients->view().data<uint64_t>(),
    vocab_table.bin_offsets->view().data<uint16_t>(),
    vocab_table.unknown_token_id,
    vocab_table.outer_hash_a,
    vocab_table.outer_hash_b,
    vocab_table.num_bins,
    device_start_word_indices,
    device_end_word_indices,
    max_word_length,
    num_words,
    device_token_ids.data(),
    device_tokens_per_word.data());
  CHECK_CUDA(stream);

  // Repurpose the input array for the token ids. In the worst case, each code point ends up being a
  // token so this will always have enough memory to store the contiguous tokens.
  uint32_t* contiguous_token_ids = device_code_points;
  cub::DeviceSelect::If(cub_temp_storage.data(),
                        max_cub_storage_bytes,
                        device_token_ids.data(),
                        contiguous_token_ids,
                        device_num_selected.data(),
                        num_code_points,
                        select_op,
                        stream);
  CHECK_CUDA(stream);

  // Repurpose start word indices since it is the same size and type as the required output.
  uint32_t* token_id_counts = device_start_word_indices;
  cub::DeviceScan::InclusiveSum(cub_temp_storage.data(),
                                max_cub_storage_bytes,
                                device_tokens_per_word.data(),
                                token_id_counts,
                                num_code_points,
                                stream);
  CHECK_CUDA(stream);

  // Update the device_strings_offsets using the token_id_counts
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<uint32_t>(1),
                     num_strings,
                     update_strings_lengths_fn{token_id_counts, device_strings_offsets});

  // Grab total number of token ids from the device
  uint32_t total_token_ids = 0;
  CUDA_TRY(cudaMemcpyAsync(&total_token_ids,
                           token_id_counts + num_code_points - 1,
                           sizeof(total_token_ids),
                           cudaMemcpyDeviceToHost,
                           stream));

  cp_and_length.length = total_token_ids;
}

}  // namespace detail
}  // namespace nvtext
