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

#include "text/subword/detail/hash_utils.cuh"
#include "text/subword/detail/tokenizer_utils.cuh"
#include "text/subword/detail/wordpiece_tokenizer.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>

#include <nvtext/subword_tokenize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Initializes the token-ids, word-indices, and token counts vectors.
 *
 * Each thread process a single code point from `code_points`.
 * This also locates the start and end of each word within the `code_points` buffer.
 * A word start is identified as a non-space character that appears right after a space.
 * A word end is identified as a space character that appears right after a non-space one.
 * If the code point at this thread does not represent a word start or word end,
 * a max uint32_t value is written to the appropriate vector instead.
 * A post processing step is required to filter the relevant values in these
 * vectors.
 *
 * It is guaranteed that the same number of valid values will be written to both the
 * start and end indices and that after the select step, the two arrays will be aligned.
 * That is, `start_word_indices[word]` and `end_word_indices[word]` are the start and
 * end for the same word.
 *
 * Memory required is 13 bytes per code point values:
 * - 4 bytes each for `start_word_indices` and `end_word_indices`
 * - 4 bytes for each `token_ids`
 * - 1 byte for each `tokens_per_word`
 * Also, there is a code point value for each byte in the input strings.
 *
 * @param[in] code_points A pointer to the code points in the strings after normalization.
 * @param[out] start_word_indices An array of size `num_code_points` which will contain the
 *        starting index for each word.
 * @param[out] end_word_indices An array of size `num_code_points` which will contain the
 *        ending index for each word.
 * @param num_code_points The total number of code_points.
 * @param[out] token_ids An array of size `num_code_points` which will hold the token ids.
 *        This kernel just sets all the values to max uint32_t.
 * @param[out] tokens_per_word An array of size `num_code_points` which hold the number of
 *        tokens. This kernel just sets all the values to 0.
 */
CUDF_KERNEL void init_data_and_mark_word_start_and_ends(uint32_t const* code_points,
                                                        uint32_t* start_word_indices,
                                                        uint32_t* end_word_indices,
                                                        size_t num_code_points,
                                                        uint32_t* token_ids,
                                                        uint8_t* tokens_per_word)
{
  auto const char_for_thread = cudf::detail::grid_1d::global_thread_id();

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
 * This kernel should be called after `init_data_and_mark_word_start_and_ends` with at
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
CUDF_KERNEL void mark_string_start_and_ends(uint32_t const* code_points,
                                            int64_t const* strings_offsets,
                                            uint32_t* start_word_indices,
                                            uint32_t* end_word_indices,
                                            uint32_t num_strings)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  // Ensure the starting character of each strings is written to the word start array.
  if (idx <= num_strings) {
    auto const offset = strings_offsets[idx];

    if ((idx < num_strings) && (code_points[offset] != SPACE_CODE_POINT)) {
      start_word_indices[offset] = offset;
    }

    if ((offset > 0) && (code_points[offset - 1] != SPACE_CODE_POINT)) {
      end_word_indices[offset - 1] = offset;
    }
  }
}

/**
 * @brief Currently supported special tokens.
 *
 * Code logic expects these to be 3 upper-case characters along
 * with a single trailing space.
 */
__constant__ char special_tokens[35]{"BOS EOS UNK SEP PAD CLS MASK "};
constexpr cudf::size_type MIN_ST_WIDTH = 4;  // Min token size in special_tokens
constexpr cudf::size_type MAX_ST_WIDTH = 5;  // Max token size in special_tokens

struct mark_special_tokens {
  /**
   * @brief Check given code-point array to the list of known
   * special tokens.
   */
  __device__ bool is_special_token(uint32_t const* token, cudf::size_type size) const
  {
    if (size < MIN_ST_WIDTH || size > MAX_ST_WIDTH) return false;
    char str_token[MAX_ST_WIDTH];
    // convert code-points to chars
    thrust::transform(thrust::seq, token, token + size, str_token, [](uint32_t cp) {
      // also upper-case them to match again special_tokens array
      return static_cast<char>(cp >= 'a' ? cp - 'a' + 'A' : cp);
    });
    // search the special tokens array for the str_token
    cudf::string_view tokens(special_tokens, sizeof(special_tokens));
    return tokens.find(str_token, size) != cudf::string_view::npos;
  }

  /**
   * @brief Check code-points for special tokens and adjust indices.
   *
   * Tokens will appear in the `code_points` array as:
   * `_[_ttt_]_` where `_` are single space characters and
   *                   ttt is the variable-length token name
   *
   * The logic below uses the following variables to represent position
   * values in the `code_points` array after locating a special token:
   * ```
   * _ [ _ t t t _  ] _
   *   ^   ^     ^  ^
   *   si  sp    ep ei
   * ```
   * where `si` is `start_index`
   *       `sp` is `start_pos`
   *       `ep` is `end_pos`
   *       `ei` is `end_index`
   *
   * When a special token is found, the `code_points` are adjusted
   * to remove the spaces and capitalize the name.
   * ```
   * _ [ _ t t t _ ] _  is updated to
   * _ [ T T T ] _ ] _
   * ```
   * This is required for the downstream word-piece tokenizer to
   * match it to the vocabulary hash table.
   *
   * The `start_word_indices` and `end_word_indices` are updated to
   * identify the token and to ignore the extra trailing `]` character.
   */
  __device__ void operator()(size_t idx) const
  {
    uint32_t const start_index = start_word_indices[idx];
    if ((start_index == std::numeric_limits<uint32_t>::max()) ||
        ((start_index + MIN_ST_WIDTH + 2) > num_code_points))
      return;
    if (code_points[start_index] != '[') return;

    // check for matching end bracket
    uint32_t const start_pos = start_index + 2;  // after the space delimiter
    // search for next start-word and then check it is a ']'
    uint32_t const end_index = [&] {
      auto const begin = start_word_indices + start_pos;
      auto const width =
        std::min(static_cast<size_t>(MAX_ST_WIDTH + 1), (num_code_points - start_pos));
      auto const end = begin + width;
      // checking the next start-word is more reliable than arbitrarily searching for ']'
      // in case the text is split across string rows
      auto const iter = thrust::find_if(thrust::seq, begin + 1, end, [](auto swi) {
        return swi != std::numeric_limits<uint32_t>::max();
      });
      return iter == end ? start_index : static_cast<uint32_t>(iter - start_word_indices);
    }();
    if (code_points[end_index] != ']') return;

    // check for special token
    auto const size = static_cast<cudf::size_type>(end_index - start_pos);
    if (!is_special_token(code_points + start_pos, size)) return;

    // special token found
    // adjust code-points
    auto const end_pos = end_index - 2;
    // change _[_ttt_]_ to _[TTT]_
    for (auto left_idx = start_pos - 1; left_idx <= end_pos; ++left_idx) {
      auto const cp         = code_points[left_idx + 1];
      code_points[left_idx] = cp >= 'a' ? cp - 'a' + 'A' : cp;
    }
    code_points[end_pos] = ']';

    // erase the intermediate indices
    thrust::fill(thrust::seq,
                 start_word_indices + start_index + 1,  // keep the first one
                 start_word_indices + end_index + 1,
                 std::numeric_limits<uint32_t>::max());
    thrust::fill(thrust::seq,
                 end_word_indices + start_index,
                 end_word_indices + end_index + 1,
                 std::numeric_limits<uint32_t>::max());

    // reset the new end-word index
    end_word_indices[end_pos] = end_pos + 1;
  }

  uint32_t* const code_points;
  uint32_t* const start_word_indices;
  uint32_t* const end_word_indices;
  size_t const num_code_points;
};

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
CUDF_KERNEL void kernel_wordpiece_tokenizer(uint32_t const* code_points,
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
  auto const word_to_tokenize = cudf::detail::grid_1d::global_thread_id();

  if (word_to_tokenize >= total_words) { return; }
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
                                         uint32_t max_sequence_length,
                                         uint32_t stride,
                                         bool do_truncate,
                                         bool do_lower_case,
                                         uint32_t max_word_length)
  : vocab_table(vocab_table),
    normalizer(vocab_table.cp_metadata->view().data<codepoint_metadata_type>(),
               vocab_table.aux_cp_table->view().data<aux_codepoint_data_type>(),
               do_lower_case),
    max_sequence_length{max_sequence_length},
    stride(stride),
    do_truncate(do_truncate),
    max_word_length{max_word_length}
{
}

uvector_pair wordpiece_tokenizer::tokenize(cudf::strings_column_view const& input,
                                           rmm::cuda_stream_view stream)
{
  auto cps_and_offsets = normalizer.normalize(input, stream);
  tokenize(cps_and_offsets, stream);
  return uvector_pair(std::move(cps_and_offsets.first), std::move(cps_and_offsets.second));
}

struct copy_if_fn {  // inline lambda not allowed in private or protected member function
  __device__ bool operator()(uint32_t cp) { return cp != std::numeric_limits<uint32_t>::max(); }
};

struct tranform_fn {  // just converting uint8 value to uint32
  __device__ uint32_t operator()(uint8_t count) { return count; }
};

void wordpiece_tokenizer::tokenize(uvector_pair& cps_and_offsets, rmm::cuda_stream_view stream)
{
  auto device_code_points     = cps_and_offsets.first->data();
  auto const num_code_points  = cps_and_offsets.first->size();
  auto device_strings_offsets = cps_and_offsets.second->data();
  auto const num_strings      = cps_and_offsets.second->size() - 1;

  size_t const four_byte_cp_chunks = 1 + (num_code_points - 1) / sizeof(uint32_t);
  size_t const rounded_num_cps     = sizeof(uint32_t) * four_byte_cp_chunks;
  rmm::device_uvector<uint8_t> device_tokens_per_word(rounded_num_cps, stream);
  rmm::device_uvector<uint32_t> device_token_ids(num_code_points, stream);
  rmm::device_uvector<uint32_t> device_word_indices(2 * num_code_points, stream);

  // make device_start_word_indices and device_end_word_indices contiguous
  uint32_t* device_start_word_indices = device_word_indices.data();
  uint32_t* device_end_word_indices   = device_start_word_indices + num_code_points;

  cudf::detail::grid_1d const grid_init{static_cast<cudf::size_type>(num_code_points),
                                        THREADS_PER_BLOCK};
  detail::init_data_and_mark_word_start_and_ends<<<grid_init.num_blocks,
                                                   grid_init.num_threads_per_block,
                                                   0,
                                                   stream.value()>>>(device_code_points,
                                                                     device_start_word_indices,
                                                                     device_end_word_indices,
                                                                     num_code_points,
                                                                     device_token_ids.data(),
                                                                     device_tokens_per_word.data());
  CUDF_CHECK_CUDA(stream.value());

  cudf::detail::grid_1d const grid_mark{static_cast<cudf::size_type>(num_strings + 1),
                                        THREADS_PER_BLOCK};
  detail::mark_string_start_and_ends<<<grid_mark.num_blocks,
                                       grid_mark.num_threads_per_block,
                                       0,
                                       stream.value()>>>(device_code_points,
                                                         device_strings_offsets,
                                                         device_start_word_indices,
                                                         device_end_word_indices,
                                                         num_strings);
  CUDF_CHECK_CUDA(stream.value());

  // check for special tokens and adjust indices
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_t>(0),
    num_code_points,
    mark_special_tokens{
      device_code_points, device_start_word_indices, device_end_word_indices, num_code_points});

  // Now start_word_indices has the word starts scattered throughout the array. We need to select
  // all values not equal to the max uint32_t and place them at the start of the array. We leverage
  // the fact that the start_word_indices and the end_word indices are contiguous to only launch one
  // device select kernel.
  auto itr_end = thrust::remove(rmm::exec_policy(stream),
                                device_word_indices.begin(),
                                device_word_indices.end(),
                                std::numeric_limits<uint32_t>::max());

  // The number of tokens selected will be double the number of words since we
  // select from both the start and end index arrays.
  uint32_t const num_words = thrust::distance(device_word_indices.begin(), itr_end) / 2;

  // We need to change the end_word_indices pointer after the selection is complete
  device_end_word_indices = device_start_word_indices + num_words;

  if (num_words > 0) {
    cudf::detail::grid_1d const grid{static_cast<cudf::size_type>(num_words), THREADS_PER_BLOCK};
    detail::kernel_wordpiece_tokenizer<<<grid.num_blocks,
                                         grid.num_threads_per_block,
                                         0,
                                         stream.value()>>>(
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
    CUDF_CHECK_CUDA(stream.value());
  }

  // Repurpose the input array for the token ids. In the worst case, each code point ends up being a
  // token so this will always have enough memory to store the contiguous tokens.
  uint32_t* contiguous_token_ids = device_code_points;
  auto const copy_size           =  // thrust::copy_if limited to copying int-max values
    std::min(device_token_ids.size(), static_cast<std::size_t>(std::numeric_limits<int>::max()));
  auto ids_itr       = device_token_ids.begin();
  auto const ids_end = device_token_ids.end();
  while (ids_itr != ids_end) {
    auto const copy_end  = (static_cast<std::size_t>(std::distance(ids_itr, ids_end)) <= copy_size)
                             ? ids_end
                             : ids_itr + copy_size;
    contiguous_token_ids = thrust::copy_if(
      rmm::exec_policy(stream), ids_itr, copy_end, contiguous_token_ids, copy_if_fn{});
    ids_itr = copy_end;
  }

  // Repurpose start word indices since it is the same size and type as the required output.
  uint32_t* token_id_counts = device_start_word_indices;
  thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                   device_tokens_per_word.data(),
                                   device_tokens_per_word.data() + num_code_points,
                                   token_id_counts,
                                   tranform_fn{},
                                   thrust::plus<uint32_t>());

  // Update the device_strings_offsets using the token_id_counts
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<uint32_t>(1),
                     num_strings,
                     update_strings_lengths_fn{token_id_counts, device_strings_offsets});
}

}  // namespace detail
}  // namespace nvtext
