/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/atomic>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf::strings::detail {

/**
 * @brief Base class for delimiter-based tokenizers
 *
 * These are common methods used by both split and rsplit tokenizer functors.
 *
 * The Derived class is required to implement the `process_tokens` function.
 */
template <typename Derived>
struct base_split_tokenizer {
  __device__ char const* get_base_ptr() const { return d_strings.head<char>(); }

  __device__ string_view const get_string(size_type idx) const
  {
    return d_strings.element<string_view>(idx);
  }

  __device__ bool is_valid(size_type idx) const { return d_strings.is_valid(idx); }

  /**
   * @brief Returns `true` if the byte at `idx` is the start of the delimiter
   *
   * @param idx Index of a byte in the chars column
   * @param d_offsets Offsets values to locate the chars ranges
   * @param chars_bytes Total number of characters to process
   * @return true if delimiter is found starting at position `idx`
   */
  __device__ bool is_delimiter(int64_t idx,
                               cudf::detail::input_offsetalator const d_offsets,
                               int64_t chars_bytes) const
  {
    auto const d_chars = get_base_ptr() + d_offsets[0];
    if (idx + d_delimiter.size_bytes() > chars_bytes) { return false; }
    return d_delimiter.compare(d_chars + idx, d_delimiter.size_bytes()) == 0;
  }

  /**
   * @brief This counts the tokens for strings that contain delimiters
   *
   * Counting tokens is the same regardless if counting from the left
   * or from the right. This logic counts from the left which is simpler.
   * The count will be truncated appropriately to the max_tokens value.
   *
   * @param idx Index of input string
   * @param d_positions Start positions of all the delimiters
   * @param d_delimiter_offsets Offsets per string to delimiters in d_positions
   */
  __device__ size_type count_tokens(size_type idx,
                                    int64_t const* d_positions,
                                    cudf::detail::input_offsetalator d_delimiter_offsets) const
  {
    if (!is_valid(idx)) { return 0; }

    auto const delim_size = d_delimiter.size_bytes();
    auto const d_str      = get_string(idx);
    auto const d_str_end  = d_str.data() + d_str.size_bytes();
    auto const base_ptr   = get_base_ptr() + delim_size - 1;

    auto const delimiters =
      cudf::device_span<int64_t const>(d_positions + d_delimiter_offsets[idx],
                                       d_delimiter_offsets[idx + 1] - d_delimiter_offsets[idx]);

    size_type token_count = 1;  // all strings will have at least one token
    auto last_pos         = !delimiters.empty() ? (delimiters[0] - delim_size) : 0L;
    for (auto d_pos : delimiters) {
      // delimiter must fit in string && overlapping delimiters are ignored
      if (((base_ptr + d_pos) < d_str_end) && ((d_pos - last_pos) >= delim_size)) {
        ++token_count;
        last_pos = d_pos;
      }
    }
    // number of tokens is capped to max_tokens
    return ((max_tokens > 0) && (token_count > max_tokens)) ? max_tokens : token_count;
  }

  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   * in which the delimiter resides
   *
   * Each token is placed in `d_all_tokens` so they align consecutively
   * with other tokens for the same output column.
   *
   * The actual token extraction is performed in the subclass process_tokens() function.
   *
   * @param idx Index of the string to tokenize
   * @param d_tokens_offsets Token offsets for each string
   * @param d_positions The beginning byte position of each delimiter
   * @param d_delimiter_offsets Offsets to d_positions to each delimiter set per string
   * @param d_all_tokens All output tokens for the strings column
   */
  __device__ void get_tokens(size_type idx,
                             cudf::detail::input_offsetalator const d_tokens_offsets,
                             int64_t const* d_positions,
                             cudf::detail::input_offsetalator d_delimiter_offsets,
                             string_index_pair* d_all_tokens) const
  {
    auto const d_tokens =  // this string's tokens output
      cudf::device_span<string_index_pair>(d_all_tokens + d_tokens_offsets[idx],
                                           d_tokens_offsets[idx + 1] - d_tokens_offsets[idx]);

    if (!is_valid(idx)) { return; }

    auto const d_str = get_string(idx);

    // max_tokens already included in token counts
    if (d_tokens.size() == 1) {
      d_tokens[0] = string_index_pair{d_str.data(), d_str.size_bytes()};
      return;
    }

    auto const delimiters =
      cudf::device_span<int64_t const>(d_positions + d_delimiter_offsets[idx],
                                       d_delimiter_offsets[idx + 1] - d_delimiter_offsets[idx]);

    auto& derived = static_cast<Derived const&>(*this);
    derived.process_tokens(d_str, delimiters, d_tokens);
  }

  base_split_tokenizer(column_device_view const& d_strings,
                       string_view const& d_delimiter,
                       size_type max_tokens)
    : d_strings(d_strings), d_delimiter(d_delimiter), max_tokens(max_tokens)
  {
  }

 protected:
  column_device_view const d_strings;  // strings to split
  string_view const d_delimiter;       // delimiter for split
  size_type max_tokens;                // maximum number of tokens to identify
};

/**
 * @brief The tokenizer functions for forward splitting
 */
struct split_tokenizer_fn : base_split_tokenizer<split_tokenizer_fn> {
  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   *
   * The tokens are processed from the beginning of each string ignoring overlapping
   * delimiters and honoring the `max_tokens` value.
   *
   * @param d_str String to tokenize
   * @param d_delimiters Positions of delimiters for this string
   * @param d_tokens Output vector to store tokens for this string
   */
  __device__ void process_tokens(string_view const d_str,
                                 device_span<int64_t const> d_delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr    = get_base_ptr();  // d_positions values based on this
    auto str_ptr           = d_str.data();
    auto const str_end     = str_ptr + d_str.size_bytes();  // end of the string
    auto const token_count = static_cast<size_type>(d_tokens.size());
    auto const delim_size  = d_delimiter.size_bytes();

    // build the index-pair of each token for this string
    size_type token_idx = 0;
    for (auto d_pos : d_delimiters) {
      auto const next_delim = base_ptr + d_pos;
      if (next_delim < str_ptr || ((next_delim + delim_size) > str_end)) { continue; }
      auto const end_ptr = (token_idx + 1 < token_count) ? next_delim : str_end;

      // store the token into the output vector
      d_tokens[token_idx++] =
        string_index_pair{str_ptr, static_cast<size_type>(thrust::distance(str_ptr, end_ptr))};

      // setup for next token
      str_ptr = end_ptr + delim_size;
    }
    // include anything leftover
    if (token_idx < token_count) {
      d_tokens[token_idx] =
        string_index_pair{str_ptr, static_cast<size_type>(thrust::distance(str_ptr, str_end))};
    }
  }

  split_tokenizer_fn(column_device_view const& d_strings,
                     string_view const& d_delimiter,
                     size_type max_tokens)
    : base_split_tokenizer(d_strings, d_delimiter, max_tokens)
  {
  }
};

/**
 * @brief The tokenizer functions for backwards splitting
 *
 * Same as split_tokenizer_fn except delimiters are searched from the end of each string.
 */
struct rsplit_tokenizer_fn : base_split_tokenizer<rsplit_tokenizer_fn> {
  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   *
   * The tokens are processed from the end of each string ignoring overlapping
   * delimiters and honoring the `max_tokens` value.
   *
   * @param d_str String to tokenize
   * @param d_delimiters Positions of delimiters for this string
   * @param d_tokens Output vector to store tokens for this string
   */
  __device__ void process_tokens(string_view const d_str,
                                 device_span<int64_t const> d_delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr    = get_base_ptr();  // d_positions values are based on this ptr
    auto const str_begin   = d_str.data();    // beginning of the string
    auto const token_count = static_cast<size_type>(d_tokens.size());
    auto const delim_count = static_cast<size_type>(d_delimiters.size());
    auto const delim_size  = d_delimiter.size_bytes();

    // build the index-pair of each token for this string
    auto str_ptr        = str_begin + d_str.size_bytes();
    size_type token_idx = 0;
    for (auto d = delim_count - 1; d >= 0; --d) {  // read right-to-left
      auto const prev_delim = base_ptr + d_delimiters[d] + delim_size;
      if (prev_delim > str_ptr || ((prev_delim - delim_size) < str_begin)) { continue; }
      auto const start_ptr = (token_idx + 1 < token_count) ? prev_delim : str_begin;

      // store the token into the output vector right-to-left
      d_tokens[token_count - token_idx - 1] =
        string_index_pair{start_ptr, static_cast<size_type>(thrust::distance(start_ptr, str_ptr))};

      // setup for next token
      str_ptr = start_ptr - delim_size;
      ++token_idx;
    }
    // include anything leftover (rightover?)
    if (token_idx < token_count) {
      d_tokens[0] =
        string_index_pair{str_begin, static_cast<size_type>(thrust::distance(str_begin, str_ptr))};
    }
  }

  rsplit_tokenizer_fn(column_device_view const& d_strings,
                      string_view const& d_delimiter,
                      size_type max_tokens)
    : base_split_tokenizer(d_strings, d_delimiter, max_tokens)
  {
  }
};

/**
 * @brief Create offsets for position values within a strings column
 *
 * The positions usually identify target sub-strings in the input column.
 * The offsets identify the set of positions for each string row.
 *
 * @param input Strings column corresponding to the input positions
 * @param positions Indices of target bytes within the input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned objects' device memory
 * @return Offsets of the position values for each string in input
 */
std::unique_ptr<column> create_offsets_from_positions(strings_column_view const& input,
                                                      device_span<int64_t const> const& positions,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

/**
 * @brief Helper function used by split/rsplit and split_record/rsplit_record
 *
 * This function returns all the token/split positions within the input column as processed by
 * the given tokenizer. It also returns the offsets for each set of tokens identified per string.
 *
 * @tparam Tokenizer Type of the tokenizer object
 *
 * @param input The input column of strings to split
 * @param tokenizer Object used for counting and identifying delimiters and tokens
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned objects' device memory
 * @return Token offsets and a vector of string indices
 */
template <typename Tokenizer>
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  Tokenizer tokenizer,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const strings_count = input.size();
  auto const chars_bytes =
    get_offset_value(input.offsets(), input.offset() + strings_count, stream) -
    get_offset_value(input.offsets(), input.offset(), stream);
  auto const d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // count the number of delimiters in the entire column
  auto const delimiter_count =
    thrust::count_if(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0),
                     thrust::counting_iterator<int64_t>(chars_bytes),
                     [tokenizer, d_offsets, chars_bytes] __device__(int64_t idx) {
                       return tokenizer.is_delimiter(idx, d_offsets, chars_bytes);
                     });
  // Create a vector of every delimiter position in the chars column.
  // These may include overlapping or otherwise out-of-bounds delimiters which
  // will be resolved during token processing.
  auto delimiter_positions = rmm::device_uvector<int64_t>(delimiter_count, stream);
  auto d_positions         = delimiter_positions.data();
  cudf::detail::copy_if_safe(
    thrust::counting_iterator<int64_t>(0),
    thrust::counting_iterator<int64_t>(chars_bytes),
    delimiter_positions.begin(),
    [tokenizer, d_offsets, chars_bytes] __device__(int64_t idx) {
      return tokenizer.is_delimiter(idx, d_offsets, chars_bytes);
    },
    stream);

  // create a vector of offsets to each string's delimiter set within delimiter_positions
  auto const delimiter_offsets =
    create_offsets_from_positions(input, delimiter_positions, stream, mr);
  auto const d_delimiter_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(delimiter_offsets->view());

  // compute the number of tokens per string
  auto token_counts = rmm::device_uvector<size_type>(strings_count, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    token_counts.begin(),
    [tokenizer, d_positions, d_delimiter_offsets] __device__(size_type idx) -> size_type {
      return tokenizer.count_tokens(idx, d_positions, d_delimiter_offsets);
    });

  // create offsets from the counts for return to the caller
  auto [offsets, total_tokens] =
    cudf::detail::make_offsets_child_column(token_counts.begin(), token_counts.end(), stream, mr);
  auto const d_tokens_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // build a vector of all the token positions for all the strings
  auto tokens   = rmm::device_uvector<string_index_pair>(total_tokens, stream);
  auto d_tokens = tokens.data();
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [tokenizer, d_tokens_offsets, d_positions, d_delimiter_offsets, d_tokens] __device__(
      size_type idx) {
      tokenizer.get_tokens(idx, d_tokens_offsets, d_positions, d_delimiter_offsets, d_tokens);
    });

  return std::make_pair(std::move(offsets), std::move(tokens));
}

}  // namespace cudf::strings::detail
