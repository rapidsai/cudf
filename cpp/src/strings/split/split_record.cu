/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

enum class Dir { FORWARD, BACKWARD };

/**
 * @brief Base class for delimiter-based tokenizers.
 *
 * These are common methods used by both split and rsplit tokenizer functors.
 */
struct base_split_tokenizer {
  __device__ const char* get_base_ptr() const
  {
    return d_strings.child(strings_column_view::chars_column_index).data<char>();
  }

  __device__ string_view const get_string(size_type idx) const
  {
    return d_strings.element<string_view>(idx);
  }

  __device__ bool is_valid(size_type idx) const { return d_strings.is_valid(idx); }

  /**
   * @brief Initialize token elements for all strings.
   *
   * The process_tokens() only handles creating tokens for strings that contain
   * delimiters. This function will initialize the output tokens for all
   * strings by assigning null entries for null and empty strings and the
   * string itself for strings with no delimiters.
   *
   * The tokens are placed in output order so that all tokens for each output
   * column are stored consecutively in `d_all_tokens`.
   *
   * @param idx Index of string in column
   * @param d_tokens_offsets Offsets to each string's tokens output
   * @param d_all_tokens Tokens vector for all strings
   */
  __device__ void init_tokens(size_type idx,
                              size_type const* d_tokens_offsets,
                              string_index_pair* d_all_tokens) const
  {
    auto const d_tokens = d_all_tokens + d_tokens_offsets[idx];
    if (is_valid(idx)) {
      auto const d_str = get_string(idx);
      // initialize pair to current string
      *d_tokens = string_index_pair{d_str.data(), d_str.size_bytes()};
    } else {
      *d_tokens = string_index_pair{nullptr, 0};
    }
  }

  /**
   * @brief This counts the tokens for strings that contain delimiters
   *
   * Counting tokens is the same regardless if counting from the left
   * or from the right. This logic counts from the left which is simpler
   * logic.
   *
   * @param idx Index of a delimiter position
   * @param d_positions Start positions of all the delimiters
   * @param positions_count The number of delimiters
   * @param d_indexes Indices of the strings for each delimiter
   * @param d_counts The token counts for all the strings
   */

  __device__ void count_tokens(size_type idx,
                               size_type const* d_positions,
                               size_type positions_count,
                               size_type const* d_indexes,
                               size_type* d_counts) const
  {
    auto const str_idx = d_indexes[idx] - 1;  // adjust for upper_bound
    if ((idx > 0) && (d_indexes[idx - 1] == (str_idx + 1))) return;

    auto const delim_length = d_delimiter.size_bytes();
    auto const d_str        = get_string(str_idx);
    auto const base_ptr     = get_base_ptr();
    size_type delim_count   = 0;  // re-count delimiters to compute the token-count
    size_type last_pos      = d_positions[idx] - delim_length;
    while ((idx < positions_count) && (d_indexes[idx] == (str_idx + 1))) {
      // make sure the whole delimiter is inside the string before counting it
      auto d_pos = d_positions[idx];
      if (((base_ptr + d_pos + delim_length - 1) < (d_str.data() + d_str.size_bytes())) &&
          ((d_pos - last_pos) >= delim_length)) {
        ++delim_count;     // only count if the delimiter fits
        last_pos = d_pos;  // overlapping delimiters are ignored too
      }
      ++idx;
    }
    // the number of tokens is delim_count+1 but capped to max_tokens
    d_counts[str_idx] =
      ((max_tokens > 0) && (delim_count + 1 > max_tokens)) ? max_tokens : delim_count + 1;
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
  size_type max_tokens;
};

/**
 * @brief The tokenizer functions for split().
 *
 * The methods here count delimiters, tokens, and output token elements
 * for each string in a strings column.
 */
struct split_tokenizer_fn : base_split_tokenizer {
  /**
   * @brief Returns `true` if the byte at `idx` is the start of the delimiter.
   *
   * @param idx Index of a byte in the chars column.
   * @param d_offsets Offsets values to locate the chars ranges.
   * @param chars_bytes Total number of characters to process.
   * @return true if delimiter is found starting at position `idx`
   */
  __device__ bool is_delimiter(size_type idx,
                               size_type const* d_offsets,
                               size_type chars_bytes) const
  {
    auto d_chars = get_base_ptr() + d_offsets[0];
    if (idx + d_delimiter.size_bytes() > chars_bytes) return false;
    return d_delimiter.compare(d_chars + idx, d_delimiter.size_bytes()) == 0;
  }

  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   * in which the delimiter resides.
   *
   * Each token is placed in `d_all_tokens` so they align consecutively
   * with other tokens for the same output column.
   * That is, `d_tokens[col * strings_count + string_index]` is the token at column `col`
   * for string at `string_index`.
   *
   * @param idx Index of the delimiter in the chars column
   * @param d_tokens_offsets Token offsets for each string
   * @param d_positions The beginning byte position of each delimiter
   * @param positions_count Number of delimiters
   * @param d_indexes Indices of the strings for each delimiter
   * @param d_all_tokens All output tokens for the strings column
   */
  __device__ void process_tokens(size_type idx,
                                 size_type const* d_tokens_offsets,
                                 size_type const* d_positions,
                                 size_type positions_count,
                                 size_type const* d_indexes,
                                 string_index_pair* d_all_tokens) const
  {
    auto const str_idx = d_indexes[idx] - 1;  // adjust for upper_bound
    if ((idx > 0) && d_indexes[idx - 1] == (str_idx + 1)) return;

    // max_tokens already included in token-count
    auto const token_count = d_tokens_offsets[str_idx + 1] - d_tokens_offsets[str_idx];
    auto const base_ptr    = get_base_ptr();  // d_positions values are based on this ptr
    auto const d_tokens = d_all_tokens + d_tokens_offsets[str_idx];  // this string's tokens output
    // this string
    auto const d_str   = get_string(str_idx);
    auto str_ptr       = d_str.data();                  // beginning of the string
    auto const str_end = str_ptr + d_str.size_bytes();  // end of the string
    // build the index-pair of each token for this string
    for (size_type col = 0; col < token_count; ++col) {
      auto next_delim = ((idx + col) < positions_count)  // boundary check for delims in last string
                          ? (base_ptr + d_positions[idx + col])  // start of next delimiter
                          : str_end;                             // or end of this string

      auto eptr = (next_delim < str_end)          // make sure delimiter is inside this string
                      && (col + 1 < token_count)  // and this is not the last token
                    ? next_delim
                    : str_end;
      // store the token into the output vector
      d_tokens[col] =
        string_index_pair{str_ptr, static_cast<size_type>(thrust::distance(str_ptr, eptr))};
      // setup for next token
      str_ptr = eptr + d_delimiter.size_bytes();
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
 * @brief The tokenizer functions for split().
 *
 * The methods here count delimiters, tokens, and output token elements
 * for each string in a strings column.
 *
 * Same as split_tokenizer_fn except tokens are counted from the end of each string.
 */
struct rsplit_tokenizer_fn : base_split_tokenizer {
  /**
   * @brief Returns `true` if the byte at `idx` is the end of the delimiter.
   *
   * @param idx Index of a byte in the chars column.
   * @param d_offsets Offsets values to locate the chars ranges.
   * @return true if delimiter is found ending at position `idx`
   */
  __device__ bool is_delimiter(size_type idx, size_type const* d_offsets, size_type) const
  {
    auto delim_length = d_delimiter.size_bytes();
    if (idx < delim_length - 1) return false;
    auto d_chars = get_base_ptr() + d_offsets[0];
    return d_delimiter.compare(d_chars + idx - (delim_length - 1), delim_length) == 0;
  }

  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   * in which the delimiter resides.
   *
   * The tokens are processed from the end of each string so the `max_tokens`
   * is honored correctly.
   *
   * Each token is placed in `d_all_tokens` so they align consecutively
   * with other tokens for the same output column.
   * That is, `d_tokens[col * strings_count + string_index]` is the token at column `col`
   * for string at `string_index`.
   *
   * @param idx Index of the delimiter in the chars column
   * @param d_tokens_offsets Tokens offsets for each string
   * @param d_positions The ending byte position of each delimiter
   * @param positions_count Number of delimiters
   * @param d_indexes Indices of the strings for each delimiter
   * @param d_all_tokens All output tokens for the strings column
   */
  __device__ void process_tokens(size_type idx,
                                 size_type const* d_tokens_offsets,
                                 size_type const* d_positions,
                                 size_type positions_count,
                                 size_type const* d_indexes,
                                 string_index_pair* d_all_tokens) const
  {
    auto const str_idx = d_indexes[idx] - 1;  // adjust for upper_bound
    if ((idx + 1 < positions_count) && d_indexes[idx + 1] == (str_idx + 1)) return;

    // max_tokens already included in token-counts
    auto const token_count = d_tokens_offsets[str_idx + 1] - d_tokens_offsets[str_idx];
    auto const base_ptr    = get_base_ptr();  // d_positions values are based on this ptr
    // this string's tokens output
    auto const d_tokens = d_all_tokens + d_tokens_offsets[str_idx];
    // this string
    auto const d_str     = get_string(str_idx);
    auto const str_begin = d_str.data();                    // beginning of the string
    auto str_ptr         = str_begin + d_str.size_bytes();  // end of the string
    // build the index-pair of each token for this string
    for (size_type col = 0; col < token_count; ++col) {
      auto prev_delim = (idx >= col)  // boundary check for delims in first string
                          ? (base_ptr + d_positions[idx - col] + 1)  // end of prev delimiter
                          : str_begin;                               // or the start of this string

      auto sptr = (prev_delim > str_begin)        // make sure delimiter is inside the string
                      && (col + 1 < token_count)  // and this is not the last token
                    ? prev_delim
                    : str_begin;
      // store the token into the output -- building the array backwards
      d_tokens[(token_count - 1 - col)] =
        string_index_pair{sptr, static_cast<size_type>(str_ptr - sptr)};
      // setup for next/prev token
      str_ptr = sptr - d_delimiter.size_bytes();
    }
  }

  rsplit_tokenizer_fn(column_device_view const& d_strings,
                      string_view const& d_delimiter,
                      size_type max_tokens)
    : base_split_tokenizer(d_strings, d_delimiter, max_tokens)
  {
  }
};

}  // namespace

template <typename Tokenizer>  //, typename TokenCounter, typename TokenReader>
std::unique_ptr<column> split_record_fn(strings_column_view const& input,
                                        Tokenizer tokenizer,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::LIST); }
  if (strings_count == input.null_count()) {
    auto offsets = std::make_unique<column>(input.offsets(), stream, mr);
    auto results = std::make_unique<column>(input.parent(), stream, mr);
    return make_lists_column(strings_count,
                             std::move(offsets),
                             std::move(results),
                             input.null_count(),
                             copy_bitmask(input.parent(), stream, mr),
                             stream,
                             mr);
  }

  auto const chars_bytes =
    cudf::detail::get_value<size_type>(input.offsets(), input.offset() + strings_count, stream) -
    cudf::detail::get_value<size_type>(input.offsets(), input.offset(), stream);

  auto d_offsets = input.offsets_begin();

  // count the number of delimiters in the entire column
  auto const delimiter_count =
    thrust::count_if(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(chars_bytes),
                     [tokenizer, d_offsets, chars_bytes] __device__(size_type idx) {
                       return tokenizer.is_delimiter(idx, d_offsets, chars_bytes);
                     });

  // create vector of every delimiter position in the chars column
  rmm::device_uvector<size_type> delimiter_positions(delimiter_count, stream);
  auto d_positions = delimiter_positions.data();
  auto copy_end    = thrust::copy_if(rmm::exec_policy(stream),
                                  thrust::make_counting_iterator<size_type>(0),
                                  thrust::make_counting_iterator<size_type>(chars_bytes),
                                  delimiter_positions.begin(),
                                  [tokenizer, d_offsets, chars_bytes] __device__(size_type idx) {
                                    return tokenizer.is_delimiter(idx, d_offsets, chars_bytes);
                                  });

  // create vector of string indices for each delimiter
  rmm::device_uvector<size_type> string_indices(delimiter_count, stream);  // these will
  auto d_string_indices = string_indices.data();  // be strings that only contain delimiters
  thrust::upper_bound(rmm::exec_policy(stream),
                      d_offsets,
                      d_offsets + strings_count,
                      delimiter_positions.begin(),
                      copy_end,
                      string_indices.begin());

  // compute the number of tokens per string
  rmm::device_uvector<size_type> token_counts(strings_count, stream);
  auto d_token_counts = token_counts.data();
  // first, initialize token counts for strings without delimiters in them
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_token_counts,
                    [tokenizer] __device__(size_type idx) {
                      // null are 0, all others 1
                      return static_cast<size_type>(tokenizer.is_valid(idx));
                    });

  // now compute the number of tokens in each string
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    delimiter_count,
    [tokenizer, d_positions, delimiter_count, d_string_indices, d_token_counts] __device__(
      size_type idx) {
      tokenizer.count_tokens(idx, d_positions, delimiter_count, d_string_indices, d_token_counts);
    });

  auto offsets      = std::get<0>(cudf::detail::make_offsets_child_column(
    d_token_counts, d_token_counts + strings_count, stream, mr));
  auto total_tokens = cudf::detail::get_value<size_type>(offsets->view(), strings_count, stream);

  auto d_tokens_offsets = offsets->view().data<size_type>();

  // create working area to hold all token positions
  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  string_index_pair* d_tokens = tokens.data();
  // initialize the token positions
  // -- accounts for nulls, empty, and strings with no delimiter in them
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [tokenizer, d_tokens_offsets, d_tokens] __device__(size_type idx) {
                       tokenizer.init_tokens(idx, d_tokens_offsets, d_tokens);
                     });

  // get the positions for every token using the delimiter positions
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    delimiter_count,
    [tokenizer,
     d_tokens_offsets,
     d_positions,
     delimiter_count,
     d_string_indices,
     d_tokens] __device__(size_type idx) {
      tokenizer.process_tokens(
        idx, d_tokens_offsets, d_positions, delimiter_count, d_string_indices, d_tokens);
    });

  auto strings_output = make_strings_column(tokens.begin(), tokens.end(), stream, mr);
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(strings_output),
                           input.null_count(),
                           copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

namespace {

/**
 * @brief Compute the number of tokens for the `idx'th` string element of `d_strings`.
 */
struct whitespace_token_counter_fn {
  column_device_view const d_strings;  // strings to split
  size_type const max_tokens = std::numeric_limits<size_type>::max();

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }

    auto const d_str        = d_strings.element<string_view>(idx);
    size_type token_count   = 0;
    auto spaces             = true;
    auto reached_max_tokens = false;
    for (auto ch : d_str) {
      if (spaces != (ch <= ' ')) {
        if (!spaces) {
          if (token_count < max_tokens - 1) {
            token_count++;
          } else {
            reached_max_tokens = true;
            break;
          }
        }
        spaces = !spaces;
      }
    }
    // pandas.Series.str.split("") returns 0 tokens.
    if (reached_max_tokens || !spaces) token_count++;
    return token_count;
  }
};

/**
 * @brief Identify the tokens from the `idx'th` string element of `d_strings`.
 */
template <Dir dir>
struct whitespace_token_reader_fn {
  column_device_view const d_strings;  // strings to split
  size_type const max_tokens{};
  int32_t* d_token_offsets{};
  string_index_pair* d_tokens{};

  __device__ void operator()(size_type idx)
  {
    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = d_token_offsets[idx + 1] - token_offset;
    if (token_count == 0) { return; }
    auto d_result = d_tokens + token_offset;

    auto const d_str = d_strings.element<string_view>(idx);
    whitespace_string_tokenizer tokenizer(d_str, dir != Dir::FORWARD);
    size_type token_idx = 0;
    position_pair token{0, 0};
    if (dir == Dir::FORWARD) {
      while (tokenizer.next_token() && (token_idx < token_count)) {
        token = tokenizer.get_token();
        d_result[token_idx++] =
          string_index_pair{d_str.data() + token.first, token.second - token.first};
      }
      --token_idx;
      token.second = d_str.size_bytes() - token.first;
    } else {
      while (tokenizer.prev_token() && (token_idx < token_count)) {
        token = tokenizer.get_token();
        d_result[token_count - 1 - token_idx] =
          string_index_pair{d_str.data() + token.first, token.second - token.first};
        ++token_idx;
      }
      token_idx   = token_count - token_idx;  // token_count - 1 - (token_idx-1)
      token.first = 0;
    }
    // reset last token only if we hit the max
    if (token_count == max_tokens)
      d_result[token_idx] = string_index_pair{d_str.data() + token.first, token.second};
  }
};

}  // namespace

// The output is one list item per string
template <typename TokenCounter, typename TokenReader>
std::unique_ptr<column> whitespace_split_record_fn(strings_column_view const& strings,
                                                   TokenCounter counter,
                                                   TokenReader reader,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  // create offsets column by counting the number of tokens per string
  auto strings_count = strings.size();
  auto offsets       = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto d_offsets = offsets->mutable_view().data<int32_t>();
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_offsets,
                    counter);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // last entry is the total number of tokens to be generated
  auto total_tokens = cudf::detail::get_value<int32_t>(offsets->view(), strings_count, stream);
  // split each string into an array of index-pair values
  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  reader.d_token_offsets = d_offsets;
  reader.d_tokens        = tokens.data();
  thrust::for_each_n(
    rmm::exec_policy(stream), thrust::make_counting_iterator<size_type>(0), strings_count, reader);
  // convert the index-pairs into one big strings column
  auto strings_output = make_strings_column(tokens.begin(), tokens.end(), stream, mr);
  // create a lists column using the offsets and the strings columns
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(strings_output),
                           strings.null_count(),
                           copy_bitmask(strings.parent(), stream, mr),
                           stream,
                           mr);
}

template <Dir dir>
std::unique_ptr<column> split_record(strings_column_view const& strings,
                                     string_scalar const& delimiter,
                                     size_type maxsplit,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  // makes consistent with Pandas
  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto d_strings_column_ptr = column_device_view::create(strings.parent(), stream);
  if (delimiter.size() == 0) {
    return whitespace_split_record_fn(
      strings,
      whitespace_token_counter_fn{*d_strings_column_ptr, max_tokens},
      whitespace_token_reader_fn<dir>{*d_strings_column_ptr, max_tokens},
      stream,
      mr);
  } else {
    string_view d_delimiter(delimiter.data(), delimiter.size());
    if (dir == Dir::FORWARD) {
      return split_record_fn(
        strings, split_tokenizer_fn{*d_strings_column_ptr, d_delimiter, max_tokens}, stream, mr);
    } else {
      return split_record_fn(
        strings, rsplit_tokenizer_fn{*d_strings_column_ptr, d_delimiter, max_tokens}, stream, mr);
    }
  }
}

}  // namespace detail

// external APIs

std::unique_ptr<column> split_record(strings_column_view const& strings,
                                     string_scalar const& delimiter,
                                     size_type maxsplit,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_record<detail::Dir::FORWARD>(
    strings, delimiter, maxsplit, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> rsplit_record(strings_column_view const& strings,
                                      string_scalar const& delimiter,
                                      size_type maxsplit,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_record<detail::Dir::BACKWARD>(
    strings, delimiter, maxsplit, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
