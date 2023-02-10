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
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Base class for delimiter-based tokenizers
 *
 * These are common methods used by both split and rsplit tokenizer functors.
 */
template <typename Derived>
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
   * @brief This counts the tokens for strings that contain delimiters
   *
   * Counting tokens is the same regardless if counting from the left
   * or from the right. This logic counts from the left which is simpler
   * logic. The count will be truncated appropriately to the max_tokens value.
   *
   * @param idx Index of input string
   * @param d_positions Start positions of all the delimiters
   * @param d_delimiter_offsets Offsets per string to delimiters in d_positions
   */
  __device__ size_type count_tokens(size_type idx,
                                    size_type const* d_positions,
                                    size_type const* d_delimiter_offsets) const
  {
    if (!is_valid(idx)) { return 0; }

    auto const delim_size = d_delimiter.size_bytes();
    auto const d_str      = get_string(idx);
    auto const d_str_end  = d_str.data() + d_str.size_bytes();
    auto const base_ptr   = get_base_ptr() + delim_size - 1;
    auto const delimiters =
      cudf::device_span<size_type const>(d_positions + d_delimiter_offsets[idx],
                                         d_delimiter_offsets[idx + 1] - d_delimiter_offsets[idx]);

    size_type token_count = 1;  // all strings will have at least one token
    size_type last_pos    = delimiters[0] - delim_size;
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
   * @param idx Index of the string to tokenize
   * @param d_tokens_offsets Token offsets for each string
   * @param d_positions The beginning byte position of each delimiter
   * @param d_delimiter_offsets Offsets to d_positions to each delimiter set per string
   * @param d_all_tokens All output tokens for the strings column
   */
  __device__ void get_tokens(size_type idx,
                             size_type const* d_tokens_offsets,
                             size_type const* d_positions,
                             size_type const* d_delimiter_offsets,
                             string_index_pair* d_all_tokens) const
  {
    auto const d_tokens =  // this string's tokens output
      cudf::device_span<string_index_pair>(d_all_tokens + d_tokens_offsets[idx],
                                           d_tokens_offsets[idx + 1] - d_tokens_offsets[idx]);

    if (!is_valid(idx)) {
      d_tokens[0] = string_index_pair{nullptr, 0};
      return;
    }

    auto const d_str = get_string(idx);

    // max_tokens already included in token counts
    if (d_tokens.size() == 1) {
      d_tokens[0] = string_index_pair{d_str.data(), d_str.size_bytes()};
      return;
    }

    auto const delimiters =
      cudf::device_span<size_type const>(d_positions + d_delimiter_offsets[idx],
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
  size_type max_tokens;
};

/**
 * @brief The tokenizer functions for split().
 *
 * The methods here count delimiters, tokens, and output token elements
 * for each string in a strings column.
 */
struct split_tokenizer_fn : base_split_tokenizer<split_tokenizer_fn> {
  /**
   * @brief Returns `true` if the byte at `idx` is the start of the delimiter
   *
   * @param idx Index of a byte in the chars column
   * @param d_offsets Offsets values to locate the chars ranges
   * @param chars_bytes Total number of characters to process
   * @return true if delimiter is found starting at position `idx`
   */
  __device__ bool is_delimiter(size_type idx,
                               size_type const* d_offsets,
                               size_type chars_bytes) const
  {
    auto const d_chars = get_base_ptr() + d_offsets[0];
    if (idx + d_delimiter.size_bytes() > chars_bytes) { return false; }
    return d_delimiter.compare(d_chars + idx, d_delimiter.size_bytes()) == 0;
  }

  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   * in which the delimiter resides
   *
   * @param d_str String to tokenize
   * @param d_delimiters Positions of delimiters for this string
   * @param d_tokens Output vector to store tokens for this string
   */
  __device__ void process_tokens(string_view const d_str,
                                 device_span<size_type const> d_delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr    = get_base_ptr();  // d_positions values based on this
    auto str_ptr           = d_str.data();
    auto const str_end     = str_ptr + d_str.size_bytes();  // end of the string
    auto const token_count = static_cast<size_type>(d_tokens.size());
    auto const delim_count = static_cast<size_type>(d_delimiters.size());

    // build the index-pair of each token for this string
    for (size_type t = 0; t < token_count; ++t) {
      auto next_delim = (t < delim_count)                 // bounds check for delims in last string
                          ? (base_ptr + d_delimiters[t])  // start of next delimiter
                          : str_end;                      // or end of this string

      auto eptr = (next_delim < str_end)        // make sure delimiter is inside this string
                      && (t + 1 < token_count)  // and this is not the last token
                    ? next_delim
                    : str_end;

      // store the token into the output vector
      d_tokens[t] =
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
 * @brief The tokenizer functions for split_record
 *
 * The methods here identify delimiters and output token elements
 * for each string in a strings column.
 *
 * Same as split_tokenizer_fn except delimiters are searched from the end of each string.
 */
struct rsplit_tokenizer_fn : base_split_tokenizer<rsplit_tokenizer_fn> {
  /**
   * @brief Returns `true` if the byte at `idx` is the end of the delimiter
   *
   * @param idx Index of a byte in the chars column
   * @param d_offsets Offsets values to locate the chars ranges
   * @return true if delimiter is found ending at position `idx`
   */
  __device__ bool is_delimiter(size_type idx, size_type const* d_offsets, size_type) const
  {
    auto const delim_length = d_delimiter.size_bytes();
    if (idx < delim_length - 1) { return false; }
    auto const d_chars = get_base_ptr() + d_offsets[0];
    return d_delimiter.compare(d_chars + idx - (delim_length - 1), delim_length) == 0;
  }

  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   * in which the delimiter resides
   *
   * The tokens are processed from the end of each string so the `max_tokens`
   * and any overlapping delimiters are honored correctly.
   *
   * @param d_str String to tokenize
   * @param d_delimiters Positions of delimiters for this string
   * @param d_tokens Output vector to store tokens for this string
   */
  __device__ void process_tokens(string_view const d_str,
                                 device_span<size_type const> d_delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr    = get_base_ptr();  // d_positions values are based on this ptr
    auto const str_begin   = d_str.data();    // beginning of the string
    auto const token_count = static_cast<size_type>(d_tokens.size());
    auto const delim_count = static_cast<size_type>(d_delimiters.size());

    // build the index-pair of each token for this string
    auto str_ptr = str_begin + d_str.size_bytes();
    for (size_type t = 0; t < token_count; ++t) {
      auto prev_delim =
        (t < delim_count)                                       // boundary check;
          ? (base_ptr + d_delimiters[delim_count - 1 - t] + 1)  // end of prev delimiter
          : str_begin;                                          // or the start of this string

      auto sptr = (prev_delim > str_begin)      // make sure delimiter is inside the string
                      && (t + 1 < token_count)  // and this is not the last token
                    ? prev_delim
                    : str_begin;

      // store the token into the output -- building the array backwards
      d_tokens[(token_count - 1 - t)] =
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

template <typename Tokenizer>
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
  auto delimiter_positions = rmm::device_uvector<size_type>(delimiter_count, stream);
  auto d_positions         = delimiter_positions.data();
  auto const copy_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(chars_bytes),
                    delimiter_positions.begin(),
                    [tokenizer, d_offsets, chars_bytes] __device__(size_type idx) {
                      return tokenizer.is_delimiter(idx, d_offsets, chars_bytes);
                    });

  // create a vector of offsets to each string's delimiter set within delimiter_positions
  auto const delimiter_offsets = [&] {
    // first, create a vector of string indices for each delimiter
    auto string_indices = rmm::device_uvector<size_type>(delimiter_count, stream);
    thrust::upper_bound(rmm::exec_policy(stream),
                        d_offsets,
                        d_offsets + strings_count,
                        delimiter_positions.begin(),
                        copy_end,
                        string_indices.begin());

    // compute delimiter offsets per string
    auto delimiter_offsets   = rmm::device_uvector<size_type>(strings_count + 1, stream);
    auto d_delimiter_offsets = delimiter_offsets.data();

    // memset required to zero-out any null-entries or strings with no delimiters
    cudaMemsetAsync(d_delimiter_offsets, 0, delimiter_offsets.size() * sizeof(size_type), stream);

    // next, count the number of delimiters per string
    auto d_string_indices = string_indices.data();  // identifies strings with delimiters only
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       delimiter_count,
                       [d_string_indices, d_delimiter_offsets] __device__(size_type idx) {
                         auto const str_idx = d_string_indices[idx] - 1;
                         atomicAdd(d_delimiter_offsets + str_idx, 1);
                       });
    // finally, convert the counts into offsets
    thrust::exclusive_scan(rmm::exec_policy(stream),
                           delimiter_offsets.begin(),
                           delimiter_offsets.end(),
                           delimiter_offsets.begin());
    return delimiter_offsets;
  }();
  auto const d_delimiter_offsets = delimiter_offsets.data();

  auto token_counts = rmm::device_uvector<size_type>(strings_count, stream);

  // compute the number of tokens per string
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    token_counts.begin(),
    [tokenizer, d_positions, d_delimiter_offsets] __device__(size_type idx) -> size_type {
      return tokenizer.count_tokens(idx, d_positions, d_delimiter_offsets);
    });

  auto offsets = std::get<0>(
    cudf::detail::make_offsets_child_column(token_counts.begin(), token_counts.end(), stream, mr));
  auto const total_tokens =
    cudf::detail::get_value<size_type>(offsets->view(), strings_count, stream);
  auto const d_tokens_offsets = offsets->view().data<size_type>();

  // create working area to hold all token positions
  auto tokens   = rmm::device_uvector<string_index_pair>(total_tokens, stream);
  auto d_tokens = tokens.data();
  // fill in the token objects
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [tokenizer, d_tokens_offsets, d_positions, d_delimiter_offsets, d_tokens] __device__(
      size_type idx) {
      tokenizer.get_tokens(idx, d_tokens_offsets, d_positions, d_delimiter_offsets, d_tokens);
    });

  // build strings column from tokens
  auto strings_child = make_strings_column(tokens.begin(), tokens.end(), stream, mr);
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(strings_child),
                           input.null_count(),
                           copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);

  // auto const columns_count = thrust::reduce(
  //  rmm::exec_policy(stream), token_counts.begin(), token_counts.end(), 0, thrust::maximum{});
  // std::cout << "count = " << columns_count << "\n";
  // size_type token_index = 10;
  // auto itr = cudf::detail::make_counting_transform_iterator(
  //  0, [d_tokens, d_tokens_offsets, token_index, columns_count] __device__(size_type idx) {
  //    auto offset      = d_tokens_offsets[idx];
  //    auto token_count = d_tokens_offsets[idx + 1] - offset;
  //    return (token_index < token_count) ? d_tokens[offset + token_index]
  //                                       : string_index_pair{nullptr, 0};
  //  });
  // return make_strings_column(itr, itr + strings_count, stream, mr);
}

namespace {

enum class Dir { FORWARD, BACKWARD };

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
    if constexpr (dir == Dir::FORWARD) {
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
