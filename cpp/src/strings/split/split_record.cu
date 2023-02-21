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

#include "split.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

template <typename Tokenizer>
std::unique_ptr<column> split_record_fn(strings_column_view const& input,
                                        Tokenizer tokenizer,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::LIST); }
  if (input.size() == input.null_count()) {
    auto offsets = std::make_unique<column>(input.offsets(), stream, mr);
    auto results = make_empty_column(type_id::STRING);
    return make_lists_column(input.size(),
                             std::move(offsets),
                             std::move(results),
                             input.null_count(),
                             copy_bitmask(input.parent(), stream, mr),
                             stream,
                             mr);
  }

  // builds the offsets and the vector of all tokens
  auto [offsets, tokens] = split_helper(input, tokenizer, stream, mr);

  // build a strings column from the tokens
  auto strings_child = make_strings_column(tokens.begin(), tokens.end(), stream, mr);

  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(strings_child),
                           input.null_count(),
                           copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

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
  auto total_tokens = cudf::detail::get_value<size_type>(offsets->view(), strings_count, stream);
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
