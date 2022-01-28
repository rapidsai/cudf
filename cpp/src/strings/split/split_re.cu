/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <strings/count_matches.hpp>
#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

using string_index_pair = thrust::pair<const char*, size_type>;

namespace {

/**
 * @brief Convert match counts to token counts.
 *
 * The matches are the delimiters and the tokens are what is left:
 * `token1, delimiter, token2, delimiter, token3, etc`
 * Usually `token_count = match_count + 1` even with empty strings.
 * However, we need to account for the max_tokens and null rows.
 */
struct match_to_token_count_fn {
  column_device_view const d_strings;
  size_type const* d_counts;
  size_type const max_tokens;

  __device__ size_type operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const match_count = d_counts[idx];
    return std::min(match_count, max_tokens) + 1;
  }
};

/**
 * @brief Identify the tokens from the `idx'th` string element of `d_strings`.
 */
template <int stack_size>
struct token_reader_fn {
  column_device_view const d_strings;
  reprog_device prog;
  offset_type const* d_token_offsets;
  string_index_pair* d_tokens;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }

    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = d_token_offsets[idx + 1] - token_offset;
    auto d_result           = d_tokens + token_offset;
    auto const d_str        = d_strings.element<string_view>(idx);

    size_type token_idx = 0;
    size_type begin     = 0;
    size_type end       = d_str.length();
    size_type last_pos  = 0;
    while (token_idx < token_count - 1) {
      if (prog.find<stack_size>(idx, d_str, begin, end) <= 0) { break; }

      auto const start_pos = d_str.byte_offset(begin);
      auto const end_pos   = d_str.byte_offset(end);
      d_result[token_idx]  = string_index_pair{d_str.data() + last_pos, start_pos - last_pos};

      begin = end + (begin == end);
      end   = d_str.length();
      token_idx++;
      last_pos = end_pos;
    }

    // set last token to remainder of the string
    if (last_pos <= d_str.size_bytes()) {
      d_result[token_idx] =
        string_index_pair{d_str.data() + last_pos, d_str.size_bytes() - last_pos};
    }
  }
};

struct tokens_transform_fn {
  column_device_view const d_strings;
  string_index_pair const* d_tokens;
  offset_type const* d_token_offsets;
  size_type const column_index;

  __device__ string_index_pair operator()(size_type idx) const
  {
    auto const offset      = d_token_offsets[idx];
    auto const token_count = d_token_offsets[idx + 1] - offset;
    if (d_strings.is_null(idx)) { return string_index_pair{nullptr, 0}; }
    if (column_index > token_count - 1) { return string_index_pair{nullptr, 0}; }
    return d_tokens[offset + column_index];
  }
};

/**
 * @brief Call regex to split each input string into tokens.
 *
 * This will also convert the `offsets` values from counts to offsets.
 *
 * @param d_strings Strings to split
 * @param d_prog Regex to evaluate against each string
 * @param max_tokens The maximum number of tokens for each split.
 * @param offsets The number of matches on input.
 *                The offsets for each token in each string on output.
 * @param stream CUDA stream used for kernel launches.
 */
rmm::device_uvector<string_index_pair> split_utility(column_device_view const& d_strings,
                                                     reprog_device& d_prog,
                                                     size_type max_tokens,
                                                     mutable_column_view& offsets,
                                                     rmm::cuda_stream_view stream)
{
  auto d_offsets           = offsets.data<offset_type>();
  auto const strings_count = d_strings.size();

  auto const begin = thrust::make_counting_iterator<size_type>(0);
  auto const end   = thrust::make_counting_iterator<size_type>(strings_count);

  // convert match counts to tokens
  match_to_token_count_fn match_fn{d_strings, d_offsets, max_tokens};
  thrust::transform(rmm::exec_policy(stream), begin, end, d_offsets, match_fn);

  // convert counts into offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         offsets.begin<offset_type>(),
                         offsets.end<offset_type>(),
                         offsets.begin<offset_type>());

  // the last entry is the total number of tokens to be generated
  auto const total_tokens = cudf::detail::get_value<offset_type>(offsets, strings_count, stream);

  // generate tokens for each string
  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  auto const regex_insts = d_prog.insts_counts();
  if (regex_insts <= RX_SMALL_INSTS) {
    token_reader_fn<RX_STACK_SMALL> reader{d_strings, d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  } else if (regex_insts <= RX_MEDIUM_INSTS) {
    token_reader_fn<RX_STACK_MEDIUM> reader{d_strings, d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  } else if (regex_insts <= RX_LARGE_INSTS) {
    token_reader_fn<RX_STACK_LARGE> reader{d_strings, d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  } else {
    token_reader_fn<RX_STACK_ANY> reader{d_strings, d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  }

  return tokens;
}

}  // namespace

// The output is one list item per string
std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        std::string const& pattern,
                                        size_type maxsplit,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");

  auto const max_tokens    = maxsplit > 0 ? maxsplit : std::numeric_limits<size_type>::max();
  auto const strings_count = input.size();

  auto d_prog = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto offsets      = count_matches(*d_strings, *d_prog, stream, mr);
  auto offsets_view = offsets->mutable_view();

  // get split tokens from the input column
  auto tokens = split_utility(*d_strings, *d_prog, max_tokens, offsets_view, stream);

  // convert the tokens into one big strings column
  auto strings_output = make_strings_column(tokens.begin(), tokens.end(), stream, mr);

  // create a lists column using the offsets and the strings columns
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(strings_output),
                           input.null_count(),
                           copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

std::unique_ptr<table> split_re(strings_column_view const& input,
                                std::string const& pattern,
                                size_type maxsplit,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");

  auto const max_tokens    = maxsplit > 0 ? maxsplit : std::numeric_limits<size_type>::max();
  auto const strings_count = input.size();

  std::vector<std::unique_ptr<column>> results;
  if (strings_count == 0) {
    results.push_back(make_empty_column(type_id::STRING));
    return std::make_unique<table>(std::move(results));
  }

  auto d_prog = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto offsets = count_matches(*d_strings, *d_prog, stream, rmm::mr::get_current_device_resource());
  auto offsets_view = offsets->mutable_view();

  // get split tokens from the input column
  auto tokens = split_utility(*d_strings, *d_prog, max_tokens, offsets_view, stream);

  // the columns_count is the maximum number of tokens for any string in the input column
  auto const begin = thrust::make_counting_iterator<size_type>(0);
  auto const end   = thrust::make_counting_iterator<size_type>(strings_count);
  auto d_offsets   = offsets_view.data<offset_type>();
  auto size_lambda = [d_offsets] __device__(auto const idx) -> size_type {
    return d_offsets[idx + 1] - d_offsets[idx];
  };
  auto const columns_count = thrust::transform_reduce(
    rmm::exec_policy(stream), begin, end, size_lambda, 0, thrust::maximum<size_type>{});

  // boundary case: if no columns, return one all-null column (custrings issue #119)
  if (columns_count == 0) {
    results.push_back(std::make_unique<column>(
      data_type{type_id::STRING},
      strings_count,
      rmm::device_buffer{0, stream, mr},  // no data
      cudf::detail::create_null_mask(strings_count, mask_state::ALL_NULL, stream, mr),
      strings_count));
    return std::make_unique<table>(std::move(results));
  }

  // convert the tokens into multiple strings columns
  auto make_strings_lambda = [&](size_type column_index) {
    // returns appropriate token for each row/column
    auto indices_itr = cudf::detail::make_counting_transform_iterator(
      0, tokens_transform_fn{*d_strings, tokens.data(), d_offsets, column_index});
    return make_strings_column(indices_itr, indices_itr + strings_count, stream, mr);
  };
  // create each column of tokens
  results.resize(columns_count);
  std::transform(thrust::make_counting_iterator<size_type>(0),
                 thrust::make_counting_iterator<size_type>(columns_count),
                 results.begin(),
                 make_strings_lambda);

  return std::make_unique<table>(std::move(results));
}

}  // namespace detail

// external APIs

std::unique_ptr<table> split_re(strings_column_view const& input,
                                std::string const& pattern,
                                size_type maxsplit,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_re(input, pattern, maxsplit, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        std::string const& pattern,
                                        size_type maxsplit,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_record_re(input, pattern, maxsplit, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
