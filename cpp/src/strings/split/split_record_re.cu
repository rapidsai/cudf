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

#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
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
 * @brief Compute the number of tokens for the `idx'th` string element of `d_strings`.
 */
template <int stack_size>
struct token_counter_fn {
  column_device_view const d_strings;  // strings to split
  reprog_device prog;
  size_type const max_tokens;

  __device__ size_type operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return 0; }

    auto const d_str      = d_strings.element<string_view>(idx);
    size_type token_count = 0;

    int32_t begin = 0;
    int32_t end   = -1;
    while (token_count < max_tokens - 1) {
      if (prog.find<stack_size>(idx, d_str, begin, end) <= 0) { break; }
      token_count++;
      begin = end + (begin == end);
      end   = -1;
    }
    return token_count + 1;  // always at least one token
  }
};

/**
 * @brief Identify the tokens from the `idx'th` string element of `d_strings`.
 */
template <int stack_size>
struct token_reader_fn {
  column_device_view const d_strings;
  reprog_device prog;
  int32_t const* d_token_offsets;
  string_index_pair* d_tokens;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }

    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = d_token_offsets[idx + 1] - token_offset;
    auto d_result           = d_tokens + token_offset;
    auto const d_str        = d_strings.element<string_view>(idx);
    if (d_str.empty()) {
      // return empty string output for empty string input
      *d_result = string_index_pair{"", 0};
      return;
    }

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

}  // namespace

// The output is one list item per string
std::unique_ptr<column> split_record_re(
  strings_column_view const& input,
  std::string const& pattern,
  size_type maxsplit,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");

  auto const max_tokens    = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();
  auto const strings_count = input.size();

  auto d_prog = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto offsets = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto d_offsets = offsets->mutable_view().data<int32_t>();

  auto const begin = thrust::make_counting_iterator<size_type>(0);
  auto const end   = thrust::make_counting_iterator<size_type>(strings_count);

  // create offsets column by counting the number of tokens per string
  auto const regex_insts = d_prog->insts_counts();
  if (regex_insts <= RX_SMALL_INSTS) {
    token_counter_fn<RX_STACK_SMALL> counter{*d_strings, *d_prog, max_tokens};
    thrust::transform(rmm::exec_policy(stream), begin, end, d_offsets, counter);
  } else if (regex_insts <= RX_MEDIUM_INSTS) {
    token_counter_fn<RX_STACK_MEDIUM> counter{*d_strings, *d_prog, max_tokens};
    thrust::transform(rmm::exec_policy(stream), begin, end, d_offsets, counter);
  } else if (regex_insts <= RX_LARGE_INSTS) {
    token_counter_fn<RX_STACK_LARGE> counter{*d_strings, *d_prog, max_tokens};
    thrust::transform(rmm::exec_policy(stream), begin, end, d_offsets, counter);
  } else {
    token_counter_fn<RX_STACK_ANY> counter{*d_strings, *d_prog, max_tokens};
    thrust::transform(rmm::exec_policy(stream), begin, end, d_offsets, counter);
  }
  // convert counts into offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // last entry is the total number of tokens to be generated
  auto total_tokens = cudf::detail::get_value<int32_t>(offsets->view(), strings_count, stream);

  printf("instruction = %d\ntotal_tokens = %d\nbegin,end = %d,%d\n",
         regex_insts,
         total_tokens,
         *begin,
         *end);
  // split each string into an array of index-pair values
  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  if (regex_insts <= RX_SMALL_INSTS) {
    token_reader_fn<RX_STACK_SMALL> reader{*d_strings, *d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  } else if (regex_insts <= RX_MEDIUM_INSTS) {
    token_reader_fn<RX_STACK_MEDIUM> reader{*d_strings, *d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  } else if (regex_insts <= RX_LARGE_INSTS) {
    token_reader_fn<RX_STACK_LARGE> reader{*d_strings, *d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  } else {
    token_reader_fn<RX_STACK_ANY> reader{*d_strings, *d_prog, d_offsets, tokens.data()};
    thrust::for_each_n(rmm::exec_policy(stream), begin, strings_count, reader);
  }

  // convert the index-pairs into one big strings column
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

}  // namespace detail

// external APIs

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
