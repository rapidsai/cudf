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
#include <strings/regex/utilities.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

using string_index_pair = thrust::pair<const char*, size_type>;

enum class split_direction {
  FORWARD,  ///< for split logic
  BACKWARD  ///< for rsplit logic
};

/**
 * @brief Identify the tokens from the `idx'th` string element of `d_strings`.
 *
 * Each string's tokens are stored in the `d_tokens` vector.
 * The `d_token_offsets` specifies the output position within `d_tokens`
 * for each string.
 */
struct token_reader_fn {
  column_device_view const d_strings;
  split_direction const direction;
  offset_type const* d_token_offsets;
  string_index_pair* d_tokens;

  __device__ void operator()(size_type const idx, reprog_device const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = d_token_offsets[idx + 1] - token_offset;
    auto const d_result     = d_tokens + token_offset;  // store tokens here

    size_type token_idx = 0;
    size_type begin     = 0;  // characters
    size_type end       = nchars;
    size_type last_pos  = 0;  // bytes
    while (prog.find(prog_idx, d_str, begin, end) > 0) {
      // get the token (characters just before this match)
      auto const token =
        string_index_pair{d_str.data() + last_pos, d_str.byte_offset(begin) - last_pos};
      // store it if we have space
      if (token_idx < token_count - 1) {
        d_result[token_idx++] = token;
      } else {
        if (direction == split_direction::FORWARD) { break; }  // we are done
        for (auto l = 0; l < token_idx - 1; ++l) {
          d_result[l] = d_result[l + 1];  // shift left
        }
        d_result[token_idx - 1] = token;
      }
      // setup for next match
      last_pos = d_str.byte_offset(end);
      begin    = end + (begin == end);
      end      = nchars;
    }

    // set the last token to the remainder of the string
    d_result[token_idx] = string_index_pair{d_str.data() + last_pos, d_str.size_bytes() - last_pos};

    if (direction == split_direction::BACKWARD) {
      // update first entry -- this happens when max_tokens is hit before the end of the string
      auto const first_offset =
        d_result[0].first
          ? static_cast<size_type>(thrust::distance(d_str.data(), d_result[0].first))
          : 0;
      if (first_offset) {
        d_result[0] = string_index_pair{d_str.data(), first_offset + d_result[0].second};
      }
    }
  }
};

/**
 * @brief Call regex to split each input string into tokens.
 *
 * This will also convert the `offsets` values from counts to offsets.
 *
 * @param d_strings Strings to split
 * @param d_prog Regex to evaluate against each string
 * @param direction Whether tokens are generated forwards or backwards.
 * @param max_tokens The maximum number of tokens for each split.
 * @param offsets The number of matches on input.
 *                The offsets for each token in each string on output.
 * @param stream CUDA stream used for kernel launches.
 */
rmm::device_uvector<string_index_pair> generate_tokens(column_device_view const& d_strings,
                                                       reprog_device& d_prog,
                                                       split_direction direction,
                                                       size_type maxsplit,
                                                       mutable_column_view& offsets,
                                                       rmm::cuda_stream_view stream)
{
  auto const strings_count = d_strings.size();

  auto const max_tokens = maxsplit > 0 ? maxsplit : std::numeric_limits<size_type>::max();

  auto const begin     = thrust::make_counting_iterator<size_type>(0);
  auto const end       = thrust::make_counting_iterator<size_type>(strings_count);
  auto const d_offsets = offsets.data<offset_type>();

  // convert match counts to token offsets
  auto map_fn = [d_strings, d_offsets, max_tokens] __device__(auto idx) {
    return d_strings.is_null(idx) ? 0 : std::min(d_offsets[idx], max_tokens) + 1;
  };
  thrust::transform_exclusive_scan(
    rmm::exec_policy(stream), begin, end + 1, d_offsets, map_fn, 0, thrust::plus<offset_type>{});

  // the last offset entry is the total number of tokens to be generated
  auto const total_tokens = cudf::detail::get_value<offset_type>(offsets, strings_count, stream);

  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  if (total_tokens == 0) { return tokens; }

  launch_for_each_kernel(token_reader_fn{d_strings, direction, d_offsets, tokens.data()},
                         d_prog,
                         d_strings.size(),
                         stream);

  return tokens;
}

/**
 * @brief Returns string pair for the specified column for each string in `d_strings`
 *
 * This is used to build the table result of a split.
 * Null is returned if the row is null or if the `column_index` is larger
 * than the token count for that string.
 */
struct tokens_transform_fn {
  column_device_view const d_strings;
  string_index_pair const* d_tokens;
  offset_type const* d_token_offsets;
  size_type const column_index;

  __device__ string_index_pair operator()(size_type idx) const
  {
    auto const offset      = d_token_offsets[idx];
    auto const token_count = d_token_offsets[idx + 1] - offset;
    return (column_index >= token_count) || d_strings.is_null(idx)
             ? string_index_pair{nullptr, 0}
             : d_tokens[offset + column_index];
  }
};

std::unique_ptr<table> split_re(strings_column_view const& input,
                                std::string_view pattern,
                                split_direction direction,
                                size_type maxsplit,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");

  auto const strings_count = input.size();

  std::vector<std::unique_ptr<column>> results;
  if (strings_count == 0) {
    results.push_back(make_empty_column(type_id::STRING));
    return std::make_unique<table>(std::move(results));
  }

  // create the regex device prog from the given pattern
  auto d_prog    = reprog_device::create(pattern, stream);
  auto d_strings = column_device_view::create(input.parent(), stream);

  // count the number of delimiters matched in each string
  auto offsets      = count_matches(*d_strings, *d_prog, strings_count + 1, stream);
  auto offsets_view = offsets->mutable_view();
  auto d_offsets    = offsets_view.data<offset_type>();

  // get the split tokens from the input column; this also converts the counts into offsets
  auto tokens = generate_tokens(*d_strings, *d_prog, direction, maxsplit, offsets_view, stream);

  // the output column count is the maximum number of tokens generated for any input string
  auto const columns_count = thrust::transform_reduce(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_offsets] __device__(auto const idx) -> size_type {
      return d_offsets[idx + 1] - d_offsets[idx];
    },
    0,
    thrust::maximum<size_type>{});

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
  // build a vector of columns
  results.resize(columns_count);
  std::transform(thrust::make_counting_iterator<size_type>(0),
                 thrust::make_counting_iterator<size_type>(columns_count),
                 results.begin(),
                 make_strings_lambda);

  return std::make_unique<table>(std::move(results));
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        std::string_view pattern,
                                        split_direction direction,
                                        size_type maxsplit,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");

  auto const strings_count = input.size();

  // create the regex device prog from the given pattern
  auto d_prog    = reprog_device::create(pattern, stream);
  auto d_strings = column_device_view::create(input.parent(), stream);

  // count the number of delimiters matched in each string
  auto offsets      = count_matches(*d_strings, *d_prog, strings_count + 1, stream, mr);
  auto offsets_view = offsets->mutable_view();

  // get the split tokens from the input column; this also converts the counts into offsets
  auto tokens = generate_tokens(*d_strings, *d_prog, direction, maxsplit, offsets_view, stream);

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

}  // namespace

std::unique_ptr<table> split_re(strings_column_view const& input,
                                std::string_view pattern,
                                size_type maxsplit,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  return split_re(input, pattern, split_direction::FORWARD, maxsplit, stream, mr);
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        std::string_view pattern,
                                        size_type maxsplit,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return split_record_re(input, pattern, split_direction::FORWARD, maxsplit, stream, mr);
}

std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                 std::string_view pattern,
                                 size_type maxsplit,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  return split_re(input, pattern, split_direction::BACKWARD, maxsplit, stream, mr);
}

std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                         std::string_view pattern,
                                         size_type maxsplit,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  return split_record_re(input, pattern, split_direction::BACKWARD, maxsplit, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<table> split_re(strings_column_view const& input,
                                std::string_view pattern,
                                size_type maxsplit,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_re(input, pattern, maxsplit, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        std::string_view pattern,
                                        size_type maxsplit,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_record_re(input, pattern, maxsplit, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                 std::string_view pattern,
                                 size_type maxsplit,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit_re(input, pattern, maxsplit, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                         std::string_view pattern,
                                         size_type maxsplit,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit_record_re(input, pattern, maxsplit, rmm::cuda_stream_default, mr);
}
}  // namespace strings
}  // namespace cudf
