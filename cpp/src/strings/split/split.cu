/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Generic split function called by split() and rsplit().
 *
 * This function will first count the number of delimiters in the entire strings
 * column. Next it records the position of all the delimiters. These positions
 * are used for the remainder of the code to build string_index_pair elements
 * for each output column.
 *
 * The number of tokens for each string is computed by analyzing the delimiter
 * position values and mapping them to each string.
 * The number of output columns is determined by the string with the most tokens.
 * Next the `string_index_pairs` for the entire column are created using the
 * delimiter positions and their string indices vector.
 *
 * Finally, each column is built by creating a vector of tokens (`string_index_pairs`)
 * according to their position in each string. The first token from each string goes
 * into the first output column, the 2nd token from each string goes into the 2nd
 * output column, etc.
 *
 * Output should be comparable to Pandas `split()` with `expand=True` but the
 * rows/columns are transposed.
 *
 * ```
 *   import pandas as pd
 *   pd_series = pd.Series(['', None, 'a_b', '_a_b_', '__aa__bb__', '_a__bbb___c', '_aa_b__ccc__'])
 *   print(pd_series.str.split(pat='_', expand=True))
 *            0     1     2     3     4     5     6
 *      0    ''  None  None  None  None  None  None
 *      1  None  None  None  None  None  None  None
 *      2     a     b  None  None  None  None  None
 *      3    ''     a     b    ''  None  None  None
 *      4    ''    ''    aa    ''    bb    ''    ''
 *      5    ''     a    ''   bbb    ''    ''     c
 *      6    ''    aa     b    ''   ccc    ''    ''
 *
 *   print(pd_series.str.split(pat='_', n=1, expand=True))
 *            0            1
 *      0    ''         None
 *      1  None         None
 *      2     a            b
 *      3    ''         a_b_
 *      4    ''    _aa__bb__
 *      5    ''   a__bbb___c
 *      6    ''  aa_b__ccc__
 *
 *   print(pd_series.str.split(pat='_', n=2, expand=True))
 *            0     1         2
 *      0    ''  None      None
 *      1  None  None      None
 *      2     a     b      None
 *      3    ''     a        b_
 *      4    ''        aa__bb__
 *      5    ''     a  _bbb___c
 *      6    ''    aa  b__ccc__
 * ```
 *
 * @tparam Tokenizer provides unique functions for split/rsplit.
 * @param strings_column The strings to split
 * @param tokenizer Tokenizer for counting and producing tokens
 * @return table of columns for the output of the split
 */
template <typename Tokenizer>
std::unique_ptr<table> split_fn(strings_column_view const& input,
                                Tokenizer tokenizer,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> results;
  if (input.size() == input.null_count()) {
    results.push_back(std::make_unique<column>(input.parent(), stream, mr));
    return std::make_unique<table>(std::move(results));
  }

  // builds the offsets and the vector of all tokens
  auto [offsets, tokens] = split_helper(input, tokenizer, stream, mr);
  auto const d_offsets   = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());
  auto const d_tokens    = tokens.data();

  // compute the maximum number of tokens for any string
  auto const columns_count = thrust::transform_reduce(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(input.size()),
    cuda::proclaim_return_type<size_type>([d_offsets] __device__(auto idx) -> size_type {
      return static_cast<size_type>(d_offsets[idx + 1] - d_offsets[idx]);
    }),
    0,
    thrust::maximum{});

  // build strings columns for each token position
  for (size_type col = 0; col < columns_count; ++col) {
    auto itr = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<string_index_pair>(
        [d_tokens, d_offsets, col] __device__(size_type idx) {
          auto const offset      = d_offsets[idx];
          auto const token_count = static_cast<size_type>(d_offsets[idx + 1] - offset);
          return (col < token_count) ? d_tokens[offset + col] : string_index_pair{nullptr, 0};
        }));
    results.emplace_back(make_strings_column(itr, itr + input.size(), stream, mr));
  }

  return std::make_unique<table>(std::move(results));
}

/**
 * @brief Base class for whitespace tokenizers.
 *
 * These are common methods used by both split and rsplit tokenizer functors.
 */
struct base_whitespace_split_tokenizer {
  // count the tokens only between non-whitespace characters
  __device__ size_type count_tokens(size_type idx) const
  {
    if (d_strings.is_null(idx)) return 0;
    string_view const d_str = d_strings.element<string_view>(idx);
    return count_tokens_whitespace(d_str, max_tokens);
  }

  base_whitespace_split_tokenizer(column_device_view const& d_strings, size_type max_tokens)
    : d_strings(d_strings), max_tokens(max_tokens)
  {
  }

 protected:
  column_device_view const d_strings;
  size_type max_tokens;  // maximum number of tokens
};

/**
 * @brief The tokenizer functions for split() with whitespace.
 *
 * The whitespace tokenizer has no delimiter and handles one or more
 * consecutive whitespace characters as a single delimiter.
 */
struct whitespace_split_tokenizer_fn : base_whitespace_split_tokenizer {
  /**
   * @brief This will create tokens around each runs of whitespace characters.
   *
   * Each token is placed in `d_all_tokens` so they align consecutively
   * with other tokens for the same output column.
   * That is, `d_tokens[col * strings_count + string_index]` is the token at column `col`
   * for string at `string_index`.
   *
   * @param idx Index of the string to process
   * @param d_token_counts Token counts for each string
   * @param d_all_tokens All output tokens for the strings column
   */
  __device__ void process_tokens(size_type idx,
                                 size_type const* d_token_counts,
                                 string_index_pair* d_all_tokens) const
  {
    string_index_pair* d_tokens = d_all_tokens + idx;
    if (d_strings.is_null(idx)) return;
    string_view const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return;
    whitespace_string_tokenizer tokenizer(d_str);
    size_type token_count = d_token_counts[idx];
    size_type token_idx   = 0;
    position_pair token{0, 0};
    while (tokenizer.next_token() && (token_idx < token_count)) {
      token = tokenizer.get_token();
      d_tokens[d_strings.size() * (token_idx++)] =
        string_index_pair{d_str.data() + token.first, (token.second - token.first)};
    }
    if (token_count == max_tokens)
      d_tokens[d_strings.size() * (token_idx - 1)] =
        string_index_pair{d_str.data() + token.first, (d_str.size_bytes() - token.first)};
  }

  whitespace_split_tokenizer_fn(column_device_view const& d_strings, size_type max_tokens)
    : base_whitespace_split_tokenizer(d_strings, max_tokens)
  {
  }
};

/**
 * @brief The tokenizer functions for rsplit() with whitespace.
 *
 * The whitespace tokenizer has no delimiter and handles one or more
 * consecutive whitespace characters as a single delimiter.
 *
 * This one processes tokens from the end of each string.
 */
struct whitespace_rsplit_tokenizer_fn : base_whitespace_split_tokenizer {
  /**
   * @brief This will create tokens around each runs of whitespace characters.
   *
   * Each token is placed in `d_all_tokens` so they align consecutively
   * with other tokens for the same output column.
   * That is, `d_tokens[col * strings_count + string_index]` is the token at column `col`
   * for string at `string_index`.
   *
   * @param idx Index of the string to process
   * @param d_token_counts Token counts for each string
   * @param d_all_tokens All output tokens for the strings column
   */
  __device__ void process_tokens(size_type idx,  // string position index
                                 size_type const* d_token_counts,
                                 string_index_pair* d_all_tokens) const
  {
    string_index_pair* d_tokens = d_all_tokens + idx;
    if (d_strings.is_null(idx)) return;
    string_view const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return;
    whitespace_string_tokenizer tokenizer(d_str, true);
    size_type token_count = d_token_counts[idx];
    size_type token_idx   = 0;
    position_pair token{0, 0};
    while (tokenizer.prev_token() && (token_idx < token_count)) {
      token = tokenizer.get_token();
      d_tokens[d_strings.size() * (token_count - 1 - token_idx)] =
        string_index_pair{d_str.data() + token.first, (token.second - token.first)};
      ++token_idx;
    }
    if (token_count == max_tokens)
      d_tokens[d_strings.size() * (token_count - token_idx)] =
        string_index_pair{d_str.data(), token.second};
  }

  whitespace_rsplit_tokenizer_fn(column_device_view const& d_strings, size_type max_tokens)
    : base_whitespace_split_tokenizer(d_strings, max_tokens)
  {
  }
};

/**
 * @brief Generic split function called by split() and rsplit() using whitespace as a delimiter.
 *
 * The number of tokens for each string is computed by counting consecutive characters
 * between runs of whitespace in each string. The number of output columns is determined
 * by the string with the most tokens. Next the string_index_pairs for the entire column
 * is created.
 *
 * Finally, each column is built by creating a vector of tokens (string_index_pairs)
 * according to their position in each string. The first token from each string goes
 * into the first output column, the 2nd token from each string goes into the 2nd
 * output column, etc.
 *
 * This can be compared to Pandas `split()` with no delimiter and with `expand=True` but
 * with the rows/columns transposed.
 *
 *  import pandas as pd
 *  pd_series = pd.Series(['', None, 'a b', ' a b ', '  aa  bb  ', ' a  bbb   c', ' aa b  ccc  '])
 *  print(pd_series.str.split(pat=None, expand=True))
 *            0     1     2
 *      0  None  None  None
 *      1  None  None  None
 *      2     a     b  None
 *      3     a     b  None
 *      4    aa    bb  None
 *      5     a   bbb     c
 *      6    aa     b   ccc
 *
 *  print(pd_series.str.split(pat=None, n=1, expand=True))
 *            0         1
 *      0  None      None
 *      1  None      None
 *      2     a         b
 *      3     a        b
 *      4    aa      bb
 *      5     a   bbb   c
 *      6    aa  b  ccc
 *
 *  print(pd_series.str.split(pat=None, n=2, expand=True))
 *            0     1      2
 *      0  None  None   None
 *      1  None  None   None
 *      2     a     b   None
 *      3     a     b   None
 *      4    aa    bb   None
 *      5     a   bbb      c
 *      6    aa     b  ccc
 *
 * @tparam Tokenizer provides unique functions for split/rsplit.
 * @param strings_count The number of strings in the column
 * @param tokenizer Tokenizer for counting and producing tokens
 * @return table of columns for the output of the split
 */
template <typename Tokenizer>
std::unique_ptr<table> whitespace_split_fn(size_type strings_count,
                                           Tokenizer tokenizer,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  // compute the number of tokens per string
  rmm::device_uvector<size_type> token_counts(strings_count, stream);
  auto d_token_counts = token_counts.data();
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_token_counts,
                    cuda::proclaim_return_type<size_type>([tokenizer] __device__(size_type idx) {
                      return tokenizer.count_tokens(idx);
                    }));

  // column count is the maximum number of tokens for any string
  size_type const columns_count = thrust::reduce(
    rmm::exec_policy(stream), token_counts.begin(), token_counts.end(), 0, thrust::maximum{});

  std::vector<std::unique_ptr<column>> results;
  // boundary case: if no columns, return one null column (issue #119)
  if (columns_count == 0) {
    results.push_back(std::make_unique<column>(
      data_type{type_id::STRING},
      strings_count,
      rmm::device_buffer{0, stream, mr},  // no data
      cudf::detail::create_null_mask(strings_count, mask_state::ALL_NULL, stream, mr),
      strings_count));
  }

  // get the positions for every token
  rmm::device_uvector<string_index_pair> tokens(
    static_cast<int64_t>(columns_count) * static_cast<int64_t>(strings_count), stream);
  string_index_pair* d_tokens = tokens.data();
  thrust::fill(
    rmm::exec_policy(stream), tokens.begin(), tokens.end(), string_index_pair{nullptr, 0});
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [tokenizer, d_token_counts, d_tokens] __device__(size_type idx) {
                       tokenizer.process_tokens(idx, d_token_counts, d_tokens);
                     });

  // Create each column.
  // - Each pair points to a string for that column for each row.
  // - Create the strings column from the vector using the strings factory.
  for (size_type col = 0; col < columns_count; ++col) {
    auto column_tokens = d_tokens + (col * strings_count);
    results.emplace_back(
      make_strings_column(column_tokens, column_tokens + strings_count, stream, mr));
  }
  return std::make_unique<table>(std::move(results));
}

}  // namespace

std::unique_ptr<column> create_offsets_from_positions(strings_column_view const& input,
                                                      device_span<int64_t const> const& positions,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // first, create a vector of string indices for each position
  auto indices = rmm::device_uvector<size_type>(positions.size(), stream);
  thrust::upper_bound(rmm::exec_policy_nosync(stream),
                      d_offsets,
                      d_offsets + input.size(),
                      positions.begin(),
                      positions.end(),
                      indices.begin());

  // compute position offsets per string
  auto counts = rmm::device_uvector<size_type>(input.size(), stream);
  // memset to zero-out the counts for any null-entries or strings with no positions
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), counts.begin(), counts.end(), 0);

  // next, count the number of positions per string
  auto d_counts  = counts.data();
  auto d_indices = indices.data();
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<int64_t>(0),
    positions.size(),
    [d_indices, d_counts] __device__(int64_t idx) {
      auto const str_idx = d_indices[idx] - 1;
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*(d_counts + str_idx)};
      ref.fetch_add(1L, cuda::std::memory_order_relaxed);
    });

  // finally, convert the counts into offsets
  return std::get<0>(
    cudf::strings::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr));
}

std::unique_ptr<table> split(strings_column_view const& strings_column,
                             string_scalar const& delimiter,
                             size_type maxsplit,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto strings_device_view = column_device_view::create(strings_column.parent(), stream);
  if (delimiter.size() == 0) {
    return whitespace_split_fn(strings_column.size(),
                               whitespace_split_tokenizer_fn{*strings_device_view, max_tokens},
                               stream,
                               mr);
  }

  string_view d_delimiter(delimiter.data(), delimiter.size());
  return split_fn(
    strings_column, split_tokenizer_fn{*strings_device_view, d_delimiter, max_tokens}, stream, mr);
}

std::unique_ptr<table> rsplit(strings_column_view const& strings_column,
                              string_scalar const& delimiter,
                              size_type maxsplit,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto strings_device_view = column_device_view::create(strings_column.parent(), stream);
  if (delimiter.size() == 0) {
    return whitespace_split_fn(strings_column.size(),
                               whitespace_rsplit_tokenizer_fn{*strings_device_view, max_tokens},
                               stream,
                               mr);
  }

  string_view d_delimiter(delimiter.data(), delimiter.size());
  return split_fn(
    strings_column, rsplit_tokenizer_fn{*strings_device_view, d_delimiter, max_tokens}, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<table> split(strings_column_view const& strings_column,
                             string_scalar const& delimiter,
                             size_type maxsplit,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::split(strings_column, delimiter, maxsplit, stream, mr);
}

std::unique_ptr<table> rsplit(strings_column_view const& strings_column,
                              string_scalar const& delimiter,
                              size_type maxsplit,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit(strings_column, delimiter, maxsplit, stream, mr);
}

}  // namespace strings
}  // namespace cudf
