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

#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvtext/detail/tokenize.hpp>
#include <nvtext/replace.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/atomic>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/pair.h>
#include <thrust/remove.h>

namespace nvtext {
namespace detail {
namespace {

using replace_result = thrust::pair<bool, cudf::string_view>;

struct base_token_replacer_fn {
  cudf::column_device_view const d_strings;  ///< strings to tokenize
  cudf::string_view const d_delimiter;       ///< delimiter characters for tokenizing
  cudf::size_type* d_sizes{};                ///< for output string size
  char* d_chars{};                           ///< output buffer
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Tokenizes each string and calls the provided `replacer` function
   * for each token.
   *
   * @tparam ReplaceFn Should accept a `string_view` and return a `replace_result`
   * @param idx Index of the current string to process
   * @param replacer Function to call for each token to determined its replacement
   */
  template <typename ReplaceFn>
  __device__ void process_string(cudf::size_type idx, ReplaceFn replacer)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const in_ptr = d_str.data();
    auto out_ptr      = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto nbytes       = d_str.size_bytes();  // count the output bytes
    auto last_pos     = cudf::size_type{0};
    auto tokenizer    = characters_tokenizer{d_str, d_delimiter};
    // process each token
    while (tokenizer.next_token()) {
      auto const token_pos = tokenizer.token_byte_positions();
      auto const token =
        cudf::string_view{d_str.data() + token_pos.first, token_pos.second - token_pos.first};
      // ask replacer if this token should be replaced
      auto const result = replacer(token);
      if (result.first) {  // first == replace indicator, second == new string
        auto d_replacement = result.second;
        nbytes += d_replacement.size_bytes() - token.size_bytes();
        if (out_ptr) {
          // copy over string up to the token location
          out_ptr = cudf::strings::detail::copy_and_increment(
            out_ptr, in_ptr + last_pos, token_pos.first - last_pos);
          // copy over replacement string
          out_ptr  = cudf::strings::detail::copy_string(out_ptr, d_replacement);
          last_pos = token_pos.second;  // update last byte position for this string
        }
      }
    }

    // copy the remainder of the string's bytes to the output buffer
    if (out_ptr) {
      memcpy(out_ptr, in_ptr + last_pos, d_str.size_bytes() - last_pos);
    } else {
      d_sizes[idx] = nbytes;
    }
  }
};

using strings_iterator = cudf::column_device_view::const_iterator<cudf::string_view>;

/**
 * @brief Functor to replace tokens in each string.
 *
 * This tokenizes a string using the given d_delimiter and replaces any tokens that match
 * a string in d_targets_begin/end with those from the d_replacements column.
 * Strings with no matching tokens are left unchanged.
 *
 * This should be called first to compute the size of each output string and then a second
 * time to fill in the allocated output buffer for each string.
 */
struct replace_tokens_fn : base_token_replacer_fn {
  strings_iterator d_targets_begin;  ///< strings to search for
  strings_iterator d_targets_end;
  cudf::column_device_view const d_replacements;  ///< replacement strings

  replace_tokens_fn(cudf::column_device_view const& d_strings,
                    cudf::string_view const& d_delimiter,
                    strings_iterator d_targets_begin,
                    strings_iterator d_targets_end,
                    cudf::column_device_view const& d_replacements)
    : base_token_replacer_fn{d_strings, d_delimiter},
      d_targets_begin{d_targets_begin},
      d_targets_end{d_targets_end},
      d_replacements{d_replacements}
  {
  }

  /**
   * @brief Return replacement string for the given token.
   *
   * @param token Token candidate to be replaced.
   * @return result pair specifies replacement condition and new string
   */
  __device__ replace_result token_replacement(cudf::string_view const& token)
  {
    // check if the token matches any of the targets
    auto const found_itr = thrust::find(thrust::seq, d_targets_begin, d_targets_end, token);
    if (found_itr != d_targets_end) {  // match found
      // retrieve the corresponding replacement string or
      // if only one repl string, use that one for all targets
      auto const d_repl = [&] {
        auto const repl_idx = thrust::distance(d_targets_begin, found_itr);
        return d_replacements.size() == 1 ? d_replacements.element<cudf::string_view>(0)
                                          : d_replacements.element<cudf::string_view>(repl_idx);
      }();
      return replace_result{true, d_repl};
    }
    // otherwise, do not replace this token
    return replace_result{false, cudf::string_view()};
  }

  __device__ void operator()(cudf::size_type idx)
  {
    process_string(
      idx, [this] __device__(cudf::string_view const& token) { return token_replacement(token); });
  }
};

struct replace_tokens_ls : replace_tokens_fn {
  cudf::size_type const* d_indices;
  cudf::size_type* d_output_sizes;
  replace_tokens_ls(cudf::column_device_view const& d_strings,
                    cudf::string_view const& d_delimiter,
                    strings_iterator d_targets_begin,
                    strings_iterator d_targets_end,
                    cudf::column_device_view const& d_replacements,
                    cudf::size_type* d_indices,
                    cudf::size_type* d_output_sizes)
    : replace_tokens_fn{d_strings, d_delimiter, d_targets_begin, d_targets_end, d_replacements},
      d_indices{d_indices},
      d_output_sizes{d_output_sizes}
  {
  }

  __device__ void operator()(cudf::size_type idx)
  {
    process_string(
      idx, [this] __device__(cudf::string_view const& token) { return token_replacement(token); });
    if (!d_chars) {
      // accumulate sub-row sizes into output row size
      auto const size = d_sizes[idx];
      if (size > 0) {
        auto out_idx = d_indices[idx] - 1;  // adjust for upper_bound
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_block> ref{
          *(d_output_sizes + out_idx)};
        ref.fetch_add(size, cuda::std::memory_order_relaxed);
      }
    }
  }
};

constexpr cudf::size_type AVG_CHAR_BYTES_THRESHOLD = 64;
constexpr cudf::size_type LS_SUB_BLOCK_SIZE        = 64;

struct sub_offset_fn {
  char const* d_input_chars;
  int64_t first_offset;
  int64_t last_offset;
  cudf::string_view const d_delimiter;

  __device__ int64_t operator()(int64_t idx) const
  {
    auto end      = d_input_chars + last_offset;
    auto position = (idx + 1) * LS_SUB_BLOCK_SIZE;
    auto begin    = d_input_chars + first_offset + position;
    while ((begin < end) &&
           cudf::strings::detail::is_utf8_continuation_char(static_cast<u_char>(*begin))) {
      ++begin;
      ++position;
    }
    if (begin >= end) { return 0; }  // or last_offset
    // keep delimiter search within this sub-block
    end = d_input_chars + std::min(last_offset, (idx + 2) * LS_SUB_BLOCK_SIZE + first_offset);
    auto tokenizer = characters_tokenizer(cudf::string_view{}, d_delimiter);
    while (begin < end) {
      auto chr      = cudf::char_utf8{};
      auto chr_size = cudf::strings::detail::to_char_utf8(begin, chr);
      if (tokenizer.is_delimiter(chr)) { break; }
      begin += chr_size;
      position += chr_size;
    }
    return (begin < end) ? position + first_offset : 0;
  }
};

/**
 * @brief Functor to filter tokens in each string.
 *
 * This tokenizes a string using the given d_delimiter and replaces any tokens
 * that are shorter than min_token_length with a replacement string.
 *
 * This should be called first to compute the size of each output string and then
 * a second time to fill in the allocated output buffer for each string.
 */
struct remove_small_tokens_fn : base_token_replacer_fn {
  cudf::size_type min_token_length;       ///< minimum size for found tokens
  cudf::string_view const d_replacement;  ///< replacement string

  remove_small_tokens_fn(cudf::column_device_view const& d_strings,
                         cudf::string_view const& d_delimiter,
                         cudf::size_type min_token_length,
                         cudf::string_view const& d_replacement)
    : base_token_replacer_fn{d_strings, d_delimiter},
      min_token_length{min_token_length},
      d_replacement{d_replacement}
  {
  }

  __device__ void operator()(cudf::size_type idx)
  {
    auto replacer = [this] __device__(cudf::string_view const& token) {
      return replace_result{token.length() < min_token_length, d_replacement};
    };
    process_string(idx, replacer);
  }
};

}  // namespace

// detail APIs

std::unique_ptr<cudf::column> replace_tokens(cudf::strings_column_view const& input,
                                             cudf::strings_column_view const& targets,
                                             cudf::strings_column_view const& replacements,
                                             cudf::string_scalar const& delimiter,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!targets.has_nulls(), "Parameter targets must not have nulls");
  CUDF_EXPECTS(!replacements.has_nulls(), "Parameter replacements must not have nulls");
  if (replacements.size() != 1)
    CUDF_EXPECTS(replacements.size() == targets.size(),
                 "Parameter targets and replacements must be the same size");
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  cudf::size_type const strings_count = input.size();
  if (strings_count == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  auto strings_column      = cudf::column_device_view::create(input.parent(), stream);
  auto targets_column      = cudf::column_device_view::create(targets.parent(), stream);
  auto replacements_column = cudf::column_device_view::create(replacements.parent(), stream);
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());

  auto const first_offset = (input.offset() == 0) ? 0L
                                                  : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset =
    cudf::strings::detail::get_offset_value(input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size = last_offset - first_offset;

  if ((chars_size / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    replace_tokens_fn replacer{*strings_column,
                               d_delimiter,
                               targets_column->begin<cudf::string_view>(),
                               targets_column->end<cudf::string_view>(),
                               *replacements_column};
    // this utility calls replacer to build the offsets and chars columns
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(replacer, strings_count, stream, mr);
    // return new strings column
    return cudf::make_strings_column(strings_count,
                                     std::move(offsets_column),
                                     chars.release(),
                                     input.null_count(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  auto const input_chars = input.chars_begin(stream);
  auto input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  auto sub_count   = chars_size / LS_SUB_BLOCK_SIZE;
  auto tmp_offsets = rmm::device_uvector<int64_t>(sub_count + input.size() + 1, stream);
  {
    rmm::device_uvector<int64_t> sub_offsets(sub_count, stream);
    auto const count_itr = thrust::make_counting_iterator<int64_t>(0);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      count_itr,
                      count_itr + sub_count,
                      sub_offsets.data(),
                      sub_offset_fn{input_chars, first_offset, last_offset});

    auto rend =
      thrust::remove(rmm::exec_policy_nosync(stream), sub_offsets.begin(), sub_offsets.end(), 0L);
    sub_count = thrust::distance(sub_offsets.begin(), rend);
    sub_offsets.resize(sub_count, stream);

    // merge them with input offsets
    thrust::merge(rmm::exec_policy_nosync(stream),
                  input_offsets,
                  input_offsets + input.size() + 1,
                  sub_offsets.begin(),
                  sub_offsets.end(),
                  tmp_offsets.begin());
    tmp_offsets.resize(sub_count + input.size() + 1, stream);
    stream.synchronize();  // protect against destruction of sub_offsets
  }

  auto const tmp_size = static_cast<cudf::size_type>(tmp_offsets.size());
  auto tmp_strings    = cudf::column_view(
    cudf::data_type{cudf::type_id::STRING},
    tmp_size - 1,
    input_chars,
    nullptr,
    0,
    0,
    {cudf::column_view(
      cudf::data_type{cudf::type_id::INT64}, tmp_size, tmp_offsets.data(), nullptr, 0)});
  auto d_strings = cudf::column_device_view::create(tmp_strings, stream);

  auto indices = rmm::device_uvector<cudf::size_type>(tmp_offsets.size(), stream);
  thrust::upper_bound(rmm::exec_policy_nosync(stream),
                      input_offsets,
                      input_offsets + input.size() + 1,
                      tmp_offsets.begin(),
                      tmp_offsets.end(),
                      indices.begin());

  auto d_sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  thrust::fill(rmm::exec_policy_nosync(stream), d_sizes.begin(), d_sizes.end(), 0);
  replace_tokens_ls replacer{*d_strings,
                             d_delimiter,
                             targets_column->begin<cudf::string_view>(),
                             targets_column->end<cudf::string_view>(),
                             *replacements_column,
                             indices.data(),
                             d_sizes.data()};
  auto chars = std::get<1>(
    cudf::strings::detail::make_strings_children(replacer, tmp_strings.size(), stream, mr));
  auto offsets_column = std::get<0>(
    cudf::strings::detail::make_offsets_child_column(d_sizes.begin(), d_sizes.end(), stream, mr));
  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

std::unique_ptr<cudf::column> filter_tokens(cudf::strings_column_view const& strings,
                                            cudf::size_type min_token_length,
                                            cudf::string_scalar const& replacement,
                                            cudf::string_scalar const& delimiter,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  cudf::size_type const strings_count = strings.size();
  if (strings_count == 0) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  cudf::string_view d_replacement(replacement.data(), replacement.size());
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  remove_small_tokens_fn filterer{*strings_column, d_delimiter, min_token_length, d_replacement};

  // copy null mask from input column
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);

  // this utility calls filterer to build the offsets and chars columns
  auto [offsets_column, chars] =
    cudf::strings::detail::make_strings_children(filterer, strings_count, stream, mr);

  // return new strings column
  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_column),
                                   chars.release(),
                                   strings.null_count(),
                                   std::move(null_mask));
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> replace_tokens(cudf::strings_column_view const& input,
                                             cudf::strings_column_view const& targets,
                                             cudf::strings_column_view const& replacements,
                                             cudf::string_scalar const& delimiter,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_tokens(input, targets, replacements, delimiter, stream, mr);
}

std::unique_ptr<cudf::column> filter_tokens(cudf::strings_column_view const& input,
                                            cudf::size_type min_token_length,
                                            cudf::string_scalar const& replacement,
                                            cudf::string_scalar const& delimiter,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_tokens(input, min_token_length, replacement, delimiter, stream, mr);
}

}  // namespace nvtext
