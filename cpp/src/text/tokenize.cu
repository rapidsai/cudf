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

#include <text/utilities/tokenize_ops.cuh>

#include <nvtext/detail/tokenize.hpp>
#include <nvtext/tokenize.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {
// common pattern for token_count functions
template <typename TokenCounter>
std::unique_ptr<cudf::column> token_count_fn(cudf::size_type strings_count,
                                             TokenCounter tokenizer,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  // create output column
  auto token_counts =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              strings_count,
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto d_token_counts = token_counts->mutable_view().data<cudf::size_type>();
  // add the counts to the column
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(strings_count),
                    d_token_counts,
                    tokenizer);
  return token_counts;
}

// common pattern for tokenize functions
template <typename Tokenizer>
std::unique_ptr<cudf::column> tokenize_fn(cudf::size_type strings_count,
                                          Tokenizer tokenizer,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  // get the number of tokens in each string
  auto const token_counts =
    token_count_fn(strings_count, tokenizer, stream, rmm::mr::get_current_device_resource());
  auto d_token_counts = token_counts->view();
  // create token-index offsets from the counts
  rmm::device_uvector<cudf::size_type> token_offsets(strings_count + 1, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         d_token_counts.template begin<cudf::size_type>(),
                         d_token_counts.template end<cudf::size_type>(),
                         token_offsets.begin() + 1);
  token_offsets.set_element_to_zero_async(0, stream);
  auto const total_tokens = token_offsets.back_element(stream);
  // build a list of pointers to each token
  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  // now go get the tokens
  tokenizer.d_offsets = token_offsets.data();
  tokenizer.d_tokens  = tokens.data();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     strings_count,
                     tokenizer);
  // create the strings column using the tokens pointers
  return cudf::strings::detail::make_strings_column(tokens.begin(), tokens.end(), stream, mr);
}

}  // namespace

// detail APIs

// zero or more character tokenizer
std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::string_scalar const& delimiter,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  return tokenize_fn(strings.size(), strings_tokenizer{*strings_column, d_delimiter}, stream, mr);
}

// zero or more character token counter
std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::string_scalar const& delimiter,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  return token_count_fn(
    strings.size(), strings_tokenizer{*strings_column, d_delimiter}, stream, mr);
}

// one or more string delimiter tokenizer
std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::strings_column_view const& delimiters,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiters.size() > 0, "Parameter delimiters must not be empty");
  CUDF_EXPECTS(!delimiters.has_nulls(), "Parameter delimiters must not have nulls");
  auto strings_column    = cudf::column_device_view::create(strings.parent(), stream);
  auto delimiters_column = cudf::column_device_view::create(delimiters.parent(), stream);
  return tokenize_fn(
    strings.size(),
    multi_delimiter_strings_tokenizer{*strings_column,
                                      delimiters_column->begin<cudf::string_view>(),
                                      delimiters_column->end<cudf::string_view>()},
    stream,
    mr);
}

// one or more string delimiter token counter
std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::strings_column_view const& delimiters,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiters.size() > 0, "Parameter delimiters must not be empty");
  CUDF_EXPECTS(!delimiters.has_nulls(), "Parameter delimiters must not have nulls");
  auto strings_column    = cudf::column_device_view::create(strings.parent(), stream);
  auto delimiters_column = cudf::column_device_view::create(delimiters.parent(), stream);
  return token_count_fn(
    strings.size(),
    multi_delimiter_strings_tokenizer{*strings_column,
                                      delimiters_column->begin<cudf::string_view>(),
                                      delimiters_column->end<cudf::string_view>()},
    stream,
    mr);
}

// tokenize on every character
std::unique_ptr<cudf::column> character_tokenize(cudf::strings_column_view const& strings_column,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  auto strings_count = strings_column.size();
  if (strings_count == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  auto offsets = strings_column.offsets();
  auto offset  = cudf::detail::get_value<cudf::size_type>(offsets, strings_column.offset(), stream);
  auto chars_bytes = cudf::detail::get_value<cudf::size_type>(
                       offsets, strings_column.offset() + strings_count, stream) -
                     offset;
  auto d_chars =
    strings_column.parent().data<uint8_t>();  // unsigned is necessary for checking bits
  d_chars += offset;

  // To minimize memory, count the number of characters so we can
  // build the output offsets without an intermediate buffer.
  // In the worst case each byte is a character so the output is 4x the input.
  cudf::size_type num_characters = thrust::count_if(
    rmm::exec_policy(stream), d_chars, d_chars + chars_bytes, [] __device__(uint8_t byte) {
      return cudf::strings::detail::is_begin_utf8_char(byte);
    });

  // no characters check -- this could happen in all-empty or all-null strings column
  if (num_characters == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  // create output offsets column
  // -- conditionally copy a counting iterator where
  //    the first byte of each character is located
  auto offsets_column =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              num_characters + 1,
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto d_new_offsets = offsets_column->mutable_view().begin<cudf::size_type>();
  thrust::copy_if(
    rmm::exec_policy(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(chars_bytes + 1),
    d_new_offsets,
    [d_chars, chars_bytes] __device__(auto idx) {
      // this will also set the final value to the size chars_bytes
      return idx < chars_bytes ? cudf::strings::detail::is_begin_utf8_char(d_chars[idx]) : true;
    });

  // create the output chars buffer -- just a copy of the input's chars
  rmm::device_uvector<char> output_chars(chars_bytes, stream, mr);
  thrust::copy(rmm::exec_policy(stream), d_chars, d_chars + chars_bytes, output_chars.data());

  // return new strings column
  return cudf::make_strings_column(
    num_characters, std::move(offsets_column), output_chars.release(), 0, rmm::device_buffer{});
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& input,
                                       cudf::string_scalar const& delimiter,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize(input, delimiter, stream, mr);
}

std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& input,
                                       cudf::strings_column_view const& delimiters,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize(input, delimiters, stream, mr);
}

std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& input,
                                           cudf::string_scalar const& delimiter,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_tokens(input, delimiter, stream, mr);
}

std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& input,
                                           cudf::strings_column_view const& delimiters,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_tokens(input, delimiters, stream, mr);
}

std::unique_ptr<cudf::column> character_tokenize(cudf::strings_column_view const& input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::character_tokenize(input, stream, mr);
}

}  // namespace nvtext
