/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <nvtext/detail/tokenize.hpp>
#include <nvtext/tokenize.hpp>
#include <text/utilities/tokenize_ops.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
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
  auto token_counts   = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                strings_count,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);
  auto d_token_counts = token_counts->mutable_view().data<int32_t>();
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
  rmm::device_vector<int32_t> token_offsets(strings_count + 1);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         d_token_counts.template begin<int32_t>(),
                         d_token_counts.template end<int32_t>(),
                         token_offsets.begin() + 1);
  CUDA_TRY(cudaMemsetAsync(token_offsets.data().get(), 0, sizeof(int32_t), stream.value()));
  auto const total_tokens = token_offsets.back();
  // build a list of pointers to each token
  rmm::device_vector<string_index_pair> tokens(total_tokens);
  // now go get the tokens
  tokenizer.d_offsets = token_offsets.data().get();
  tokenizer.d_tokens  = tokens.data().get();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     strings_count,
                     tokenizer);
  // create the strings column using the tokens pointers
  return cudf::make_strings_column(tokens, stream, mr);
}

}  // namespace

// detail APIs

// zero or more character tokenizer
std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::string_scalar const& delimiter,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");
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
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");
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
  auto offset  = cudf::detail::get_value<int32_t>(offsets, strings_column.offset(), stream);
  auto chars_bytes =
    cudf::detail::get_value<int32_t>(offsets, strings_column.offset() + strings_count, stream) -
    offset;
  auto d_chars = strings_column.chars().data<uint8_t>();  // unsigned is necessary for checking bits
  d_chars += offset;

  // To minimize memory, count the number of characters so we can
  // build the output offsets without an intermediate buffer.
  // In the worst case each byte is a character so the output is 4x the input.
  auto strings_view = cudf::column_device_view::create(strings_column.parent(), stream);
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
  auto offsets_column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  num_characters + 1,
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
  auto d_new_offsets  = offsets_column->mutable_view().begin<int32_t>();
  thrust::copy_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<int32_t>(0),
    thrust::make_counting_iterator<int32_t>(chars_bytes + 1),
    d_new_offsets,
    [d_chars, chars_bytes] __device__(auto idx) {
      // this will also set the final value to the size chars_bytes
      return idx < chars_bytes ? cudf::strings::detail::is_begin_utf8_char(d_chars[idx]) : true;
    });

  // create the output chars column -- just a copy of the input's chars column
  cudf::column_view chars_view(cudf::data_type{cudf::type_id::INT8}, chars_bytes, d_chars);
  auto chars_column = std::make_unique<cudf::column>(chars_view, stream, mr);

  // return new strings column
  return cudf::make_strings_column(num_characters,
                                   std::move(offsets_column),
                                   std::move(chars_column),
                                   0,
                                   rmm::device_buffer{},
                                   stream,
                                   mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::string_scalar const& delimiter,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize(strings, delimiter, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::strings_column_view const& delimiters,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize(strings, delimiters, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::string_scalar const& delimiter,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_tokens(strings, delimiter, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::strings_column_view const& delimiters,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_tokens(strings, delimiters, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> character_tokenize(cudf::strings_column_view const& strings,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::character_tokenize(strings, rmm::cuda_stream_default, mr);
}

}  // namespace nvtext
