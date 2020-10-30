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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <nvtext/detail/tokenize.hpp>
#include <nvtext/ngrams_tokenize.hpp>
#include <strings/utilities.cuh>
#include <text/utilities/tokenize_ops.cuh>

#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief This records the byte positions of each token within each string.
 *
 * The position values are recorded since we need to reference tokens
 * within a string multiple times to generate the ngrams. For example,
 * to generate tri-grams for string "aa b ccc dd" requires creating
 * the following two strings ["aa_b_ccc","b_ccc_dd"]. Notice the
 * tokens "b" and "ccc" needed to be copied twice for this string.
 *
 * Most of the work is done in the characters_tokenizer locating the tokens.
 * This functor simply records the byte positions in the d_token_positions
 * member.
 */
struct string_tokens_positions_fn {
  cudf::column_device_view const d_strings;  // strings to tokenize
  cudf::string_view const d_delimiter;       // delimiter to tokenize around
  int32_t const* d_token_offsets;            // offsets into the d_token_positions for each string
  position_pair* d_token_positions;          // token positions in each string

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    cudf::string_view d_str = d_strings.element<cudf::string_view>(idx);
    // create tokenizer for this string
    characters_tokenizer tokenizer(d_str, d_delimiter);
    // record the token positions for this string
    cudf::size_type token_index = 0;
    auto token_positions        = d_token_positions + d_token_offsets[idx];
    while (tokenizer.next_token())
      token_positions[token_index++] = tokenizer.token_byte_positions();
  }
};

/**
 * @brief Generate the ngrams for each string.
 *
 * The ngrams for each string are placed contiguously within the section of memory
 * assigned for the input string. At the same time, the size of each ngram is recorded
 * in order to build the output offsets column.
 *
 * This functor can be called to compute the size of memory needed to write out
 * each set of ngrams per string. Once the memory offsets (d_chars_offsets) are
 * set and the output memory is allocated (d_chars), the ngrams for each string
 * can be generated into the output buffer.
 */
struct ngram_builder_fn {
  cudf::column_device_view const d_strings;  // strings to generate ngrams from
  cudf::string_view const d_separator;       // separator to place between them 'grams
  cudf::size_type ngrams;                    // ngram number to generate (2=bi-gram, 3=tri-gram)
  int32_t const* d_token_offsets;            // offsets for token position for each string
  position_pair const* d_token_positions;    // token positions for each string
  int32_t const* d_chars_offsets{};          // offsets for each string's ngrams
  char* d_chars{};                           // write ngram strings to here
  int32_t const* d_ngram_offsets{};          // offsets for sizes of each string's ngrams
  int32_t* d_ngram_sizes{};                  // write ngram sizes to here

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    cudf::string_view d_str     = d_strings.element<cudf::string_view>(idx);
    auto token_positions        = d_token_positions + d_token_offsets[idx];
    auto token_count            = d_token_offsets[idx + 1] - d_token_offsets[idx];
    cudf::size_type nbytes      = 0;  // total number of output bytes needed for this string
    cudf::size_type ngram_index = 0;
    char* out_ptr               = d_chars ? d_chars + d_chars_offsets[idx] : nullptr;
    int32_t* d_sizes            = d_ngram_sizes ? d_ngram_sizes + d_ngram_offsets[idx] : nullptr;
    // for ngrams=2, this will turn string "a b c d e" into "a_bb_cc_dd_e"
    for (cudf::size_type token_index = (ngrams - 1); token_index < token_count; ++token_index) {
      cudf::size_type length = 0;                          // calculate size of each ngram in bytes
      for (cudf::size_type n = (ngrams - 1); n >= 0; --n)  // sliding window of tokens
      {
        position_pair item = token_positions[token_index - n];
        length += item.second - item.first;
        if (out_ptr)
          out_ptr = cudf::strings::detail::copy_and_increment(
            out_ptr, d_str.data() + item.first, item.second - item.first);
        if (n > 0) {  // include the separator (except for the last one)
          if (out_ptr) out_ptr = cudf::strings::detail::copy_string(out_ptr, d_separator);
          length += d_separator.size_bytes();
        }
      }
      if (d_sizes) d_sizes[ngram_index++] = length;
      nbytes += length;
    }
    return nbytes;
  }
};

}  // namespace

// detail APIs

std::unique_ptr<cudf::column> ngrams_tokenize(
  cudf::strings_column_view const& strings,
  cudf::size_type ngrams               = 2,
  cudf::string_scalar const& delimiter = cudf::string_scalar(""),
  cudf::string_scalar const& separator = cudf::string_scalar{"_"},
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                  = 0)
{
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be valid");
  cudf::string_view d_separator(separator.data(), separator.size());

  CUDF_EXPECTS(ngrams >= 1, "Parameter ngrams should be an integer value of 1 or greater");
  if (ngrams == 1)  // this is just a straight tokenize
    return tokenize(strings, delimiter, mr, stream);
  auto strings_count = strings.size();
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto execpol        = rmm::exec_policy(stream);
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // Example for comments with ngrams=2
  // ["a bb ccc","dd e"] => ["a_bb", "bb_ccc", "dd_e"]

  // first, get the number of tokens per string to get the token-offsets
  // Ex. token-counts = [3,2]; token-offsets = [0,3,5]
  rmm::device_vector<int32_t> token_offsets(strings_count + 1);
  auto d_token_offsets = token_offsets.data().get();
  thrust::transform_inclusive_scan(rmm::exec_policy(stream)->on(stream),
                                   thrust::make_counting_iterator<cudf::size_type>(0),
                                   thrust::make_counting_iterator<cudf::size_type>(strings_count),
                                   d_token_offsets + 1,
                                   strings_tokenizer{d_strings, d_delimiter},
                                   thrust::plus<int32_t>());
  CUDA_TRY(cudaMemsetAsync(d_token_offsets, 0, sizeof(int32_t), stream));
  auto total_tokens = token_offsets[strings_count];  // Ex. 5 tokens

  // get the token positions (in bytes) per string
  // Ex. start/end pairs: [(0,1),(2,4),(5,8), (0,2),(3,4)]
  rmm::device_vector<position_pair> token_positions(total_tokens);
  auto d_token_positions = token_positions.data().get();
  thrust::for_each_n(
    execpol->on(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count,
    string_tokens_positions_fn{d_strings, d_delimiter, d_token_offsets, d_token_positions});

  // compute the number of ngrams per string to get the total number of ngrams to generate
  // Ex. ngram-counts = [2,1]; ngram-offsets = [0,2,3]; total = 3 bigrams
  rmm::device_vector<int32_t> ngram_offsets(strings_count + 1);
  auto d_ngram_offsets = ngram_offsets.data().get();
  thrust::transform_inclusive_scan(
    execpol->on(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(strings_count),
    d_ngram_offsets + 1,
    [d_token_offsets, ngrams] __device__(cudf::size_type idx) {
      auto token_count = d_token_offsets[idx + 1] - d_token_offsets[idx];
      return (token_count >= ngrams) ? token_count - ngrams + 1 : 0;
    },
    thrust::plus<int32_t>());
  CUDA_TRY(cudaMemsetAsync(d_ngram_offsets, 0, sizeof(int32_t), stream));
  auto total_ngrams = ngram_offsets[strings_count];

  // Compute the total size of the ngrams for each string (not for each ngram)
  // Ex. 2 bigrams in 1st string total to 10 bytes; 1 bigram in 2nd string is 4 bytes
  //     => sizes = [10,4]; offsets = [0,10,14]
  //
  // This produces a set of offsets for the output memory where we can build adjacent
  // ngrams for each string.
  // Ex. bigram for first string produces 2 bigrams ("a_bb","bb_ccc") which
  //     is built in memory like this: "a_bbbb_ccc"
  rmm::device_vector<int32_t> chars_offsets(strings_count + 1);  // output memory offsets
  auto d_chars_offsets = chars_offsets.data().get();             // per input string
  thrust::transform_inclusive_scan(
    execpol->on(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(strings_count),
    d_chars_offsets + 1,
    ngram_builder_fn{d_strings, d_separator, ngrams, d_token_offsets, d_token_positions},
    thrust::plus<int32_t>());
  CUDA_TRY(cudaMemsetAsync(d_chars_offsets, 0, sizeof(int32_t), stream));
  auto output_chars_size = chars_offsets[strings_count];  // Ex. 14 output bytes total

  rmm::device_vector<int32_t> ngram_sizes(total_ngrams);  // size in bytes of each
  auto d_ngram_sizes = ngram_sizes.data().get();          // ngram to generate

  // build chars column
  auto chars_column = cudf::strings::detail::create_chars_child_column(
    strings_count, 0, output_chars_size, mr, stream);
  auto d_chars = chars_column->mutable_view().data<char>();
  // Generate the ngrams into the chars column data buffer.
  // The ngram_builder_fn functor also fills the d_ngram_sizes vector with the
  // size of each ngram.
  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<int32_t>(0),
                     strings_count,
                     ngram_builder_fn{d_strings,
                                      d_separator,
                                      ngrams,
                                      d_token_offsets,
                                      d_token_positions,
                                      d_chars_offsets,
                                      d_chars,
                                      d_ngram_offsets,
                                      d_ngram_sizes});
  // build the offsets column -- converting the ngram sizes into offsets
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    ngram_sizes.begin(), ngram_sizes.end(), mr, stream);
  chars_column->set_null_count(0);
  offsets_column->set_null_count(0);
  // create the output strings column
  return make_strings_column(total_ngrams,
                             std::move(offsets_column),
                             std::move(chars_column),
                             0,
                             rmm::device_buffer{0, stream, mr},
                             stream,
                             mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> ngrams_tokenize(cudf::strings_column_view const& strings,
                                              cudf::size_type ngrams,
                                              cudf::string_scalar const& delimiter,
                                              cudf::string_scalar const& separator,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ngrams_tokenize(strings, ngrams, delimiter, separator, mr);
}

}  // namespace nvtext
