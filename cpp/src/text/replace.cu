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
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <nvtext/detail/tokenize.hpp>
#include <nvtext/tokenize.hpp>
#include <strings/utilities.cuh>
#include <text/utilities/tokenize_ops.cuh>

#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

struct replace_tokens_fn {
  cudf::column_device_view const d_strings;  ///< strings to tokenize
  cudf::column_device_view const d_targets;
  cudf::column_device_view const d_repls;
  cudf::string_view const d_delimiter;  ///< delimiter characters to tokenize around
  const int32_t* d_offsets{};
  char* d_chars{};

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    cudf::string_view d_str  = d_strings.element<cudf::string_view>(idx);
    cudf::size_type nbytes   = d_str.size_bytes();
    auto in_ptr              = d_str.data();  // input buffer
    char* out_ptr            = d_chars ? nullptr : d_chars + d_offsets[idx];
    cudf::size_type last_pos = 0;
    characters_tokenizer tokenizer(d_str, d_delimiter);
    while (tokenizer.next_token()) {
      auto token_pos = tokenizer.token_byte_positions();
      cudf::string_view token{d_str.data() + token_pos.first, (token_pos.second - token_pos.first)};
      // check if matches a target
      for (auto tidx = 0; tidx < d_targets.size(); ++tidx) {
        cudf::string_view d_target = d_targets.element<cudf::string_view>(tidx);
        if (token.compare(d_target) == 0) {
          cudf::string_view d_repl = d_repls.element<cudf::string_view>(tidx);
          nbytes += d_repl.size_bytes() - token.size_bytes();
          if (out_ptr) {
            out_ptr = cudf::strings::detail::copy_and_increment(
              out_ptr, in_ptr + last_pos, token_pos.first - last_pos);
            out_ptr  = cudf::strings::detail::copy_string(out_ptr, d_repl);
            last_pos = token_pos.second;
          }
          break;
        }
      }
    }
    if (out_ptr) memcpy(out_ptr, in_ptr + last_pos, d_str.size_bytes() - last_pos);
    return nbytes;
  }
};

// common pattern for tokenize functions
template <typename Tokenizer>
std::unique_ptr<cudf::column> tokenize_fn(cudf::size_type strings_count,
                                          Tokenizer tokenizer,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream)
{
}

}  // namespace

// detail APIs

// zero or more character tokenizer
std::unique_ptr<cudf::column> replace_tokens(cudf::strings_column_view const& strings,
                                             cudf::strings_column_view const& targets,
                                             cudf::strings_column_view const& repls,
                                             cudf::string_scalar const& delimiter,
                                             cudaStream_t stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!targets.has_nulls(), "Parameter targets must not have nulls");
  CUDF_EXPECTS(!repls.has_nulls(), "Parameter repls must not have nulls");
  CUDF_EXPECTS(repls.size() == targets.size(), "Parameter targets and repls must be the same size");
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  return tokenize_fn(strings.size(), strings_tokenizer{*strings_column, d_delimiter}, mr, stream);

  auto execpol = rmm::exec_policy(stream);
  // get the number of tokens in each string
  auto const token_counts =
    token_count_fn(strings_count, tokenizer, rmm::mr::get_default_resource(), stream);
  auto d_token_counts = token_counts->view();
  // create token-index offsets from the counts
  rmm::device_vector<int32_t> token_offsets(strings_count + 1);
  thrust::inclusive_scan(execpol->on(stream),
                         d_token_counts.template begin<int32_t>(),
                         d_token_counts.template end<int32_t>(),
                         token_offsets.begin() + 1);
  CUDA_TRY(cudaMemsetAsync(token_offsets.data().get(), 0, sizeof(int32_t), stream));
  auto const total_tokens = token_offsets.back();
  // build a list of pointers to each token
  rmm::device_vector<string_index_pair> tokens(total_tokens);
  // now go get the tokens
  tokenizer.d_offsets = token_offsets.data().get();
  tokenizer.d_tokens  = tokens.data().get();
  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     strings_count,
                     tokenizer);
  // create the strings column using the tokens pointers
  return cudf::make_strings_column(tokens, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> replace_tokens(cudf::strings_column_view const& strings,
                                             cudf::strings_column_view const& targets,
                                             cudf::strings_column_view const& repls,
                                             cudf::string_scalar const& delimiter,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_tokens(strings, targets, repls, delimiter, 0, mr);
}

}  // namespace nvtext
