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
#include <nvtext/tokenize.hpp>
#include <strings/utilities.cuh>
#include <text/utilities/tokenize_ops.cuh>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Functor to replace tokens in each string.
 *
 * This tokenizes a string using the given d_delimiter and replaces any tokens that match
 * a string in d_targets with those from the d_repls column. Strings with no matching tokens
 * are left unchanged.
 *
 * This should be called first to compute the size of each output string and then a second
 * time to fill in the allocated output buffer for each string.
 */
struct replace_tokens_fn {
  cudf::column_device_view const d_strings;  ///< strings to tokenize
  cudf::column_device_view const d_targets;  ///< strings to replace
  cudf::column_device_view const d_repls;    ///< replacement strings
  cudf::string_view const d_delimiter;       ///< delimiter characters for tokenizing
  const int32_t* d_offsets{};
  char* d_chars{};

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    auto nbytes      = d_str.size_bytes();
    auto in_ptr      = d_str.data();
    auto out_ptr     = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto last_pos    = cudf::size_type{0};
    characters_tokenizer tokenizer(d_str, d_delimiter);
    while (tokenizer.next_token()) {
      auto token_pos = tokenizer.token_byte_positions();
      cudf::string_view token{d_str.data() + token_pos.first, (token_pos.second - token_pos.first)};
      // check if the token matches any of the targets
      for (auto tidx = 0; tidx < d_targets.size(); ++tidx) {
        cudf::string_view d_target = d_targets.element<cudf::string_view>(tidx);
        if (token.compare(d_target) == 0) {
          // match found
          // retrieve the corresponding replacement string or
          // if only one repl string, use that one for all targets
          cudf::string_view d_repl = d_repls.size() == 1 ? d_repls.element<cudf::string_view>(0)
                                                         : d_repls.element<cudf::string_view>(tidx);
          nbytes += d_repl.size_bytes() - token.size_bytes();  // total output bytes
          if (out_ptr) {
            // copy over string upto the token location
            out_ptr = cudf::strings::detail::copy_and_increment(
              out_ptr, in_ptr + last_pos, token_pos.first - last_pos);
            // copy over replacement string
            out_ptr  = cudf::strings::detail::copy_string(out_ptr, d_repl);
            last_pos = token_pos.second;  // update last byte position for this string
          }
          break;
        }
      }
    }
    // copy the remainder of the string bytes to the output buffer
    if (out_ptr) memcpy(out_ptr, in_ptr + last_pos, d_str.size_bytes() - last_pos);
    return nbytes;
  }
};

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
  if (repls.size() != 1)
    CUDF_EXPECTS(repls.size() == targets.size(),
                 "Parameter targets and repls must be the same size");
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");

  cudf::size_type const strings_count = strings.size();
  if (strings_count == 0) return cudf::make_empty_column(cudf::data_type{cudf::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto targets_column = cudf::column_device_view::create(targets.parent(), stream);
  auto repls_column   = cudf::column_device_view::create(repls.parent(), stream);
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  replace_tokens_fn replacer{*strings_column, *targets_column, *repls_column, d_delimiter};

  // copy null mask from input column
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(), stream, mr);

  // create offsets by calculating size of each string for output
  auto offsets_transformer_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0), replacer);
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  replacer.d_offsets = offsets_column->view().data<int32_t>();

  // build the chars column
  cudf::size_type const bytes = thrust::device_pointer_cast(replacer.d_offsets)[strings_count];
  auto chars_column           = cudf::strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), bytes, mr, stream);
  replacer.d_chars = chars_column->mutable_view().data<char>();

  // copy tokens to the chars buffer
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     strings_count,
                     replacer);
  chars_column->set_null_count(0);  // reset null count for child column

  // return new strings column
  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_column),
                                   std::move(chars_column),
                                   strings.null_count(),
                                   std::move(null_mask),
                                   stream,
                                   mr);
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
