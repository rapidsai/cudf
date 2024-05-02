/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

constexpr char multi_wildcard  = '%';
constexpr char single_wildcard = '_';

template <typename PatternIterator>
struct like_fn {
  column_device_view const d_strings;
  PatternIterator const patterns_itr;
  string_view const d_escape;

  like_fn(column_device_view d_strings, PatternIterator patterns_itr, string_view d_escape)
    : d_strings{d_strings}, patterns_itr{patterns_itr}, d_escape{d_escape}
  {
  }

  __device__ bool operator()(size_type const idx)
  {
    if (d_strings.is_null(idx)) return false;
    auto const d_str     = d_strings.element<string_view>(idx);
    auto const d_pattern = patterns_itr[idx];

    // incrementing by bytes instead of character improves performance 10-20%
    auto target_itr  = d_str.data();
    auto pattern_itr = d_pattern.begin();

    auto const target_end  = target_itr + d_str.size_bytes();
    auto const pattern_end = d_pattern.end();
    auto const esc_char    = d_escape.empty() ? 0 : d_escape[0];

    auto last_target_itr  = target_end;
    auto last_pattern_itr = pattern_end;

    bool result = true;
    while (true) {
      // walk through the pattern and check against the current character
      while (pattern_itr < pattern_end) {
        auto const escaped = *pattern_itr == esc_char;
        auto const pattern_char =
          escaped && (pattern_itr + 1 < pattern_end) ? *(++pattern_itr) : *pattern_itr;

        if (escaped || (pattern_char != multi_wildcard)) {
          size_type char_width = 0;
          // check match with the current character
          result = (target_itr != target_end);
          if (result) {
            if (escaped || pattern_char != single_wildcard) {
              char_utf8 target_char = 0;
              // retrieve the target character to compare with the current pattern_char
              char_width = to_char_utf8(target_itr, target_char);
              result     = (pattern_char == target_char);
            }
          }
          if (!result) { break; }
          ++pattern_itr;
          target_itr += char_width ? char_width : bytes_in_utf8_byte(*target_itr);
        } else {
          // process wildcard '%'
          result = true;
          ++pattern_itr;
          if (pattern_itr == pattern_end) {  // pattern ends with '%' so we are done
            target_itr = target_end;
            break;
          }
          // save positions
          last_pattern_itr = pattern_itr;
          last_target_itr  = target_itr;
        }  // next pattern character
      }

      if (result && (target_itr == target_end)) { break; }  // success

      result = false;
      // check if exhausted either the pattern or the target string
      if (last_pattern_itr == pattern_end || last_target_itr == target_end) { break; }

      // restore saved positions
      pattern_itr = last_pattern_itr;
      last_target_itr += bytes_in_utf8_byte(*last_target_itr);
      target_itr = last_target_itr;
    }
    return result;
  }
};

template <typename PatternIterator>
std::unique_ptr<column> like(strings_column_view const& input,
                             PatternIterator const patterns_itr,
                             string_view const& d_escape,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  if (input.is_empty()) { return results; }

  auto const d_strings = column_device_view::create(input.parent(), stream);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(input.size()),
                    results->mutable_view().data<bool>(),
                    like_fn{*d_strings, patterns_itr, d_escape});

  results->set_null_count(input.null_count());
  return results;
}

}  // namespace

std::unique_ptr<column> like(strings_column_view const& input,
                             string_scalar const& pattern,
                             string_scalar const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(pattern.is_valid(stream), "Parameter pattern must be valid");
  CUDF_EXPECTS(escape_character.is_valid(stream), "Parameter escape_character must be valid");

  auto const d_pattern    = pattern.value(stream);
  auto const patterns_itr = thrust::make_constant_iterator(d_pattern);

  return like(input, patterns_itr, escape_character.value(stream), stream, mr);
}

std::unique_ptr<column> like(strings_column_view const& input,
                             strings_column_view const& patterns,
                             string_scalar const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(patterns.size() == input.size(), "Number of patterns must match the input size");
  CUDF_EXPECTS(patterns.has_nulls() == false, "Parameter patterns must not contain nulls");
  CUDF_EXPECTS(escape_character.is_valid(stream), "Parameter escape_character must be valid");

  auto const d_patterns   = column_device_view::create(patterns.parent(), stream);
  auto const patterns_itr = d_patterns->begin<string_view>();

  return like(input, patterns_itr, escape_character.value(stream), stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> like(strings_column_view const& input,
                             string_scalar const& pattern,
                             string_scalar const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::like(input, pattern, escape_character, stream, mr);
}

std::unique_ptr<column> like(strings_column_view const& input,
                             strings_column_view const& patterns,
                             string_scalar const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::like(input, patterns, escape_character, stream, mr);
}

}  // namespace strings
}  // namespace cudf
