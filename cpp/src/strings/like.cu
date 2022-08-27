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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

constexpr char multi_wildcard  = '%';
constexpr char single_wildcard = '_';

struct like_fn {
  column_device_view const d_strings;
  string_view const d_pattern;
  string_view const d_escape;

  __device__ bool operator()(size_type const idx)
  {
    if (d_strings.is_null(idx)) return false;
    auto const d_str = d_strings.element<string_view>(idx);

    // using only iterators to better handle UTF-8 characters
    auto target_itr  = d_str.begin();
    auto pattern_itr = d_pattern.begin();

    auto const target_end  = d_str.end();
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
          // check match with the current character
          result = ((target_itr != target_end) && ((!escaped && pattern_char == single_wildcard) ||
                                                   (pattern_char == *target_itr)));
          if (!result) { break; }
          ++target_itr;
          ++pattern_itr;
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
        }
      }  // next pattern character

      if (result && (target_itr == target_end)) { break; }  // success

      result = false;
      // check if exhausted either the pattern or the target string
      if (last_pattern_itr == pattern_end || last_target_itr == target_end) { break; }

      // restore saved positions
      pattern_itr = last_pattern_itr;
      target_itr  = ++last_target_itr;
    }
    return result;
  }
};

}  // namespace

std::unique_ptr<column> like(
  strings_column_view const& input,
  string_scalar const& pattern,
  string_scalar const& escape_character,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto results = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  if (input.is_empty()) { return results; }

  CUDF_EXPECTS(pattern.is_valid(stream), "Parameter pattern must be valid");
  CUDF_EXPECTS(escape_character.is_valid(stream), "Parameter escape_character must be valid");

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_pattern = pattern.value(stream);
  auto const d_escape  = escape_character.value(stream);

  auto d_results = results->mutable_view().data<bool>();

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(input.size()),
                    results->mutable_view().data<bool>(),
                    like_fn{*d_strings, d_pattern, d_escape});

  return results;
}

}  // namespace detail

// external API

std::unique_ptr<column> like(strings_column_view const& input,
                             string_scalar const& pattern,
                             string_scalar const& escape_character,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::like(input, pattern, escape_character, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
