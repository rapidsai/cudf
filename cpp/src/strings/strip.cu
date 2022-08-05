/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Strip characters from the beginning and/or end of a string.
 *
 * This functor strips the beginning and/or end of each string
 * of any characters found in d_to_strip or whitespace if
 * d_to_strip is empty.
 *
 */
struct strip_fn {
  column_device_view const d_strings;
  strip_type const stype;  // right, left, or both
  string_view const d_to_strip;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str = d_strings.element<string_view>(idx);

    auto is_strip_character = [d_to_strip = d_to_strip] __device__(char_utf8 chr) -> bool {
      return d_to_strip.empty() ? (chr <= ' ') :  // whitespace check
               thrust::any_of(
                 thrust::seq, d_to_strip.begin(), d_to_strip.end(), [chr] __device__(char_utf8 c) {
                   return c == chr;
                 });
    };

    size_type const left_offset = [&] {
      if (stype != strip_type::LEFT && stype != strip_type::BOTH) return 0;
      auto const itr =
        thrust::find_if_not(thrust::seq, d_str.begin(), d_str.end(), is_strip_character);
      return itr != d_str.end() ? itr.byte_offset() : d_str.size_bytes();
    }();

    size_type right_offset = d_str.size_bytes();
    if (stype == strip_type::RIGHT || stype == strip_type::BOTH) {
      auto const length = d_str.length();
      auto itr          = d_str.end();
      for (size_type n = 0; n < length; ++n) {
        if (!is_strip_character(*(--itr))) break;
        right_offset = itr.byte_offset();
      }
    }

    auto const bytes = (right_offset > left_offset) ? right_offset - left_offset : 0;
    if (d_chars)
      memcpy(d_chars + d_offsets[idx], d_str.data() + left_offset, bytes);
    else
      d_offsets[idx] = bytes;
  }
};

}  // namespace

std::unique_ptr<column> strip(
  strings_column_view const& strings,
  strip_type stype                    = strip_type::BOTH,
  string_scalar const& to_strip       = string_scalar(""),
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (strings.is_empty()) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(to_strip.is_valid(stream), "Parameter to_strip must be valid");
  string_view const d_to_strip(to_strip.data(), to_strip.size());

  auto const d_column = column_device_view::create(strings.parent(), stream);

  // this utility calls the strip_fn to build the offsets and chars columns
  auto children = cudf::strings::detail::make_strings_children(
    strip_fn{*d_column, stype, d_to_strip}, strings.size(), stream, mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

}  // namespace detail

// external APIs

std::unique_ptr<column> strip(strings_column_view const& strings,
                              strip_type stype,
                              string_scalar const& to_strip,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::strip(strings, stype, to_strip, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
