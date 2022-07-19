
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
#pragma once

#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {

enum class strip_type {
  LEFT,   ///< strip characters from the beginning of the string
  RIGHT,  ///< strip characters from the end of the string
  BOTH    ///< strip characters from the beginning and end of the string
};

namespace detail {

__device__ cudf::string_view strip(cudf::string_view const d_str,
                                   cudf::string_view const d_to_strip,
                                   strip_type stype = strip_type::BOTH)
{
  auto is_strip_character = [d_to_strip = d_to_strip](char_utf8 chr) -> bool {
    if (d_to_strip.empty()) return chr <= ' ';  // whitespace check
    for (auto c : d_to_strip) {
      if (c == chr) return true;
    }
    return false;
  };

  size_type const left_offset = [&] {
    if (stype != strip_type::LEFT && stype != strip_type::BOTH) return 0;
    for (auto itr = d_str.begin(); itr < d_str.end(); ++itr) {
      if (!is_strip_character(*itr)) return itr.byte_offset();
    }
    return d_str.size_bytes();
  }();

  size_type const right_offset = [&] {
    if (stype != strip_type::RIGHT && stype != strip_type::BOTH) return d_str.size_bytes();
    for (auto itr = d_str.end(); itr > d_str.begin(); --itr) {
      if (!is_strip_character(*(itr - 1))) return itr.byte_offset();
    }
    return 0;
  }();

  auto const bytes = (right_offset > left_offset) ? right_offset - left_offset : 0;
  return cudf::string_view{d_str.data() + left_offset, bytes};
}

__device__ cudf::string_view strip(cudf::string_view const d_str,
                                   char const* to_strip,
                                   cudf::size_type bytes,
                                   strip_type stype = strip_type::BOTH)
{
  auto const sv = cudf::string_view{to_strip, bytes};
  return strip(d_str, sv, stype);
}

__device__ cudf::string_view lstrip(cudf::string_view const d_str, cudf::string_view d_to_strip)
{
  return strip(d_str, d_to_strip, strip_type::LEFT);
}

__device__ cudf::string_view lstrip(cudf::string_view const d_str,
                                    char const* to_strip,
                                    cudf::size_type bytes)
{
  auto const sv = cudf::string_view{to_strip, bytes};
  return strip(d_str, sv, strip_type::LEFT);
}

__device__ cudf::string_view rstrip(cudf::string_view const d_str, cudf::string_view d_to_strip)
{
  return strip(d_str, d_to_strip, strip_type::RIGHT);
}

__device__ cudf::string_view rstrip(cudf::string_view const d_str,
                                    char const* to_strip,
                                    cudf::size_type bytes)
{
  auto const sv = cudf::string_view{to_strip, bytes};
  return strip(d_str, sv, strip_type::RIGHT);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
