/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/strings/side_type.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Strips a specified character from the either or both ends of a string
 *
 * @param d_str Input string to strip
 * @param d_to_strip String containing the character to strip;
 *                   only the first character is used
 * @param side Which ends of the input string to strip from
 * @return New string excluding the stripped ends
 */
__device__ cudf::string_view strip(cudf::string_view const d_str,
                                   cudf::string_view const d_to_strip,
                                   side_type side = side_type::BOTH)
{
  auto is_strip_character = [d_to_strip](char_utf8 chr) -> bool {
    if (d_to_strip.empty()) return chr <= ' ';  // whitespace check
    for (auto c : d_to_strip) {
      if (c == chr) return true;
    }
    return false;
  };

  auto const left_offset = [&] {
    if (side != side_type::LEFT && side != side_type::BOTH) return 0;
    for (auto itr = d_str.begin(); itr < d_str.end(); ++itr) {
      if (!is_strip_character(*itr)) return itr.byte_offset();
    }
    return d_str.size_bytes();
  }();

  auto const right_offset = [&] {
    if (side != side_type::RIGHT && side != side_type::BOTH) return d_str.size_bytes();
    for (auto itr = d_str.end(); itr > d_str.begin(); --itr) {
      if (!is_strip_character(*(itr - 1))) return itr.byte_offset();
    }
    return 0;
  }();

  auto const bytes = (right_offset > left_offset) ? right_offset - left_offset : 0;
  return cudf::string_view{d_str.data() + left_offset, bytes};
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
