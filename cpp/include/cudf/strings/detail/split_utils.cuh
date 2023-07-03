/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
namespace detail {

// JIT has trouble including thrust/pair.h
struct position_pair {
  size_type first;
  size_type second;
};

/**
 * @brief Instantiated for each string to manage navigating tokens from
 * the beginning or the end of that string.
 */
struct whitespace_string_tokenizer {
  /**
   * @brief Identifies the position range of the next token in the given
   * string at the specified iterator position.
   *
   * Tokens are delimited by one or more whitespace characters.
   *
   * @return true if a token has been found
   */
  __device__ bool next_token()
  {
    if (itr != d_str.begin()) {            // skip these 2 lines the first time through
      ++itr;
      start_position = itr.byte_offset();  // end_position + 1;
    }
    if (start_position >= d_str.size_bytes()) return false;
    // continue search for the next token
    end_position = d_str.size_bytes();
    for (; itr < d_str.end(); ++itr) {
      if (spaces == (*itr <= ' ')) {
        if (spaces)
          start_position = (itr + 1).byte_offset();
        else
          end_position = (itr + 1).byte_offset();
        continue;
      }
      spaces = !spaces;
      if (spaces) {
        end_position = itr.byte_offset();
        break;
      }
    }
    return start_position < end_position;
  }

  /**
   * @brief Identifies the position range of the previous token in the given
   * string at the specified iterator position.
   *
   * Tokens are delimited by one or more whitespace characters.
   *
   * @return true if a token has been found
   */
  __device__ bool prev_token()
  {
    end_position = start_position - 1;
    --itr;
    if (end_position <= 0) return false;
    // continue search for the next token
    start_position = 0;
    for (; itr >= d_str.begin(); --itr) {
      if (spaces == (*itr <= ' ')) {
        if (spaces)
          end_position = itr.byte_offset();
        else
          start_position = itr.byte_offset();
        continue;
      }
      spaces = !spaces;
      if (spaces) {
        start_position = (itr + 1).byte_offset();
        break;
      }
    }
    return start_position < end_position;
  }

  __device__ position_pair get_token() const { return position_pair{start_position, end_position}; }

  __device__ whitespace_string_tokenizer(string_view const& d_str, bool reverse = false)
    : d_str{d_str},
      spaces(true),
      start_position{reverse ? d_str.size_bytes() + 1 : 0},
      end_position{d_str.size_bytes()},
      itr{reverse ? d_str.end() : d_str.begin()}
  {
  }

 private:
  string_view const d_str;
  bool spaces;  // true if current position is whitespace
  cudf::string_view::const_iterator itr;
  size_type start_position;
  size_type end_position;
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
