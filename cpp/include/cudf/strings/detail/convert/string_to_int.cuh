/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

/**
 * @brief Converts a single string into an integer.
 *
 * The '+' and '-' are allowed but only at the beginning of the string.
 * The string is expected to contain base-10 [0-9] characters only.
 * Any other character will end the parse.
 * Overflow of the int64 type is not detected.
 */
__device__ inline int64_t string_to_integer(string_view const& d_str)
{
  int64_t value   = 0;
  size_type bytes = d_str.size_bytes();
  if (bytes == 0) return value;
  char const* ptr = d_str.data();
  int sign        = 1;
  if (*ptr == '-' || *ptr == '+') {
    sign = (*ptr == '-' ? -1 : 1);
    ++ptr;
    --bytes;
  }
  for (size_type idx = 0; idx < bytes; ++idx) {
    char chr = *ptr++;
    if (chr < '0' || chr > '9') break;
    value = (value * 10) + static_cast<int64_t>(chr - '0');
  }
  return value * static_cast<int64_t>(sign);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
