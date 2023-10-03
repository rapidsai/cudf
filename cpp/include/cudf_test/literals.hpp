/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

namespace cudf {
namespace test {
namespace literals {

/**
 * @brief User-defined literal operator for creating an `__int128_t`
 *
 * @param s The string to be converted to `__int128_t`
 * @return The `__int128_t` value
 */
constexpr __int128_t operator""_int128_t(const char* s)
{
  __int128_t ret  = 0;
  __int128_t sign = 1;
  for (int i = 0; s[i] != '\0'; ++i) {
    ret *= 10;
    if (i == 0 && s[i] == '-') {
      sign = -1;
    } else if ('0' <= s[i] && s[i] <= '9') {
      ret += s[i] - '0';
    }
  }
  return sign * ret;
}

}  // namespace literals
}  // namespace test
}  // namespace cudf
