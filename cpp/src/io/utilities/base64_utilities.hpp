/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/*
 * Portions of this file are derived from Rene Nyffenegger's codebase at
 * https://github.com/ReneNyffenegger/cpp-base64, original license text below.
 */

/*
 *  base64_utils.cpp and base64_utils.hpp
 *
 *  base64 encoding and decoding with C++.
 *
 *  Version: 1.01.00
 *
 *  Copyright (C) 2004-2017 René Nyffenegger
 *
 *  This source code is provided 'as-is', without any express or implied
 *  warranty. In no event will the author be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  1. The origin of this source code must not be misrepresented; you must not
 *     claim that you wrote the original source code. If you use this source code
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *
 *  2. Altered source versions must be plainly marked as such, and must not be
 *     misrepresented as being the original source code.
 *
 *  3. This notice may not be removed or altered from any source distribution.
 *
 *  René Nyffenegger rene.nyffenegger@adp-gmbh.ch
 */

/**
 * @file base64_utils.hpp
 * @brief base64 string encoding/decoding utilities and implementation
 */

#pragma once

// altered: including required std headers
#include <array>
#include <iostream>
#include <string>
#include <vector>

// altered: merged base64.h and base64.cpp into one file.
// altered: applying clang-format for libcudf on this file.

// altered: use cudf namespaces
namespace cudf::io::detail {

static const std::string base64_chars =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz"
  "0123456789+/";

static inline auto is_base64(unsigned char c) { return (isalnum(c) or (c == '+') or (c == '/')); }

// merging the encoder wrapper into the single function
std::string base64_encode(std::string_view string_to_encode)
{
  // get bytes to encode and length
  auto bytes_to_encode = reinterpret_cast<const unsigned char*>(string_to_encode.data());
  auto input_length    = string_to_encode.size();

  std::string encoded;
  std::array<unsigned char, 4> char_array_4;
  std::array<unsigned char, 3> char_array_3;
  int i = 0;
  int j = 0;

  // altered: added braces to one liner loops in the rest of this function
  while (input_length--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; (i < 4); i++) {
        encoded += base64_chars[char_array_4[i]];
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 3; j++) {
      char_array_3[j] = '\0';
    }

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (j = 0; (j < i + 1); j++) {
      encoded += base64_chars[char_array_4[j]];
    }
    while ((i++ < 3)) {
      encoded += '=';
    }
  }

  return encoded;
}

// base64 decode lambda function
std::string base64_decode(std::string_view encoded_string)
{
  std::array<unsigned char, 4> char_array_4;
  std::array<unsigned char, 3> char_array_3;
  std::string decoded;
  size_t input_len = encoded_string.size();

  int i   = 0;
  int j   = 0;
  int in_ = 0;

  // altered: added braces to one liner loops in the rest of this function
  while (input_len-- and (encoded_string[in_] != '=') and is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_];
    in_++;
    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = base64_chars.find(char_array_4[i]) & 0xff;
      }

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++) {
        decoded += char_array_3[i];
      }
      i = 0;
    }
  }

  // altered: modify to i!=0 for better readability
  if (i != 0) {
    for (j = 0; j < i; j++) {
      char_array_4[j] = base64_chars.find(char_array_4[j]) & 0xff;
    }
    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    // altered: arrow source code doesn't have the below line.
    //          This is inconsequential as it is never appended to
    //          `decoded` as max(i) = 3 and 0 <= j < 2.
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; j < i - 1; j++) {
      decoded += char_array_3[j];
    }
  }

  return decoded;
}

}  // namespace cudf::io::detail
