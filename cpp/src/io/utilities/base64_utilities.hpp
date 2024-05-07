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
 *  base64.cpp and base64.h
 *
 *  base64 encoding and decoding with C++.
 *  More information at
 *    https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp
 *
 *  Version: 2.rc.09 (release candidate)
 *
 *  Copyright (C) 2004-2017, 2020-2022 René Nyffenegger
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
#include <cudf/utilities/error.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <functional>
#include <string>

// altered: merged base64.h and base64.cpp into one file.
// altered: applying clang-format for libcudf on this file.

// altered: use cudf namespaces
namespace cudf::io::detail {

static const std::string base64_chars =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz"
  "0123456789+/";

static constexpr unsigned char trailing_char = '=';

// Function to encode input string to base64 and return the encoded string
std::string base64_encode(std::string_view string_to_encode)
{
  // altered: use braces around if else
  auto bytes_to_encode = string_to_encode.data();
  auto input_length    = string_to_encode.size();

  // altered: compute complete encoding length = floor(multiple of 3)
  int32_t complete_encoding_length = (input_length / 3) * 3;
  auto remaining_bytes             = input_length - complete_encoding_length;
  CUDF_EXPECTS(remaining_bytes < 3, "Remaining bytes must be < 3");

  std::string encoded;
  size_t encoded_length = (input_length + 2) / 3 * 4;
  encoded.reserve(encoded_length);

  // altered: modify base64 encoder loop using STL and Thrust.
  // TODO: Port this loop to thrust cooperative groups of size 4 if needed for too-wide tables.
  std::for_each(thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(complete_encoding_length),
                [&](auto&& idx) {
                  auto modulus = idx % 3;
                  switch (modulus) {
                    case 0:
                      encoded.push_back(base64_chars[(bytes_to_encode[idx] & 0xfc) >> 2]);
                      break;

                    case 1:
                      encoded.push_back(base64_chars[((bytes_to_encode[idx - 1] & 0x03) << 4) +
                                                     ((bytes_to_encode[idx] & 0xf0) >> 4)]);
                      break;

                    case 2:
                      encoded.push_back(base64_chars[((bytes_to_encode[idx - 1] & 0x0f) << 2) +
                                                     ((bytes_to_encode[idx] & 0xc0) >> 6)]);
                      encoded.push_back(base64_chars[bytes_to_encode[idx] & 0x3f]);
                      break;

                    default:
                      // altered: default case should never be reached
                      CUDF_UNREACHABLE("Invalid modulus");
                      break;
                  }
                });

  // altered: encode the remaining 1 or 2 bytes
  switch (remaining_bytes) {
    case 0: break;

    case 1:
      // from case 0
      encoded.push_back(base64_chars[(bytes_to_encode[complete_encoding_length] & 0xfc) >> 2]);
      // from case 1
      encoded.push_back(base64_chars[(bytes_to_encode[complete_encoding_length] & 0x03) << 4]);
      // two trailing characters
      encoded.push_back(trailing_char);
      encoded.push_back(trailing_char);
      break;

    case 2:
      // from case 0
      encoded.push_back(base64_chars[(bytes_to_encode[complete_encoding_length] & 0xfc) >> 2]);
      // from case 1
      encoded.push_back(
        base64_chars[((bytes_to_encode[complete_encoding_length] & 0x03) << 4) +
                     ((bytes_to_encode[complete_encoding_length + 1] & 0xf0) >> 4)]);
      // from case 2
      encoded.push_back(base64_chars[(bytes_to_encode[complete_encoding_length + 1] & 0x0f) << 2]);
      // one trailing character
      encoded.push_back(trailing_char);
      break;
    default: CUDF_UNREACHABLE("Invalid number of remaining bytes"); break;
  }

  return encoded;
}

// base64 decode function
std::string base64_decode(std::string_view encoded_string)
{
  // altered: converted to lambda function inside base64_decode
  // Function to compute and return the position of character within base64
  static constexpr auto error_position = static_cast<size_t>(-1);
  std::function<size_t(unsigned char const)> base64_position =
    [&](unsigned char const chr) -> size_t {
    // altered: use braces around if else
    if (chr >= 'A' and chr <= 'Z') {
      return chr - 'A';
    } else if (chr >= 'a' and chr <= 'z') {
      return chr - 'a' + ('Z' - 'A') + 1;
    } else if (chr >= '0' and chr <= '9') {
      return chr - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
    } else if (chr == '+') {
      return 62;
    } else if (chr == '/') {
      return 63;
    } else {
      CUDF_LOG_ERROR(
        "Parquet reader encountered invalid base64-encoded data."
        "arrow:schema not processed.");
      return error_position;
    }
  };

  // altered: there must be at least 2 characters in the base64-encoded string
  if (encoded_string.size() < 2) {
    CUDF_LOG_ERROR(
      "Parquet reader encountered invalid base64-encoded string size."
      "arrow:schema not processed.");
    return std::string();
  }

  size_t input_length = encoded_string.length();
  std::string decoded;

  //
  // The approximate length (bytes) of the decoded string might be one or
  // two bytes smaller, depending on the amount of trailing equal signs
  // in the encoded string. This approximation is needed to reserve
  // enough space in the string to be returned.
  size_t approx_decoded_length = input_length / 4 * 3;
  decoded.reserve(approx_decoded_length);

  //
  // Iterate over encoded input string in chunks. The size of all
  // chunks except the last one is 4 bytes.
  //
  // The last chunk might be padded with equal signs or dots
  // in order to make it 4 bytes in size as well, but this
  // is not required as per RFC 2045.
  //
  // All chunks except the last one produce three output bytes.
  //
  // The last chunk produces at least one and up to three bytes.
  //
  // altered: modify base64 encoder loop using STL and Thrust.
  // TODO: Port this loop to thrust cooperative groups of size 3 if needed for too-wide tables.
  if (not std::all_of(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int32_t>(input_length)),
        [&](auto&& idx) {
          int32_t modulus              = idx % 4;
          size_t current_char_position = 0;
          size_t char1_position        = 0;
          size_t char2_position        = 0;

          // Check for data that is not padded with equal
          // signs (which is allowed by RFC 2045)
          if (encoded_string[idx] == '=') { return true; }

          switch (modulus) {
            case 0:
              current_char_position = base64_position(encoded_string[idx]);
              char1_position        = base64_position(encoded_string[idx + 1]);
              if (current_char_position == error_position or char1_position == error_position) {
                return false;
              }
              // Emit the first output byte that is produced in each chunk:
              decoded.push_back(static_cast<std::string::value_type>(
                (current_char_position << 2) + ((char1_position & 0x30) >> 4)));
              break;

            case 2:
              char1_position = base64_position(encoded_string[idx - 1]);
              char2_position = base64_position(encoded_string[idx]);
              if (char1_position == error_position or char2_position == error_position) {
                return false;
              }
              // Emit a chunk's second byte (which might not be produced in the last chunk).
              decoded.push_back(static_cast<std::string::value_type>(
                ((char1_position & 0x0f) << 4) + ((char2_position & 0x3c) >> 2)));
              break;

            case 3:
              char2_position        = base64_position(encoded_string[idx - 1]);
              current_char_position = base64_position(encoded_string[idx]);
              if (current_char_position == error_position or char2_position == error_position) {
                return false;
              }
              // Emit a chunk's third byte (which might not be produced in the last chunk).
              decoded.push_back(static_cast<std::string::value_type>(
                ((char2_position & 0x03) << 6) + current_char_position));
              break;

            default:  // case 1 (ignore)
              break;
          }
          // all good, return true
          return true;
        })) {
    return std::string();
  }

  // return the decoded string
  return decoded;
}

}  // namespace cudf::io::detail
