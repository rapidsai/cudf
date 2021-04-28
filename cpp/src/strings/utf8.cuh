/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/strings/string_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Converts a single UTF-8 character into a code-point value that
 * can be used for lookup in the character flags or the character case tables.
 *
 * @param utf8_char Single UTF-8 character to convert.
 * @return Code-point for the UTF-8 character.
 */
constexpr uint32_t utf8_to_codepoint(cudf::char_utf8 utf8_char)
{
  uint32_t unchr = 0;
  if (utf8_char < 0x00000080)  // single-byte pass thru
    unchr = utf8_char;
  else if (utf8_char < 0x0000E000)  // two bytes
  {
    unchr = (utf8_char & 0x1F00) >> 2;  // shift and
    unchr |= (utf8_char & 0x003F);      // unmask
  } else if (utf8_char < 0x00F00000)    // three bytes
  {
    unchr = (utf8_char & 0x0F0000) >> 4;         // get upper 4 bits
    unchr |= (utf8_char & 0x003F00) >> 2;        // shift and
    unchr |= (utf8_char & 0x00003F);             // unmask
  } else if (utf8_char <= (unsigned)0xF8000000)  // four bytes
  {
    unchr = (utf8_char & 0x03000000) >> 6;   // upper 3 bits
    unchr |= (utf8_char & 0x003F0000) >> 4;  // next 6 bits
    unchr |= (utf8_char & 0x00003F00) >> 2;  // next 6 bits
    unchr |= (utf8_char & 0x0000003F);       // unmask
  }
  return unchr;
}

/**
 * @brief Converts a character code-point value into a UTF-8 character.
 *
 * @param unchr Character code-point to convert.
 * @return Single UTF-8 character.
 */
constexpr cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
{
  cudf::char_utf8 utf8 = 0;
  if (unchr < 0x00000080)  // single byte utf8
    utf8 = unchr;
  else if (unchr < 0x00000800)  // double byte utf8
  {
    utf8 = (unchr << 2) & 0x1F00;  // shift bits for
    utf8 |= (unchr & 0x3F);        // utf8 encoding
    utf8 |= 0x0000C080;
  } else if (unchr < 0x00010000)  // triple byte utf8
  {
    utf8 = (unchr << 4) & 0x0F0000;   // upper 4 bits
    utf8 |= (unchr << 2) & 0x003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);           // last 6 bits
    utf8 |= 0x00E08080;
  } else if (unchr < 0x00110000)  // quadruple byte utf8
  {
    utf8 = (unchr << 6) & 0x07000000;   // upper 3 bits
    utf8 |= (unchr << 4) & 0x003F0000;  // next 6 bits
    utf8 |= (unchr << 2) & 0x00003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);             // last 6 bits
    utf8 |= (unsigned)0xF0808080;
  }
  return utf8;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
