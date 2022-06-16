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

#include <cudf/types.hpp>

/**
 * @file
 * @brief Standalone string functions.
 */

namespace cudf {

using char_utf8 = uint32_t;  ///< UTF-8 characters are 1-4 bytes

namespace strings {
namespace detail {

/**
 * @brief This will return true if passed the first byte of a UTF-8 character.
 *
 * @param byte Any byte from a valid UTF-8 character
 * @return true if this the first byte of the character
 */
constexpr bool is_begin_utf8_char(uint8_t byte)
{
  // The (0xC0 & 0x80) bit pattern identifies a continuation byte of a character.
  return (byte & 0xC0) != 0x80;
}

/**
 * @brief Returns the number of bytes in the specified character.
 *
 * @param character Single character
 * @return Number of bytes
 */
constexpr size_type bytes_in_char_utf8(char_utf8 character)
{
  return 1 + static_cast<size_type>((character & unsigned{0x0000FF00}) > 0) +
         static_cast<size_type>((character & unsigned{0x00FF0000}) > 0) +
         static_cast<size_type>((character & unsigned{0xFF000000}) > 0);
}

/**
 * @brief Returns the number of bytes used to represent the provided byte.
 *
 * This could be 0 to 4 bytes. 0 is returned for intermediate bytes within a
 * single character. For example, for the two-byte 0xC3A8 single character,
 * the first byte would return 2 and the second byte would return 0.
 *
 * @param byte Byte from an encoded character.
 * @return Number of bytes.
 */
constexpr size_type bytes_in_utf8_byte(uint8_t byte)
{
  return 1 + static_cast<size_type>((byte & 0xF0) == 0xF0)  // 4-byte character prefix
         + static_cast<size_type>((byte & 0xE0) == 0xE0)    // 3-byte character prefix
         + static_cast<size_type>((byte & 0xC0) == 0xC0)    // 2-byte character prefix
         - static_cast<size_type>((byte & 0xC0) == 0x80);   // intermediate byte
}

/**
 * @brief Convert a char array into a char_utf8 value.
 *
 * @param str String containing encoded char bytes.
 * @param[out] character Single char_utf8 value.
 * @return The number of bytes in the character
 */
constexpr size_type to_char_utf8(const char* str, char_utf8& character)
{
  size_type const chr_width = bytes_in_utf8_byte(static_cast<uint8_t>(*str));

  character = static_cast<char_utf8>(*str++) & 0xFF;
  if (chr_width > 1) {
    character = character << 8;
    character |= (static_cast<char_utf8>(*str++) & 0xFF);  // << 8;
    if (chr_width > 2) {
      character = character << 8;
      character |= (static_cast<char_utf8>(*str++) & 0xFF);  // << 16;
      if (chr_width > 3) {
        character = character << 8;
        character |= (static_cast<char_utf8>(*str++) & 0xFF);  // << 24;
      }
    }
  }
  return chr_width;
}

/**
 * @brief Place a char_utf8 value into a char array.
 *
 * @param character Single character
 * @param[out] str Output array.
 * @return The number of bytes in the character
 */
constexpr inline size_type from_char_utf8(char_utf8 character, char* str)
{
  size_type const chr_width = bytes_in_char_utf8(character);
  for (size_type idx = 0; idx < chr_width; ++idx) {
    str[chr_width - idx - 1] = static_cast<char>(character) & 0xFF;
    character                = character >> 8;
  }
  return chr_width;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
