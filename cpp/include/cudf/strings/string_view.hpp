/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <iterator>

/**
 * @file
 * @brief Class definition for cudf::string_view.
 */

namespace cudf {

using char_utf8 = uint32_t;  ///< UTF-8 characters are 1-4 bytes

/**
 * @brief The string length is initialized to this value as a place-holder
 *
 * The number of characters in a string computed on-demand.
 */
constexpr cudf::size_type UNKNOWN_STRING_LENGTH{-1};

/**
 * @brief A non-owning, immutable view of device data that is a variable length
 * char array representing a UTF-8 string.
 *
 * @ingroup strings_classes
 *
 * The caller must maintain the device memory for the lifetime of this instance.
 *
 * This may be used to wrap a device pointer and size but any member function
 * that requires accessing the device memory must be called from a kernel.
 */
class string_view {
 public:
  /**
   * @brief Return the number of bytes in this string
   */
  CUDA_HOST_DEVICE_CALLABLE size_type size_bytes() const { return _bytes; }
  /**
   * @brief Return the number of characters in this string
   */
  CUDA_DEVICE_CALLABLE size_type length() const;
  /**
   * @brief Return a pointer to the internal device array
   */
  CUDA_HOST_DEVICE_CALLABLE const char* data() const { return _data; }

  /**
   * @brief Return true if string has no characters
   */
  CUDA_HOST_DEVICE_CALLABLE bool empty() const { return size_bytes() == 0; }

  /**
   * @brief Handy iterator for navigating through encoded characters.
   */
  class const_iterator {
   public:
    using difference_type   = ptrdiff_t;
    using value_type        = char_utf8;
    using reference         = char_utf8&;
    using pointer           = char_utf8*;
    using iterator_category = std::input_iterator_tag;
    CUDA_DEVICE_CALLABLE const_iterator(const string_view& str, size_type pos);
    const_iterator(const const_iterator& mit) = default;
    const_iterator(const_iterator&& mit)      = default;
    const_iterator& operator=(const const_iterator&) = default;
    const_iterator& operator=(const_iterator&&) = default;
    CUDA_DEVICE_CALLABLE const_iterator& operator++();
    CUDA_DEVICE_CALLABLE const_iterator operator++(int);
    CUDA_DEVICE_CALLABLE const_iterator& operator+=(difference_type);
    CUDA_DEVICE_CALLABLE const_iterator operator+(difference_type);
    CUDA_DEVICE_CALLABLE const_iterator& operator--();
    CUDA_DEVICE_CALLABLE const_iterator operator--(int);
    CUDA_DEVICE_CALLABLE const_iterator& operator-=(difference_type);
    CUDA_DEVICE_CALLABLE const_iterator operator-(difference_type);
    CUDA_DEVICE_CALLABLE bool operator==(const const_iterator&) const;
    CUDA_DEVICE_CALLABLE bool operator!=(const const_iterator&) const;
    CUDA_DEVICE_CALLABLE bool operator<(const const_iterator&) const;
    CUDA_DEVICE_CALLABLE bool operator<=(const const_iterator&) const;
    CUDA_DEVICE_CALLABLE bool operator>(const const_iterator&) const;
    CUDA_DEVICE_CALLABLE bool operator>=(const const_iterator&) const;
    CUDA_DEVICE_CALLABLE char_utf8 operator*() const;
    CUDA_DEVICE_CALLABLE size_type position() const;
    CUDA_DEVICE_CALLABLE size_type byte_offset() const;

   private:
    const char* p{};
    size_type bytes{};
    size_type char_pos{};
    size_type byte_pos{};
  };

  /**
   * @brief Return new iterator pointing to the beginning of this string
   */
  CUDA_DEVICE_CALLABLE const_iterator begin() const;
  /**
   * @brief Return new iterator pointing past the end of this string
   */
  CUDA_DEVICE_CALLABLE const_iterator end() const;

  /**
   * @brief Return single UTF-8 character at the given character position
   *
   * @param pos Character position
   */
  CUDA_DEVICE_CALLABLE char_utf8 operator[](size_type pos) const;
  /**
   * @brief Return the byte offset from data() for a given character position
   *
   * @param pos Character position
   */
  CUDA_DEVICE_CALLABLE size_type byte_offset(size_type pos) const;

  /**
   * @brief Comparing target string with this string. Each character is compared
   * as a UTF-8 code-point value.
   *
   * @param str Target string to compare with this string.
   * @return 0  If they compare equal.
   *         <0 Either the value of the first character of this string that does
   *            not match is lower in the arg string, or all compared characters
   *            match but the arg string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is greater in the arg string, or all compared characters
   *            match but the arg string is longer.
   */
  CUDA_DEVICE_CALLABLE int compare(const string_view& str) const;
  /**
   * @brief Comparing target string with this string. Each character is compared
   * as a UTF-8 code-point value.
   *
   * @param str Target string to compare with this string.
   * @param bytes Number of bytes in str.
   * @return 0  If they compare equal.
   *         <0 Either the value of the first character of this string that does
   *            not match is lower in the arg string, or all compared characters
   *            match but the arg string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is greater in the arg string, or all compared characters
   *            match but the arg string is longer.
   */
  CUDA_DEVICE_CALLABLE int compare(const char* str, size_type bytes) const;

  /**
   * @brief Returns true if rhs matches this string exactly.
   */
  CUDA_DEVICE_CALLABLE bool operator==(const string_view& rhs) const;
  /**
   * @brief Returns true if rhs does not match this string.
   */
  CUDA_DEVICE_CALLABLE bool operator!=(const string_view& rhs) const;
  /**
   * @brief Returns true if this string is ordered before rhs.
   */
  CUDA_DEVICE_CALLABLE bool operator<(const string_view& rhs) const;
  /**
   * @brief Returns true if rhs is ordered before this string.
   */
  CUDA_DEVICE_CALLABLE bool operator>(const string_view& rhs) const;
  /**
   * @brief Returns true if this string matches or is ordered before rhs.
   */
  CUDA_DEVICE_CALLABLE bool operator<=(const string_view& rhs) const;
  /**
   * @brief Returns true if rhs matches or is ordered before this string.
   */
  CUDA_DEVICE_CALLABLE bool operator>=(const string_view& rhs) const;

  /**
   * @brief Returns the character position of the first occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if str is not found in this string.
   */
  CUDA_DEVICE_CALLABLE size_type find(const string_view& str,
                                      size_type pos   = 0,
                                      size_type count = -1) const;
  /**
   * @brief Returns the character position of the first occurrence where the
   * array str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target array to search within this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   */
  CUDA_DEVICE_CALLABLE size_type find(const char* str,
                                      size_type bytes,
                                      size_type pos   = 0,
                                      size_type count = -1) const;
  /**
   * @brief Returns the character position of the first occurrence where
   * character is found in this string within the character range [pos,pos+n).
   *
   * @param character Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   */
  CUDA_DEVICE_CALLABLE size_type find(char_utf8 character,
                                      size_type pos   = 0,
                                      size_type count = -1) const;
  /**
   * @brief Returns the character position of the last occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   */
  CUDA_DEVICE_CALLABLE size_type rfind(const string_view& str,
                                       size_type pos   = 0,
                                       size_type count = -1) const;
  /**
   * @brief Returns the character position of the last occurrence where the
   * array str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search with this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   */
  CUDA_DEVICE_CALLABLE size_type rfind(const char* str,
                                       size_type bytes,
                                       size_type pos   = 0,
                                       size_type count = -1) const;
  /**
   * @brief Returns the character position of the last occurrence where
   * character is found in this string within the character range [pos,pos+n).
   *
   * @param character Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   */
  CUDA_DEVICE_CALLABLE size_type rfind(char_utf8 character,
                                       size_type pos   = 0,
                                       size_type count = -1) const;

  /**
   * @brief Return a sub-string of this string. The original string and device
   * memory must still be maintained for the lifetime of the returned instance.
   *
   * @param start Character position to start the sub-string.
   * @param length Number of characters from start to include in the sub-string.
   * @return New instance pointing to a subset of the characters within this instance.
   */
  CUDA_DEVICE_CALLABLE string_view substr(size_type start, size_type length) const;

  /**
   * @brief Return minimum value associated with the string type
   *
   * This function is needed to be host callable because it is called by a host
   * callable function DeviceMax::identity<string_view>()
   *
   * @return An empty string
   */
  CUDA_HOST_DEVICE_CALLABLE static string_view min();

  /**
   * @brief Return maximum value associated with the string type
   *
   * This function is needed to be host callable because it is called by a host
   * callable function DeviceMin::identity<string_view>()
   *
   * @return A string value which represents the highest possible valid UTF-8 encoded
   * character.
   */
  CUDA_HOST_DEVICE_CALLABLE static string_view max();

  /**
   * @brief Default constructor represents an empty string.
   */
  CUDA_HOST_DEVICE_CALLABLE string_view() : _data(""), _bytes(0), _length(0) {}

  /**
   * @brief Create instance from existing device char array.
   *
   * @param data Device char array encoded in UTF8.
   * @param bytes Number of bytes in data array.
   */
  CUDA_HOST_DEVICE_CALLABLE string_view(const char* data, size_type bytes)
    : _data(data), _bytes(bytes), _length(UNKNOWN_STRING_LENGTH)
  {
  }

  string_view(const string_view&) = default;
  string_view(string_view&&)      = default;
  ~string_view()                  = default;
  string_view& operator=(const string_view&) = default;
  string_view& operator=(string_view&&) = default;

 private:
  const char* _data{};          ///< Pointer to device memory contain char array for this string
  size_type _bytes{};           ///< Number of bytes in _data for this string
  mutable size_type _length{};  ///< Number of characters in this string (computed)

  /**
   * @brief Return the character position of the given byte offset.
   *
   * @param bytepos Byte position from start of _data.
   * @return The character position for the specified byte.
   */
  CUDA_DEVICE_CALLABLE size_type character_offset(size_type bytepos) const;
};

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
CUDA_HOST_DEVICE_CALLABLE size_type to_char_utf8(const char* str, char_utf8& character)
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
 * @param[out] str Allocated char array with enough space to hold the encoded characer.
 * @return The number of bytes in the character
 */
CUDA_HOST_DEVICE_CALLABLE size_type from_char_utf8(char_utf8 character, char* str)
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
