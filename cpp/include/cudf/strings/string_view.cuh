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

#include <cudf/strings/string_view.hpp>

#include <thrust/count.h>
#include <thrust/find.h>
#include <cstdlib>

// This file should only include device code logic.
// Host-only or host/device code should be defined in the string_view.hpp header file.

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Return the number of UTF-8 characters in this provided char array.
 *
 * @param str String with encoded char bytes.
 * @param bytes Number of bytes in str.
 * @return The number of characters in the array.
 */
__device__ inline size_type characters_in_string(const char* str, size_type bytes)
{
  if ((str == 0) || (bytes == 0)) return 0;
  auto ptr = reinterpret_cast<uint8_t const*>(str);
  return thrust::count_if(
    thrust::seq, ptr, ptr + bytes, [](uint8_t chr) { return is_begin_utf8_char(chr); });
}
}  // namespace detail
}  // namespace strings

__device__ inline size_type string_view::length() const
{
  if (_length == UNKNOWN_STRING_LENGTH)
    _length = strings::detail::characters_in_string(_data, _bytes);
  if (_length && (_char_width == UNKNOWN_CHAR_WIDTH)) {
    uint8_t const* ptr = reinterpret_cast<uint8_t const*>(data());
    auto const first   = strings::detail::bytes_in_utf8_byte(*ptr);
    // see if they are all the same width
    _char_width = (thrust::find_if(thrust::seq,
                                   ptr,
                                   ptr + size_bytes(),
                                   [first](auto ch) {
                                     auto width = strings::detail::bytes_in_utf8_byte(ch);
                                     return (width != 0) && (width != first);
                                   })) == (ptr + size_bytes())
                    ? first
                    : VARIABLE_CHAR_WIDTH;
  }
  return _length;
}

// this custom iterator knows about UTF8 encoding
__device__ inline string_view::const_iterator::const_iterator(const string_view& str, size_type pos)
  : p{str.data()}, bytes{str.size_bytes()}, char_pos{pos}, byte_pos{str.byte_offset(pos)}
{
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator++()
{
  if (byte_pos < bytes)
    byte_pos += strings::detail::bytes_in_utf8_byte(static_cast<uint8_t>(p[byte_pos]));
  ++char_pos;
  return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator++(int)
{
  string_view::const_iterator tmp(*this);
  operator++();
  return tmp;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator+(
  string_view::const_iterator::difference_type offset)
{
  const_iterator tmp(*this);
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? ++tmp : --tmp;
  return tmp;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator+=(
  string_view::const_iterator::difference_type offset)
{
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? operator++() : operator--();
  return *this;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator--()
{
  if (byte_pos > 0)
    while (strings::detail::bytes_in_utf8_byte(static_cast<uint8_t>(p[--byte_pos])) == 0)
      ;
  --char_pos;
  return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator--(int)
{
  string_view::const_iterator tmp(*this);
  operator--();
  return tmp;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator-=(
  string_view::const_iterator::difference_type offset)
{
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? operator--() : operator++();
  return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator-(
  string_view::const_iterator::difference_type offset)
{
  const_iterator tmp(*this);
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? --tmp : ++tmp;
  return tmp;
}

__device__ inline bool string_view::const_iterator::operator==(
  const string_view::const_iterator& rhs) const
{
  return (p == rhs.p) && (char_pos == rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator!=(
  const string_view::const_iterator& rhs) const
{
  return (p != rhs.p) || (char_pos != rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator<(
  const string_view::const_iterator& rhs) const
{
  return (p == rhs.p) && (char_pos < rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator<=(
  const string_view::const_iterator& rhs) const
{
  return (p == rhs.p) && (char_pos <= rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator>(
  const string_view::const_iterator& rhs) const
{
  return (p == rhs.p) && (char_pos > rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator>=(
  const string_view::const_iterator& rhs) const
{
  return (p == rhs.p) && (char_pos >= rhs.char_pos);
}

__device__ inline char_utf8 string_view::const_iterator::operator*() const
{
  char_utf8 chr = 0;
  strings::detail::to_char_utf8(p + byte_offset(), chr);
  return chr;
}

__device__ inline size_type string_view::const_iterator::position() const { return char_pos; }

__device__ inline size_type string_view::const_iterator::byte_offset() const { return byte_pos; }

__device__ inline string_view::const_iterator string_view::begin() const
{
  return const_iterator(*this, 0);
}

__device__ inline string_view::const_iterator string_view::end() const
{
  return const_iterator(*this, length());
}

__device__ inline char_utf8 string_view::operator[](size_type pos) const
{
  size_type offset = byte_offset(pos);
  if (offset >= _bytes) return 0;
  char_utf8 chr = 0;
  strings::detail::to_char_utf8(data() + offset, chr);
  return chr;
}

__device__ inline size_type string_view::byte_offset(size_type pos) const
{
  size_type offset = 0;
  const char* sptr = _data;
  const char* eptr = sptr + _bytes;
  if (_char_width > 0) return pos * _char_width;
  while ((pos > 0) && (sptr < eptr)) {
    size_type charbytes = strings::detail::bytes_in_utf8_byte(static_cast<uint8_t>(*sptr++));
    if (charbytes) --pos;
    offset += charbytes;
  }
  return offset;
}

__device__ inline int string_view::compare(const string_view& in) const
{
  return compare(in.data(), in.size_bytes());
}

__device__ inline int string_view::compare(const char* data, size_type bytes) const
{
  size_type const len1      = size_bytes();
  const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(this->data());
  const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(data);
  size_type idx             = 0;
  for (; (idx < len1) && (idx < bytes); ++idx) {
    if (*ptr1 != *ptr2) return static_cast<int32_t>(*ptr1) - static_cast<int32_t>(*ptr2);
    ++ptr1;
    ++ptr2;
  }
  if (idx < len1) return 1;
  if (idx < bytes) return -1;
  return 0;
}

__device__ inline bool string_view::operator==(const string_view& rhs) const
{
  return (size_bytes() == rhs.size_bytes()) && (compare(rhs) == 0);
}

__device__ inline bool string_view::operator!=(const string_view& rhs) const
{
  return compare(rhs) != 0;
}

__device__ inline bool string_view::operator<(const string_view& rhs) const
{
  return compare(rhs) < 0;
}

__device__ inline bool string_view::operator>(const string_view& rhs) const
{
  return compare(rhs) > 0;
}

__device__ inline bool string_view::operator<=(const string_view& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc < 0);
}

__device__ inline bool string_view::operator>=(const string_view& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc > 0);
}

__device__ inline size_type string_view::find(const string_view& str,
                                              size_type pos,
                                              size_type count) const
{
  return find(str.data(), str.size_bytes(), pos, count);
}

__device__ inline size_type string_view::find(const char* str,
                                              size_type bytes,
                                              size_type pos,
                                              size_type count) const
{
  const char* sptr = data();
  if (!str || !bytes) return -1;
  size_type nchars = length();
  if (count < 0) count = nchars;
  size_type end = pos + count;
  if (end < 0 || end > nchars) end = nchars;
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(end);

  size_type len2 = bytes;
  size_type len1 = (epos - spos) - len2 + 1;

  const char* ptr1 = sptr + spos;
  const char* ptr2 = str;
  for (size_type idx = 0; idx < len1; ++idx) {
    bool match = true;
    for (size_type jdx = 0; match && (jdx < len2); ++jdx) match = (ptr1[jdx] == ptr2[jdx]);
    if (match) return character_offset(idx + spos);
    ptr1++;
  }
  return -1;
}

__device__ inline size_type string_view::find(char_utf8 chr, size_type pos, size_type count) const
{
  char str[sizeof(char_utf8)];
  size_type chwidth = strings::detail::from_char_utf8(chr, str);
  return find(str, chwidth, pos, count);
}

__device__ inline size_type string_view::rfind(const string_view& str,
                                               size_type pos,
                                               size_type count) const
{
  return rfind(str.data(), str.size_bytes(), pos, count);
}

__device__ inline size_type string_view::rfind(const char* str,
                                               size_type bytes,
                                               size_type pos,
                                               size_type count) const
{
  const char* sptr = data();
  if (!str || !bytes) return -1;
  size_type nchars = length();
  size_type end    = pos + count;
  if (end < 0 || end > nchars) end = nchars;
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(end);

  size_type len2 = bytes;
  size_type len1 = (epos - spos) - len2 + 1;

  const char* ptr1 = sptr + epos - len2;
  const char* ptr2 = str;
  for (int idx = 0; idx < len1; ++idx) {
    bool match = true;
    for (size_type jdx = 0; match && (jdx < len2); ++jdx) match = (ptr1[jdx] == ptr2[jdx]);
    if (match) return character_offset(epos - len2 - idx);
    ptr1--;  // go backwards
  }
  return -1;
}

__device__ inline size_type string_view::rfind(char_utf8 chr, size_type pos, size_type count) const
{
  char str[sizeof(char_utf8)];
  size_type chwidth = strings::detail::from_char_utf8(chr, str);
  return rfind(str, chwidth, pos, count);
}

// parameters are character position values
__device__ inline string_view string_view::substr(size_type pos, size_type length) const
{
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(pos + length);
  if (epos > size_bytes()) epos = size_bytes();
  if (spos >= epos) return string_view("", 0);
  return string_view(data() + spos, epos - spos);
}

__device__ inline size_type string_view::character_offset(size_type bytepos) const
{
  if (_char_width > 0) return bytepos / _char_width;
  return strings::detail::characters_in_string(data(), bytepos);
}

}  // namespace cudf
