/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cuda_runtime.h>
#include <cudf/types.hpp>

namespace cudf
{
namespace strings
{

// UTF-8 characters are 1-4 bytes
typedef unsigned int char_utf8;

/**---------------------------------------------------------------------------*
 * @brief A non-owning, immutable view of device data that is variable length
 * character array representing a UTF-8 string. The caller must maintain the
 * device memory for the lifetime of this instance.
 *
 * It provides a simple wrapper and string operations for individual char array
 * within a strings column.
 *---------------------------------------------------------------------------**/
class string_view
{
 public:
  string_view() = default;
  /**---------------------------------------------------------------------------*
   * @brief Create instance from existing device char array.
   *
   * @param data Device char array encoded in UTF8.
   * @param bytes Number of bytes in data array.
   *---------------------------------------------------------------------------**/
  __host__ __device__ string_view(const char* data, size_type bytes);
  /**---------------------------------------------------------------------------*
   * @brief Create instance from existing device char array. The array must
   * include a null-terminator ('\0).
   *
   * @param data Device char array encoded in UTF8.
   *---------------------------------------------------------------------------**/
  __device__ string_view(const char* data);
  string_view(const string_view&) = default;
  string_view(string_view&&) = default;
  ~string_view() = default;
  string_view& operator=(const string_view&) = default;
  string_view& operator=(string_view&&) = default;

  /**---------------------------------------------------------------------------*
   * @brief Return the number of bytes in this string
   *---------------------------------------------------------------------------**/
  __host__ __device__ size_type size_bytes() const;
  /**---------------------------------------------------------------------------*
   * @brief Return the number of characters in this string
   *---------------------------------------------------------------------------**/
  __device__ size_type length() const;
  /**---------------------------------------------------------------------------*
   * @brief Return a pointer to the internal device array
   *---------------------------------------------------------------------------**/
  __host__ __device__ const char* data() const;

  /**---------------------------------------------------------------------------*
   * @brief Return true if string has no characters
   *---------------------------------------------------------------------------**/
  __host__ __device__ bool empty() const;
  /**---------------------------------------------------------------------------*
   * @brief Return true if string pointer is null.
   * That is, `data()==nullptr` for this instance.
   *---------------------------------------------------------------------------**/
  __host__ __device__ bool is_null() const;

  /**---------------------------------------------------------------------------*
   * @brief Handy iterator for navigating through encoded characters.
   *---------------------------------------------------------------------------**/
  class iterator
  {
    public:
      using difference_type = ptrdiff_t;
      using value_type = char_utf8;
      using reference = char_utf8&;
      using pointer = char_utf8*;
      using iterator_category = std::input_iterator_tag; // do not allow going backwards
      __device__ iterator(const string_view& str, size_type pos);
      iterator(const iterator& mit) = default;
      iterator(iterator&& mit) = default;
      iterator& operator=(const iterator&) = default;
      iterator& operator=(iterator&&) = default;
      __device__ iterator& operator++();
      __device__ iterator operator++(int);
      __device__ bool operator==(const iterator& rhs) const;
      __device__ bool operator!=(const iterator& rhs) const;
      __device__ char_utf8 operator*() const;
      __device__ size_type position() const;
      __device__ size_type byte_offset() const;
    private:
      const char* p{};
      size_type cpos{}, offset{};
  };

  /**---------------------------------------------------------------------------*
   * @brief Return new iterator pointing to the beginning of this string
   *---------------------------------------------------------------------------**/
  __device__ iterator begin() const;
  /**---------------------------------------------------------------------------*
   * @brief Return new iterator pointing past the end of this string
   *---------------------------------------------------------------------------**/
  __device__ iterator end() const;

  /**---------------------------------------------------------------------------*
   * @brief Return single UTF-8 character at the given character position
   *
   * @param pos Character position
   *---------------------------------------------------------------------------**/
  __device__ char_utf8 at(size_type pos) const;
  /**---------------------------------------------------------------------------*
   * @brief Return single UTF-8 character at the given character position
   *
   * @param pos Character position
   *---------------------------------------------------------------------------**/
  __device__ char_utf8 operator[](size_type pos) const;
  /**---------------------------------------------------------------------------*
   * @brief Return the byte offset from data() for a given character position
   *
   * @param pos Character position
   *---------------------------------------------------------------------------**/
  __device__ size_type byte_offset(size_type pos) const;

  /**---------------------------------------------------------------------------*
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
   *---------------------------------------------------------------------------**/
  __device__ int compare(const string_view& str) const;
  /**---------------------------------------------------------------------------*
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
   *---------------------------------------------------------------------------**/
  __device__ int compare(const char* data, size_type bytes) const;

  /**---------------------------------------------------------------------------*
   * @brief Returns true if arg string matches this string exactly.
   *---------------------------------------------------------------------------**/
  __device__ bool operator==(const string_view& rhs) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns true if arg string does not match this string.
   *---------------------------------------------------------------------------**/
  __device__ bool operator!=(const string_view& rhs) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns true if arg string sorts ascending to this string.
   *---------------------------------------------------------------------------**/
  __device__ bool operator<(const string_view& rhs) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns true if arg string sorts descending to this string.
   *---------------------------------------------------------------------------**/
  __device__ bool operator>(const string_view& rhs) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns true if arg string sorts ascending or matches this string.
   *---------------------------------------------------------------------------**/
  __device__ bool operator<=(const string_view& rhs) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns true if arg string sorts descending or matches this string.
   *---------------------------------------------------------------------------**/
  __device__ bool operator>=(const string_view& rhs) const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the character position of the first occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if str is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type find( const string_view& str, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns the character position of the first occurrence where the
   * array str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target array to search within this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type find( const char* str, size_type bytes, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns the character position of the first occurrence where
   * character is found in this string within the character range [pos,pos+n).
   *
   * @param character Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type find( char_utf8 character, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns the character position of the last occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type rfind( const string_view& str, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns the character position of the last occurrence where the
   * array str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search with this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type rfind( const char* str, size_type bytes, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns the character position of the last occurrence where
   * character is found in this string within the character range [pos,pos+n).
   *
   * @param character Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type rfind( char_utf8 character, size_type pos=0, size_type count=-1 ) const;

  /**---------------------------------------------------------------------------*
   * @brief Return a sub-string of this string. The original string and device
   * memory but must still be maintained for the lifetime of the instance.
   *
   * @param start Character position to start the sub-string.
   * @param length Number of characters from start to include in the sub-string.
   * @return New instance pointing to a subset of the characters within this instance.
   *---------------------------------------------------------------------------**/
  __device__ string_view substr( size_type start, size_type length ) const;

  /**---------------------------------------------------------------------------*
   * @brief Tokenizes this string around the given delimiter up to count times.
   *
   * @param delimiter Character to use for separating tokens.
   * @param count Maximum number of tokens to return.
   *              Specify -1 to indicate all tokens.
   * @param[out] Array to hold output tokens.
   *             Specify nullptr here to return just the token count.
   * @return Number of tokens.
   *---------------------------------------------------------------------------**/
  __device__ size_type split( const char* delimiter, size_type count, string_view* strs ) const;

  /**---------------------------------------------------------------------------*
   * @brief Tokenizes this string around the given delimiter up to count times
   * starting at the end of the string.
   *
   * @param delimiter Character to use for separating tokens.
   * @param count Maximum number of tokens to return.
   *              Specify -1 to indicate all tokens.
   * @param[out] Array to hold output tokens.
   *             Specify nullptr here to return just the token count.
   * @return Number of tokens.
   *---------------------------------------------------------------------------**/
  __device__ size_type rsplit( const char* delimiter, size_type count, string_view* strs ) const;

private:
    const char* _data{};   ///< Pointer to device memory contain char array for this string
    size_type _bytes{};    ///< Number of bytes in _data for this string

  /**---------------------------------------------------------------------------*
   * @brief Return the character position of the given byte offset.
   *
   * @param bytepos Byte position from start of _data.
   * @return The character position for the specified byte.
   *---------------------------------------------------------------------------**/
    __device__ size_type character_offset(size_type bytepos) const;
};

namespace detail
{
/**---------------------------------------------------------------------------*
 * @brief Returns the number of bytes in the specified character.
 *
 * @param chr Single character
 *---------------------------------------------------------------------------**/
__host__ __device__ size_type bytes_in_char_utf8( char_utf8 character );

/**---------------------------------------------------------------------------*
 * @brief Convert a char array into a char_utf8 value.
 *
 * @param str String containing encoded char bytes.
 * @param[out] chr Single char_utf8 value.
 * @return The number of bytes in the character
 *---------------------------------------------------------------------------**/
__host__ __device__ size_type to_char_utf8( const char* str, char_utf8& character );

/**---------------------------------------------------------------------------*
 * @brief Place a char_utf8 value into a char array.
 *
 * @param chr Single character
 * @param[out] str Allocated char array with enough space to hold the encoded characer.
 * @return The number of bytes in the character
 *---------------------------------------------------------------------------**/
__host__ __device__ size_type from_char_utf8( char_utf8 character, char* str );

/**---------------------------------------------------------------------------*
 * @brief Return the number of characters in this provided char array.
 *
 * @param str String with encoded char bytes.
 * @param bytes Number of bytes in str.
 * @return The number of characters in the array.
 *---------------------------------------------------------------------------**/
__host__ __device__ size_type characters_in_string( const char* str, size_type bytes );

} // namespace detail
} // namespace strings
} // namespace cudf

#include "./string_view.inl"
