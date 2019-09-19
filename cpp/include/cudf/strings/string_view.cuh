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

#include <cuda_runtime.h>
#include <cudf/types.hpp>

namespace cudf
{

// utf8 characters are 1-4 bytes
typedef unsigned int Char;

/**---------------------------------------------------------------------------*
 * @brief A non-owning, immutable view of device data that is variable length
 * character array representing a UTF-8 string. The caller must maintain the
 * device memory for the lifetime of this instance.
 *
 * It provides a simple wrapper and string operations for individual char array
 * within a strings column. This is likely created dynamically and temporarily.
 * It is not recommended to be allocated directly on the global memory heap.
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
  __host__ __device__ size_type size() const;
  /**---------------------------------------------------------------------------*
   * @brief Return the number of bytes in this string
   *---------------------------------------------------------------------------**/
  __host__ __device__ size_type length() const;
  /**---------------------------------------------------------------------------*
   * @brief Return the number of characters (UTF-8) in this string
   *---------------------------------------------------------------------------**/
  __device__ size_type characters() const;
  /**---------------------------------------------------------------------------*
   * @brief Return a pointer to the internal device array
   *---------------------------------------------------------------------------**/
  __host__ __device__ const char* data() const;

  /**---------------------------------------------------------------------------*
   * @brief Return true if string has no characters
   *---------------------------------------------------------------------------**/
  __device__ bool empty() const;
  __device__ bool is_null() const;

  /**---------------------------------------------------------------------------*
   * @brief Handy iterator for navigating through encoded characters.
   *---------------------------------------------------------------------------**/
  class iterator
  {
    public:
      __device__ iterator(const string_view& str, size_type pos);
      iterator(const iterator& mit) = default;
      iterator(iterator&& mit) = default;
      __device__ iterator& operator++();
      __device__ iterator operator++(int);
      __device__ bool operator==(const iterator& rhs) const;
      __device__ bool operator!=(const iterator& rhs) const;
      __device__ Char operator*() const;
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
  __device__ Char at(size_type pos) const;
  __device__ Char operator[](size_type pos) const;
  /**---------------------------------------------------------------------------*
   * @brief Return the byte offset from data() for a given character position
   * 
   * @param pos Character position
   *---------------------------------------------------------------------------**/
  __device__ size_type byte_offset_for(size_type pos) const;

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
   * @brief Returns first character position if arg string is contained in this string.
   * 
   * @param str Target string to compare with this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type find( const string_view& str, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns first character position if arg array is contained in this string.
   * 
   * @param str Target string to compare with this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type find( const char* str, size_type bytes, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Returns first character position if arg character is contained in this string.
   * 
   * @param chr Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type find( Char chr, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Same as find() but searches from the end of this string.
   * 
   * @param str Target string to compare with this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type rfind( const string_view& str, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Same as find() but searches from the end of this string.
   * 
   * @param str Target string to compare with this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type rfind( const char* str, size_type bytes, size_type pos=0, size_type count=-1 ) const;
  /**---------------------------------------------------------------------------*
   * @brief Same as find() but searches from the end of this string.
   * 
   * @param chr Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   *---------------------------------------------------------------------------**/
  __device__ size_type rfind( Char chr, size_type pos=0, size_type count=-1 ) const;

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
   * @brief Tokenizes this string around the given delimiter up to count time.
   * 
   * @param delim Character to use for separating tokens.
   * @param count Maximum number of tokens to return.
   *              Specify -1 to indicate all tokens.
   * @param[out] Array to hold output tokens. 
   *             Specify nullptr here to return just the token count.
   * @return Number of tokens.
   *---------------------------------------------------------------------------**/
  __device__ size_type split( const char* delim, size_type count, string_view* strs ) const;

  /**---------------------------------------------------------------------------*
   * @brief Same as split() but starts tokenizing from the end of the string.
   * 
   * @param delim Character to use for separating tokens.
   * @param count Maximum number of tokens to return.
   *              Specify -1 to indicate all tokens.
   * @param[out] Array to hold output tokens. 
   *             Specify nullptr here to return just the token count.
   * @return Number of tokens.
   *---------------------------------------------------------------------------**/
  __device__ size_type rsplit( const char* delim, size_type count, string_view* strs ) const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of bytes in the specified character.
   *---------------------------------------------------------------------------**/
  __host__ __device__ static size_type bytes_in_char( Char chr );
  /**---------------------------------------------------------------------------*
   * @brief Convert a char array into a Char value.
   * 
   * @param str String containing encoded char bytes.
   * @param[out] chr Single Char value.
   * @return The number of bytes in the character
   *---------------------------------------------------------------------------**/
  __host__ __device__ static size_type char_to_Char( const char* str, Char& chr );
  /**---------------------------------------------------------------------------*
   * @brief Place a Char value into a char array.
   * 
   * @param chr Single character
   * @param[out] str Allocated char array with enough space to hold the encoded characer.
   * @return The number of bytes in the character
   *---------------------------------------------------------------------------**/
  __host__ __device__ static size_type Char_to_char( Char chr, char* str );
  /**---------------------------------------------------------------------------*
   * @brief Return the number of characters in this provided char array.
   * 
   * @param str String with encoded char bytes.
   * @param bytes Number of bytes in str.
   * @return The number of characters in the array.
   *---------------------------------------------------------------------------**/
  __host__ __device__ static size_type chars_in_string( const char* str, size_type bytes );

private:
    const char* _data{};   ///< Pointer to device memory contain char array for this string
    size_type _bytes{};    ///< Number of bytes in _data for this string

  /**---------------------------------------------------------------------------*
   * @brief Return the character position of the given byte offset.
   * 
   * @param bytepos Byte position from start of _data.
   * @return The character position for the specified byte.
   *---------------------------------------------------------------------------**/
    __device__ size_type char_offset(size_type bytepos) const;
};

}

#include "./string_view.inl"
