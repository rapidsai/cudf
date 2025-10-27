/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/string_view.hpp>

#include <cuda_runtime.h>

// This header contains all class and function declarations so that it
// can be included in a .cpp file which only has declaration requirements
// (i.e. sizeof, conditionally-comparable, explicit conversions, etc).
// The definitions are coded in udf_string.cuh which is to be included
// in .cu files that use this class in kernel calls.

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Device string class for use with user-defined functions
 *
 * This class manages a device buffer of UTF-8 encoded characters
 * for string manipulation in a device kernel.
 *
 * Its methods and behavior are modelled after std::string but
 * with special consideration for UTF-8 encoded strings and for
 * use within a cuDF UDF.
 */
class udf_string {
 public:
  /**
   * @brief Represents unknown character position or length
   */
  static constexpr cudf::size_type npos = static_cast<cudf::size_type>(-1);

  /**
   * @brief Cast to cudf::string_view operator
   */
  __device__ operator cudf::string_view() const { return cudf::string_view(m_data, m_bytes); }

  /**
   * @brief Create an empty string.
   */
  udf_string() = default;

  /**
   * @brief Create a string using existing device memory
   *
   * The given memory is copied into the instance returned.
   *
   * @param data Device pointer to UTF-8 encoded string
   * @param bytes Number of bytes in `data`
   */
  __device__ udf_string(char const* data, cudf::size_type bytes);

  /**
   * @brief Create a string object from a null-terminated character array
   *
   * The given memory is copied into the instance returned.
   *
   * @param data Device pointer to UTF-8 encoded null-terminated
   *             character array.
   */
  __device__ udf_string(char const* data);

  /**
   * @brief Create a string object from a cudf::string_view
   *
   * The input string data is copied into the instance returned.
   *
   * @param str String to copy
   */
  __device__ udf_string(cudf::string_view str);

  /**
   * @brief Create a string object with `count` copies of character `chr`
   *
   * @param count Number of times to copy `chr`
   * @param chr Character from which to create the string
   */
  __device__ udf_string(cudf::size_type count, cudf::char_utf8 chr);

  /**
   * @brief Create a string object from another instance
   *
   * The string data is copied from the `src` into the instance returned.
   *
   * @param src String to copy
   */
  __device__ udf_string(udf_string const& src);

  /**
   * @brief Move a string object from an rvalue reference
   *
   * The string data is moved from `src` into the instance returned.
   * The `src` will have no content.
   *
   * @param src String to copy
   */
  __device__ udf_string(udf_string&& src) noexcept;

  __device__ ~udf_string();

  __device__ udf_string& operator=(udf_string const&);
  __device__ udf_string& operator=(udf_string&&) noexcept;
  __device__ udf_string& operator=(cudf::string_view const);
  __device__ udf_string& operator=(char const*);

  /**
   * @brief Return the number of bytes in this string
   */
  __device__ cudf::size_type size_bytes() const noexcept;

  /**
   * @brief Return the number of characters in this string
   */
  __device__ cudf::size_type length() const noexcept;

  /**
   * @brief Return the maximum number of bytes a udf_string can hold
   */
  __device__ constexpr cudf::size_type max_size() const noexcept;

  /**
   * @brief Return the internal pointer to the character array for this object
   */
  __device__ char* data() noexcept;
  __device__ char const* data() const noexcept;

  /**
   * @brief Returns true if there are no characters in this string
   */
  __device__ bool is_empty() const noexcept;

  /**
   * @brief Returns an iterator that can be used to navigate through
   *        the UTF-8 characters in this string
   *
   * This returns a `cudf::string_view::const_iterator` which is read-only.
   */
  __device__ cudf::string_view::const_iterator begin() const noexcept;
  __device__ cudf::string_view::const_iterator end() const noexcept;

  /**
   * @brief Returns the character at the specified position
   *
   * This will return 0 if `pos >= length()`.
   *
   * @param pos Index position of character to return
   * @return Character at position `pos`
   */
  __device__ cudf::char_utf8 at(cudf::size_type pos) const;

  /**
   * @brief Returns the character at the specified index
   *
   * This will return 0 if `pos >= length()`.
   * Note this is read-only. Use replace() to modify a character.
   *
   * @param pos Index position of character to return
   * @return Character at position `pos`
   */
  __device__ cudf::char_utf8 operator[](cudf::size_type pos) const;

  /**
   * @brief Return the byte offset for a given character position
   *
   * The byte offset for the character at `pos` such that
   * `data() + byte_offset(pos)` points to the memory location
   * the character at position `pos`.
   *
   * The behavior is undefined if `pos < 0 or pos >= length()`
   *
   * @param pos Index position of character to return byte offset.
   * @return Byte offset for character at `pos`
   */
  __device__ cudf::size_type byte_offset(cudf::size_type pos) const;

  /**
   * @brief Comparing target string with this string
   *
   * @param str Target string to compare with this string
   * @return 0  If they compare equal
   *         <0 Either the value of the first character of this string that does
   *            not match is ordered before the corresponding character in `str`,
   *            or all compared characters match but the `str` string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is ordered after the corresponding character in `str`,
   *            or all compared characters match but the `str` string is longer.
   */
  __device__ int compare(cudf::string_view str) const noexcept;

  /**
   * @brief Comparing target character array with this string
   *
   * @param str Target array of UTF-8 characters.
   * @param bytes Number of bytes in `str`.
   * @return 0  If they compare equal
   *         <0 Either the value of the first character of this string that does
   *            not match is ordered before the corresponding character in `str`,
   *            or all compared characters match but `bytes < size_bytes()`.
   *         >0 Either the value of the first character of this string that does
   *            not match is ordered after the corresponding character in `str`,
   *            or all compared characters match but `bytes > size_bytes()`.
   */
  __device__ int compare(char const* str, cudf::size_type bytes) const;

  /**
   * @brief Returns true if `rhs` matches this string exactly
   */
  __device__ bool operator==(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if `rhs` does not match this string
   */
  __device__ bool operator!=(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if this string is ordered before `rhs`
   */
  __device__ bool operator<(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if `rhs` is ordered before this string
   */
  __device__ bool operator>(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if this string matches or is ordered before `rhs`
   */
  __device__ bool operator<=(cudf::string_view rhs) const noexcept;

  /**
   * @brief Returns true if `rhs` matches or is ordered before this string
   */
  __device__ bool operator>=(cudf::string_view rhs) const noexcept;

  /**
   * @brief Remove all bytes from this string
   *
   * All pointers, references, and iterators are invalidated.
   */
  __device__ void clear() noexcept;

  /**
   * @brief Resizes string to contain `count` bytes
   *
   * If `count > size_bytes()` then zero-padding is added.
   * If `count < size_bytes()` then the string is truncated to size `count`.
   *
   * All pointers, references, and iterators may be invalidated.
   *
   * The behavior is undefined if `count > max_size()`
   *
   * @param count Size in bytes of this string.
   */
  __device__ void resize(cudf::size_type count);

  /**
   * @brief Reserve `count` bytes in this string
   *
   * If `count > capacity()`, new memory is allocated and `capacity()` will
   * be greater than or equal to `count`.
   * There is no effect if `count <= capacity()`.
   *
   * @param count Total number of bytes to reserve for this string
   */
  __device__ void reserve(cudf::size_type count);

  /**
   * @brief Returns the number of bytes that the string has allocated
   */
  __device__ cudf::size_type capacity() const noexcept;

  /**
   * @brief Reduces internal allocation to just `size_bytes()`
   *
   * All pointers, references, and iterators may be invalidated.
   */
  __device__ void shrink_to_fit();

  /**
   * @brief Moves the contents of `str` into this string instance
   *
   * On return, the `str` will have no contents.
   *
   * @param str String to move
   * @return This string with new contents
   */
  __device__ udf_string& assign(udf_string&& str) noexcept;

  /**
   * @brief Replaces the contents of this string with contents of `str`
   *
   * @param str String to copy
   * @return This string with new contents
   */
  __device__ udf_string& assign(cudf::string_view str);

  /**
   * @brief Replaces the contents of this string with contents of `str`
   *
   * @param str Null-terminated UTF-8 character array
   * @return This string with new contents
   */
  __device__ udf_string& assign(char const* str);

  /**
   * @brief Replaces the contents of this string with contents of `str`
   *
   * @param str UTF-8 character array
   * @param bytes Number of bytes to copy from `str`
   * @return This string with new contents
   */
  __device__ udf_string& assign(char const* str, cudf::size_type bytes);

  /**
   * @brief Append a string to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ udf_string& operator+=(cudf::string_view str);

  /**
   * @brief Append a character to the end of this string
   *
   * @param str Character to append
   * @return This string with the appended argument
   */
  __device__ udf_string& operator+=(cudf::char_utf8 chr);

  /**
   * @brief Append a null-terminated device memory character array
   * to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ udf_string& operator+=(char const* str);

  /**
   * @brief Append a null-terminated character array to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ udf_string& append(char const* str);

  /**
   * @brief Append a character array to the end of this string
   *
   * @param str Character array to append
   * @param bytes Number of bytes from `str` to append.
   * @return This string with the appended argument
   */
  __device__ udf_string& append(char const* str, cudf::size_type bytes);

  /**
   * @brief Append a string to the end of this string
   *
   * @param str String to append
   * @return This string with the appended argument
   */
  __device__ udf_string& append(cudf::string_view str);

  /**
   * @brief Append a character to the end of this string
   * a specified number of times.
   *
   * @param chr Character to append
   * @param count Number of times to append `chr`
   * @return This string with the append character(s)
   */
  __device__ udf_string& append(cudf::char_utf8 chr, cudf::size_type count = 1);

  /**
   * @brief Insert a string into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param str String to insert into this one
   * @return This string with the inserted argument
   */
  __device__ udf_string& insert(cudf::size_type pos, cudf::string_view str);

  /**
   * @brief Insert a null-terminated character array into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param data Null-terminated character array to insert
   * @return This string with the inserted argument
   */
  __device__ udf_string& insert(cudf::size_type pos, char const* data);

  /**
   * @brief Insert a character array into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param data Character array to insert
   * @param bytes Number of bytes from `data` to insert
   * @return This string with the inserted argument
   */
  __device__ udf_string& insert(cudf::size_type pos, char const* data, cudf::size_type bytes);

  /**
   * @brief Insert a character one or more times into the character position specified
   *
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * @param pos Character position to begin insert
   * @param count Number of times to insert `chr`
   * @param chr Character to insert
   * @return This string with the inserted argument
   */
  __device__ udf_string& insert(cudf::size_type pos, cudf::size_type count, cudf::char_utf8 chr);

  /**
   * @brief Returns a substring of this string
   *
   * An empty string is returned if `pos < 0 or pos >= length()`.
   *
   * @param pos Character position to start the substring
   * @param count Number of characters for the substring;
   *              This can be greater than the number of available characters.
   *              Default npos returns characters in range `[pos, length())`.
   * @return New string with the specified characters
   */
  __device__ udf_string substr(cudf::size_type pos, cudf::size_type count = npos) const;

  /**
   * @brief Replace a range of characters with a given string
   *
   * Replaces characters in range `[pos, pos + count]` with `str`.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `str` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param str String to replace the given range
   * @return This string modified with the replacement
   */
  __device__ udf_string& replace(cudf::size_type pos, cudf::size_type count, cudf::string_view str);

  /**
   * @brief Replace a range of characters with a null-terminated character array
   *
   * Replaces characters in range `[pos, pos + count)` with `data`.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `data` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param data Null-terminated character array to replace the given range
   * @return This string modified with the replacement
   */
  __device__ udf_string& replace(cudf::size_type pos, cudf::size_type count, char const* data);

  /**
   * @brief Replace a range of characters with a given character array
   *
   * Replaces characters in range `[pos, pos + count)` with `[data, data + bytes)`.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `data` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param data String to replace the given range
   * @param bytes Number of bytes from data to use for replacement
   * @return This string modified with the replacement
   */
  __device__ udf_string& replace(cudf::size_type pos,
                                 cudf::size_type count,
                                 char const* data,
                                 cudf::size_type bytes);

  /**
   * @brief Replace a range of characters with a character one or more times
   *
   * Replaces characters in range `[pos, pos + count)` with `chr` `chr_count` times.
   * There is no effect if `pos < 0 or pos > length()`.
   *
   * If `count==0` then `chr` is inserted starting at `pos`.
   * If `count==npos` then the replacement range is `[pos,length())`.
   *
   * @param pos Position of first character to replace
   * @param count Number of characters to replace
   * @param chr_count Number of times `chr` will repeated
   * @param chr Character to use for replacement
   * @return This string modified with the replacement
   */
  __device__ udf_string& replace(cudf::size_type pos,
                                 cudf::size_type count,
                                 cudf::size_type chr_count,
                                 cudf::char_utf8 chr);

  /**
   * @brief Removes specified characters from this string
   *
   * Removes `min(count, length() - pos)` characters starting at `pos`.
   * There is no effect if `pos < 0 or pos >= length()`.
   *
   * @param pos Character position to begin insert
   * @param count Number of characters to remove starting at `pos`
   * @return This string with remove characters
   */
  __device__ udf_string& erase(cudf::size_type pos, cudf::size_type count = npos);

 private:
  char* m_data{};
  cudf::size_type m_bytes{};
  cudf::size_type m_capacity{};

  // utilities
  __device__ char* allocate(cudf::size_type bytes);
  __device__ void deallocate(char* data);
  __device__ void reallocate(cudf::size_type bytes);
  __device__ cudf::size_type char_offset(cudf::size_type byte_pos) const;
  __device__ void shift_bytes(cudf::size_type start_pos,
                              cudf::size_type end_pos,
                              cudf::size_type nbytes);
};

}  // namespace udf
}  // namespace strings
}  // namespace cudf
