
/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

namespace CUDF_EXPORT cudf {

/**
 * @brief A mutable string buffer with fixed capacity.
 *
 */
struct buffer_string {
 private:
  char* m_data               = nullptr;  ///< Pointer to the character buffer.
  cudf::size_type m_capacity = 0;        ///< Capacity of the buffer in bytes.
  cudf::size_type m_size     = 0;        ///< Size of the string in bytes.

 public:
  /**
   * @brief Constructs a buffer_string with a given buffer and capacity.
   *
   * @param buffer Pointer to the pre-allocated character buffer.
   * @param capacity Maximum capacity of the buffer in bytes.
   */
  CUDF_HOST_DEVICE buffer_string(char* buffer, cudf::size_type capacity)
    : m_data(buffer), m_capacity(capacity), m_size(0)
  {
  }

  /**
   * @brief Default constructor for buffer_string.
   */
  CUDF_HOST_DEVICE buffer_string() = default;

  buffer_string(buffer_string const&) = delete;

  buffer_string& operator=(buffer_string const&) = delete;

  /**
   * @brief Move constructor for buffer_string.
   *
   * Transfers ownership of the buffer from another buffer_string instance.
   *
   * @param other The buffer_string instance to move from.
   */
  __device__ buffer_string(buffer_string&& other)
    : m_data(other.m_data), m_capacity(other.m_capacity), m_size(other.m_size)
  {
    other.m_data     = nullptr;
    other.m_capacity = 0;
    other.m_size     = 0;
  }

  /**
   * @brief Move assignment operator for buffer_string.
   *
   * Transfers ownership of the buffer from another buffer_string instance.
   *
   * @param other The buffer_string instance to move from.
   * @return Reference to the current instance.
   */
  __device__ buffer_string& operator=(buffer_string&& other)
  {
    if (this == &other) { return *this; }

    m_data           = other.m_data;
    m_capacity       = other.m_capacity;
    m_size           = other.m_size;
    other.m_data     = nullptr;
    other.m_capacity = 0;
    other.m_size     = 0;
    return *this;
  }

  __device__ ~buffer_string() = default;

  /**
   * @brief Returns the number of characters in the string.
   *
   * @return The number of characters in the string.
   */
  __device__ cudf::size_type length() const
  {
    return cudf::strings::detail::characters_in_string(m_data, m_size);
  }

  /**
   * @brief Returns the size of the string in bytes.
   *
   * @return The size of the string in bytes.
   */
  CUDF_HOST_DEVICE cudf::size_type size_bytes() const { return m_size; }

  /**
   * @brief Returns the capacity of the buffer in bytes.
   *
   * @return The capacity of the buffer in bytes.
   */
  CUDF_HOST_DEVICE cudf::size_type capacity_bytes() const { return m_capacity; }

  /**
   * @brief Returns a `cudf::string_view` representing the current string.
   *
   * @return A `cudf::string_view` of the current string.
   */
  CUDF_HOST_DEVICE cudf::string_view view() const { return cudf::string_view{m_data, m_size}; }

  /**
   * @brief Implicit conversion operator to `cudf::string_view`.
   *
   * @return A `cudf::string_view` of the current string.
   */
  CUDF_HOST_DEVICE operator cudf::string_view() const { return view(); }

  /**
   * @brief Clears the string, setting its size to zero.
   */
  CUDF_HOST_DEVICE void clear() { m_size = 0; }

  /**
   * @brief Resets the string, setting its size to zero.
   */
  CUDF_HOST_DEVICE void reset() { m_size = 0; }

  /**
   * @brief Returns a pointer to the character buffer.
   *
   * @return Pointer to the character buffer.
   */
  CUDF_HOST_DEVICE char* data() { return m_data; }

  /**
   * @brief Checks if the string is empty.
   *
   * @return True if the string is empty, false otherwise.
   */
  CUDF_HOST_DEVICE bool is_empty() const { return m_size == 0; }

  /**
   * @brief Returns an iterator to the beginning of the string.
   *
   * @return An iterator to the beginning of the string.
   */
  __device__ cudf::string_view::const_iterator begin() const { return view().begin(); }

  /**
   * @brief Returns an iterator to the end of the string.
   *
   * @return An iterator to the end of the string.
   */
  __device__ cudf::string_view::const_iterator end() const { return view().end(); }

  /**
   * @brief Returns the character at the specified position.
   *
   * @param pos The position of the character.
   * @return The character at the specified position.
   */
  __device__ cudf::char_utf8 at(cudf::size_type pos) const
  {
    assert(pos < m_size && "Index out of bounds");
    return view()[pos];
  }

  /**
   * @brief Accesses the character at the specified position.
   *
   * @param pos The position of the character.
   * @return The character at the specified position.
   */
  __device__ cudf::char_utf8 operator[](cudf::size_type pos) const { return view()[pos]; }

  /**
   * @brief Compares the string with another `cudf::string_view`.
   *
   * @param in The `cudf::string_view` to compare with.
   * @return An integer indicating the result of the comparison.
   */
  __device__ int compare(cudf::string_view in) const { return view().compare(in); }

  /**
   * @brief Compares the string with a raw character buffer.
   *
   * @param data Pointer to the character buffer.
   * @param bytes Size of the character buffer in bytes.
   * @return An integer indicating the result of the comparison.
   */
  __device__ int compare(char const* data, cudf::size_type bytes) const
  {
    return view().compare(data, bytes);
  }

  /**
   * @brief Equality comparison operator with `cudf::string_view`.
   *
   * @param rhs The `cudf::string_view` to compare with.
   * @return True if the strings are equal, false otherwise.
   */
  __device__ bool operator==(cudf::string_view rhs) const { return view() == rhs; }

  /**
   * @brief Inequality comparison operator with `cudf::string_view`.
   *
   * @param rhs The `cudf::string_view` to compare with.
   * @return True if the strings are not equal, false otherwise.
   */
  __device__ bool operator!=(cudf::string_view rhs) const { return view() != rhs; }

  /**
   * @brief Less-than comparison operator with `cudf::string_view`.
   *
   * @param rhs The `cudf::string_view` to compare with.
   * @return True if the current string is less than rhs, false otherwise.
   */
  __device__ bool operator<(cudf::string_view rhs) const { return view() < rhs; }

  /**
   * @brief Greater-than comparison operator with `cudf::string_view`.
   *
   * @param rhs The `cudf::string_view` to compare with.
   * @return True if the current string is greater than rhs, false otherwise.
   */
  __device__ bool operator>(cudf::string_view rhs) const { return view() > rhs; }

  /**
   * @brief Less-than-or-equal comparison operator with `cudf::string_view`.
   *
   * @param rhs The `cudf::string_view` to compare with.
   * @return True if the current string is less than or equal to rhs, false otherwise.
   */
  __device__ bool operator<=(cudf::string_view rhs) const { return view() <= rhs; }

  /**
   * @brief Greater-than-or-equal comparison operator with cudf::string_view.
   *
   * @param rhs The `cudf::string_view` to compare with.
   * @return True if the current string is greater than or equal to rhs, false otherwise.
   */
  __device__ bool operator>=(cudf::string_view rhs) const { return view() >= rhs; }

 private:
  /**
   * @brief Resizes the string without initializing new bytes.
   *
   * @param size The new size of the string in bytes.
   * @return True if the resize was successful, false otherwise.
   */
  __device__  bool resize_bytes_uninitialized(cudf::size_type size)
  {
    if (size > m_capacity) { return false; }
    m_size = size;
    return true;
  }

  /**
   * @brief Resizes the string to the specified byte size.
   *
   * @param size The new size of the string in bytes.
   * @return True if the resize was successful, false otherwise.
   */
  __device__  bool resize_bytes(cudf::size_type size)
  {
    auto const old_size = m_size;
    if (!resize_bytes_uninitialized(size)) { return false; }
    if (size > m_size) { memset(m_data + old_size, 0, size - m_size); }
    return true;
  }

  /**
   * @brief Shifts bytes within the buffer to accommodate modifications.
   *
   * @param start_pos The starting position for the shift.
   * @param end_pos The ending position for the shift.
   * @param nbytes The new size of the string in bytes.
   */
  __device__ void shift_bytes(cudf::size_type start_pos,
                              cudf::size_type end_pos,
                              cudf::size_type nbytes)
  {
    if (nbytes < m_size) {
      // shift bytes to the left [...wxyz] -> [wxyzxyz]
      auto src = end_pos;
      auto tgt = start_pos;
      while (tgt < nbytes) {
        m_data[tgt++] = m_data[src++];
      }
    } else if (nbytes > m_size) {
      // shift bytes to the right [abcd...] -> [abcabcd]
      auto src = m_size;
      auto tgt = nbytes;
      while (src > end_pos) {
        m_data[--tgt] = m_data[--src];
      }
    }
  }

 public:
  /**
   * @brief Resizes the string to the specified byte size.
   *
   * @param size The new size of the string in bytes.
   * @return True if the resize was successful, false otherwise.
   */
  __device__  bool resize(cudf::size_type size) { return resize_bytes(size); }

  /**
   * @brief Appends a `cudf::string_view` to the string.
   *
   * @param str The `cudf::string_view` to append.
   * @return True if the append was successful, false otherwise.
   */
  __device__  bool append(cudf::string_view str)
  {
    auto old_size = m_size;
    if (!resize_bytes_uninitialized(m_size + str.size_bytes())) { return false; }
    memcpy(m_data + old_size, str.data(), str.size_bytes());
    return true;
  }

  __device__  bool append(cudf::char_utf8 chr, cudf::size_type count)
  {
    auto bytes          = cudf::strings::detail::bytes_in_char_utf8(chr) * count;
    auto const old_size = m_size;
    if (!resize_bytes_uninitialized(m_size + bytes)) { return false; }

    auto out = m_data + old_size;

    for (cudf::size_type idx = 0; idx < count; ++idx) {
      out += cudf::strings::detail::from_char_utf8(chr, out);
    }

    return true;
  }

  __device__  bool replace(cudf::size_type pos,
                                        cudf::size_type count,
                                        cudf::string_view in)
  {
    if (pos < 0 || in.size_bytes() < 0) {
      assert(false);  // this is not a valid state and should never happen
      return true;
    }

    auto const start_pos = view().byte_offset(pos);

    if (start_pos > m_size) { return true; }

    auto const end_pos = count < 0 ? m_size : std::min(view().byte_offset(pos + count), m_size);

    // compute new size
    auto const nbytes = m_size + in.size_bytes() - (end_pos - start_pos);
    if (!resize_bytes_uninitialized(nbytes)) { return false; }

    // move bytes -- make room for replacement
    shift_bytes(start_pos + in.size_bytes(), end_pos, nbytes);

    // insert the replacement
    memcpy(m_data + start_pos, in.data(), in.size_bytes());

    return true;
  }

  __device__  bool insert(cudf::size_type pos, cudf::string_view in)
  {
    return replace(pos, 0, in);
  }

  __device__ void erase(cudf::size_type pos, cudf::size_type count)
  {
    [[maybe_unused]] auto success = replace(pos, count, {});
  }
};

}  // namespace cudf
