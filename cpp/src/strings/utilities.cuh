/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>

#include <cstring>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param input Data to copy from.
 * @param bytes Number of bytes to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_and_increment(char* buffer, const char* input, size_type bytes)
{
  memcpy(buffer, input, bytes);
  return buffer + bytes;
}

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param d_string String to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_string(char* buffer, const string_view& d_string)
{
  return copy_and_increment(buffer, d_string.data(), d_string.size_bytes());
}

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output.
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must also have members d_offsets and d_chars which are set to
 *         memory containing the offsets and chars columns during write.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string.
 *        After that, the d_offsets and d_chars are set and this is called again to fill in the
 *        chars memory.
 * @param strings_count Number of strings.
 * @param null_count Number of nulls in the strings column.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return offsets child column and chars child column for a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(SizeAndExecuteFunction size_and_exec_fn,
                           size_type strings_count,
                           size_type null_count,
                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                           cudaStream_t stream                 = 0)
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view          = offsets_column->mutable_view();
  auto d_offsets             = offsets_view.template data<int32_t>();
  size_and_exec_fn.d_offsets = d_offsets;

  // This is called twice -- once for offsets and once for chars.
  // Reducing the number of places size_and_exec_fn is inlined speeds up compile time.
  auto for_each_fn = [strings_count, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       size_and_exec_fn);
  };

  // Compute the offsets values
  for_each_fn(size_and_exec_fn);
  thrust::exclusive_scan(
    rmm::exec_policy(stream)->on(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // Now build the chars column
  std::unique_ptr<column> chars_column = create_chars_child_column(
    strings_count, null_count, thrust::device_pointer_cast(d_offsets)[strings_count], mr, stream);
  size_and_exec_fn.d_chars = chars_column->mutable_view().template data<char>();
  for_each_fn(size_and_exec_fn);

  return std::make_pair(std::move(offsets_column), std::move(chars_column));
}

/**
 * @brief Converts a single UTF-8 character into a code-point value that
 * can be used for lookup in the character flags or the character case tables.
 *
 * @param utf8_char Single UTF-8 character to convert.
 * @return Code-point for the UTF-8 character.
 */
__device__ inline uint32_t utf8_to_codepoint(cudf::char_utf8 utf8_char)
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
__host__ __device__ inline cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
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

/**
 * @brief Converts a single string into an integer.
 *
 * The '+' and '-' are allowed but only at the beginning of the string.
 * The string is expected to contain base-10 [0-9] characters only.
 * Any other character will end the parse.
 * Overflow of the int64 type is not detected.
 */
__device__ inline int64_t string_to_integer(string_view const& d_str)
{
  int64_t value   = 0;
  size_type bytes = d_str.size_bytes();
  if (bytes == 0) return value;
  const char* ptr = d_str.data();
  int sign        = 1;
  if (*ptr == '-' || *ptr == '+') {
    sign = (*ptr == '-' ? -1 : 1);
    ++ptr;
    --bytes;
  }
  for (size_type idx = 0; idx < bytes; ++idx) {
    char chr = *ptr++;
    if (chr < '0' || chr > '9') break;
    value = (value * 10) + static_cast<int64_t>(chr - '0');
  }
  return value * static_cast<int64_t>(sign);
}
/**
 * @brief Converts an integer into string
 *
 * @tparam IntegerType integer type to convert from
 * @param value integer value to convert
 * @param d_buffer character buffer to store the converted string
 */
template <typename IntegerType>
__device__ inline void integer_to_string(IntegerType value, char* d_buffer) {
    if (value == 0) {
      *d_buffer = '0';
      return;
    }
    bool is_negative = std::is_signed<IntegerType>::value ? (value < 0) : false;
    //
    constexpr IntegerType base = 10;
    constexpr int MAX_DIGITS   = 20;  // largest 64-bit integer is 20 digits
    char digits[MAX_DIGITS];          // place-holder for digit chars
    int digits_idx = 0;
    while (value != 0) {
      assert(digits_idx < MAX_DIGITS);
      digits[digits_idx++] = '0' + cudf::util::absolute_value(value % base);
      // next digit
      value = value / base;
    }
    char* ptr = d_buffer;
    if (is_negative) *ptr++ = '-';
    // digits are backwards, reverse the string into the output
    while (digits_idx-- > 0) *ptr++ = digits[digits_idx];
}

/**
 * @brief Counts number of digits in a integer value including '-' sign
 *
 * @tparam IntegerType integer type of input value
 * @param value input value to count the digits of
 * @return size_type number of digits in input value
 */
template <typename IntegerType>
__device__ inline size_type count_digits(IntegerType value)
{
  if (value == 0) return 1;
  bool is_negative = std::is_signed<IntegerType>::value ? (value < 0) : false;
  // abs(std::numeric_limits<IntegerType>::min()) is negative;
  // for all integer types, the max() and min() values have the same number of digits
  value = (value == std::numeric_limits<IntegerType>::min())
            ? std::numeric_limits<IntegerType>::max()
            : cudf::util::absolute_value(value);
  // largest 8-byte unsigned value is 18446744073709551615 (20 digits)
  size_type digits =
    (value < 10
       ? 1
       : (value < 100
            ? 2
            : (value < 1000
                 ? 3
                 : (value < 10000
                      ? 4
                      : (value < 100000
                           ? 5
                           : (value < 1000000
                                ? 6
                                : (value < 10000000
                                     ? 7
                                     : (value < 100000000
                                          ? 8
                                          : (value < 1000000000
                                               ? 9
                                               : (value < 10000000000
                                                    ? 10
                                                    : (value < 100000000000
                                                         ? 11
                                                         : (value < 1000000000000
                                                              ? 12
                                                              : (value < 10000000000000
                                                                   ? 13
                                                                   : (value < 100000000000000
                                                                        ? 14
                                                                        : (value < 1000000000000000
                                                                             ? 15
                                                                             : (value <
                                                                                    10000000000000000
                                                                                  ? 16
                                                                                  : (value <
                                                                                         100000000000000000
                                                                                       ? 17
                                                                                       : (value <
                                                                                              1000000000000000000
                                                                                            ? 18
                                                                                            : (value <
                                                                                                   10000000000000000000
                                                                                                 ? 19
                                                                                                 : 20)))))))))))))))))));
  return digits + static_cast<size_type>(is_negative);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
