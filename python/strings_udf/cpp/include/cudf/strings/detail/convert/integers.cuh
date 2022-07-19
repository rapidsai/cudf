
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

#include <cudf/strings/string_view.cuh>

#include <limits>
#include <type_traits>

namespace cudf {
namespace strings {
namespace detail {

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
  int64_t value = 0;
  auto bytes    = d_str.size_bytes();
  if (bytes == 0) return value;
  auto ptr  = d_str.data();
  auto sign = 1;
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
__device__ inline size_type integer_to_string(IntegerType value, char* d_buffer)
{
  if (value == 0) {
    *d_buffer = '0';
    return 1;
  }
  bool const is_negative = std::is_signed<IntegerType>() ? (value < 0) : false;

  constexpr IntegerType base = 10;
  // largest 64-bit integer is 20 digits; largest 128-bit integer is 39 digits
  constexpr int MAX_DIGITS = std::numeric_limits<IntegerType>::digits10 + 1;
  char digits[MAX_DIGITS];  // place-holder for digit chars
  int digits_idx = 0;
  while (value != 0) {
    assert(digits_idx < MAX_DIGITS);
    digits[digits_idx++] = '0' + abs(value % base);
    // next digit
    value = value / base;
  }
  size_type const bytes = digits_idx + static_cast<size_type>(is_negative);

  char* ptr = d_buffer;
  if (is_negative) *ptr++ = '-';
  // digits are backwards, reverse the string into the output
  while (digits_idx-- > 0) *ptr++ = digits[digits_idx];
  return bytes;
}

/**
 * @brief Counts number of digits in a integer value including '-' sign
 *
 * @tparam IntegerType integer type of input value
 * @param value input value to count the digits of
 * @return size_type number of digits in input value
 */
template <typename IntegerType>
constexpr size_type count_digits(IntegerType value)
{
  if (value == 0) return 1;
  bool const is_negative = std::is_signed<IntegerType>() ? (value < 0) : false;
  // std::numeric_limits<IntegerType>::min() is negative;
  // for all integer types, the max() and min() values have the same number of digits
  value = (value == std::numeric_limits<IntegerType>::min())
            ? std::numeric_limits<IntegerType>::max()
            : abs(value);

  auto const digits = [value] {
    // largest 8-byte  unsigned value is 18446744073709551615 (20 digits)
    // largest 16-byte unsigned value is 340282366920938463463374607431768211455 (39 digits)
    auto constexpr max_digits = std::numeric_limits<IntegerType>::digits10 + 1;

    size_type digits = 1;
    int64_t pow10    = 10;
    for (; digits < max_digits; ++digits, pow10 *= 10)
      if (value < pow10) break;
    return digits;
  }();

  return digits + static_cast<size_type>(is_negative);
}

__device__ int64_t hex_to_integer(string_view const& d_str)
{
  int64_t result = 0;
  int64_t base   = 1;
  auto const str = d_str.data();
  auto index     = d_str.size_bytes();
  while (index-- > 0) {
    auto const ch = str[index];
    if (ch >= '0' && ch <= '9') {
      result += static_cast<int64_t>(ch - 48) * base;
      base *= 16;
    } else if (ch >= 'A' && ch <= 'F') {
      result += static_cast<int64_t>(ch - 55) * base;
      base *= 16;
    } else if (ch >= 'a' && ch <= 'f') {
      result += static_cast<int64_t>(ch - 87) * base;
      base *= 16;
    }
  }
  return result;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
