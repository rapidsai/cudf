/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/strings/string_view.cuh>

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
__device__ inline void integer_to_string(IntegerType value, char* d_buffer)
{
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
  // clang-format off
  size_type digits =
    (value < 10 ? 1 :
    (value < 100 ? 2 :
    (value < 1000 ? 3 :
    (value < 10000 ? 4 :
    (value < 100000 ? 5 :
    (value < 1000000 ? 6 :
    (value < 10000000 ? 7 :
    (value < 100000000 ? 8 :
    (value < 1000000000 ? 9 :
    (value < 10000000000 ? 10 :
    (value < 100000000000 ? 11 :
    (value < 1000000000000 ? 12 :
    (value < 10000000000000 ? 13 :
    (value < 100000000000000 ? 14 :
    (value < 1000000000000000 ? 15 :
    (value < 10000000000000000 ? 16 :
    (value < 100000000000000000 ? 17 :
    (value < 1000000000000000000 ? 18 :
    (value < 10000000000000000000 ? 19 :
    20)))))))))))))))))));
  // clang-format on
  return digits + static_cast<size_type>(is_negative);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
