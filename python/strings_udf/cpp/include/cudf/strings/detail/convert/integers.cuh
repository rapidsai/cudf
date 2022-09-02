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
 * Copied from cudf/cpp/include/cudf/strings/detail/convert/int_to_string.cuh
 * Dependencies there cannot be compiled by jitify.
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
  while (digits_idx-- > 0)
    *ptr++ = digits[digits_idx];
  return bytes;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
