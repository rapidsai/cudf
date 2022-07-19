
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

#include <cmath>
#include <limits>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Returns true if input contains the not-a-number string.
 *
 * The following are valid for this function: "NAN" and "NaN"
 * @param d_str input string
 * @return true if input is as valid NaN string.
 */
inline __device__ bool is_nan_str(string_view const& d_str)
{
  auto const ptr = d_str.data();
  return (d_str.size_bytes() == 3) && (ptr[0] == 'N' || ptr[0] == 'n') &&
         (ptr[1] == 'A' || ptr[1] == 'a') && (ptr[2] == 'N' || ptr[2] == 'n');
}

/**
 * @brief Returns true if input contains the infinity string.
 *
 * The following are valid for this function: "INF", "INFINITY", and "Inf"
 * @param d_str input string
 * @return true if input is as valid Inf string.
 */
inline __device__ bool is_inf_str(string_view const& d_str)
{
  auto const ptr  = d_str.data();
  auto const size = d_str.size_bytes();

  if (size != 3 && size != 8) return false;

  auto const prefix_valid = (ptr[0] == 'I' || ptr[0] == 'i') && (ptr[1] == 'N' || ptr[1] == 'n') &&
                            (ptr[2] == 'F' || ptr[2] == 'f');

  return prefix_valid &&
         ((size == 3) || ((ptr[3] == 'I' || ptr[3] == 'i') && (ptr[4] == 'N' || ptr[4] == 'n') &&
                          (ptr[5] == 'I' || ptr[5] == 'i') && (ptr[6] == 'T' || ptr[6] == 't') &&
                          (ptr[7] == 'Y' || ptr[7] == 'y')));
}

__device__ inline double stod(string_view const& d_str)
{
  const char* in_ptr = d_str.data();
  const char* end    = in_ptr + d_str.size_bytes();
  if (end == in_ptr) return 0.0;
  double sign{1.0};
  if (*in_ptr == '-' || *in_ptr == '+') {
    sign = (*in_ptr == '-' ? -1 : 1);
    ++in_ptr;
  }

  // could not find INFINITY and std::numeric_limits<double>::infinity() does not work;
  // same for std::numeric_limits<double>::quiet_NaN() but looks like nan() works ok
  constexpr double infinity = (1.0 / 0.0);

  // special strings: NaN, Inf
  if ((in_ptr < end) && *in_ptr > '9') {
    auto const inf_nan = string_view(in_ptr, static_cast<size_type>(end - in_ptr));
    if (is_nan_str(inf_nan)) return nan("");
    if (is_inf_str(inf_nan)) return sign * infinity;
  }

  // Parse and store the mantissa as much as we can,
  // until we are about to exceed the limit of uint64_t
  constexpr uint64_t max_holding = (18446744073709551615U - 9U) / 10U;
  uint64_t digits                = 0;
  int exp_off                    = 0;
  bool decimal                   = false;
  while (in_ptr < end) {
    char ch = *in_ptr;
    if (ch == '.') {
      decimal = true;
      ++in_ptr;
      continue;
    }
    if (ch < '0' || ch > '9') break;
    if (digits > max_holding)
      exp_off += (int)!decimal;
    else {
      digits = (digits * 10L) + static_cast<uint64_t>(ch - '0');
      if (digits > max_holding) {
        digits = digits / 10L;
        exp_off += (int)!decimal;
      } else
        exp_off -= (int)decimal;
    }
    ++in_ptr;
  }
  if (digits == 0) return sign * static_cast<double>(0);

  // check for exponent char
  int exp_ten  = 0;
  int exp_sign = 1;
  if (in_ptr < end) {
    char ch = *in_ptr++;
    if (ch == 'e' || ch == 'E') {
      if (in_ptr < end) {
        ch = *in_ptr;
        if (ch == '-' || ch == '+') {
          exp_sign = (ch == '-' ? -1 : 1);
          ++in_ptr;
        }
        while (in_ptr < end) {
          ch = *in_ptr++;
          if (ch < '0' || ch > '9') break;
          exp_ten = (exp_ten * 10) + (int)(ch - '0');
        }
      }
    }
  }

  int const num_digits = static_cast<int>(log10((double)digits)) + 1;
  exp_ten *= exp_sign;
  exp_ten += exp_off;
  exp_ten += num_digits - 1;
  if (exp_ten > std::numeric_limits<double>::max_exponent10) {
    return sign > 0 ? infinity : -infinity;
  }

  double base = sign * static_cast<double>(digits);

  exp_ten += 1 - num_digits;
  // If 10^exp_ten would result in a subnormal value, the base and
  // exponent should be adjusted so that 10^exp_ten is a normal value
  auto const subnormal_shift = std::numeric_limits<double>::min_exponent10 - exp_ten;
  if (subnormal_shift > 0) {
    // Handle subnormal values. Ensure that both base and exponent are
    // normal values before computing their product.
    base = base / exp10(static_cast<double>(num_digits - 1 + subnormal_shift));
    exp_ten += num_digits - 1;  // adjust exponent
    auto const exponent = exp10(static_cast<double>(exp_ten + subnormal_shift));
    return base * exponent;
  }

  double const exponent = exp10(static_cast<double>(std::abs(exp_ten)));
  return exp_ten < 0 ? base / exponent : base * exponent;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
