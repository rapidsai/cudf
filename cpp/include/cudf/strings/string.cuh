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

#include <thrust/logical.h>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace string {
/**
 * @addtogroup strings_classes
 * @{
 * @file
 * @brief String functions
 */

/**
 * @brief Returns `true` if all characters in the string
 * are valid for conversion to an integer.
 *
 * Valid characters are in [-+0-9]. The sign character (+/-)
 * is optional but if present must be the first character.
 * An empty string returns `false`.
 * No bounds checking is performed to verify if the integer will fit
 * within a specific integer type.
 *
 * @param d_str String to check.
 * @return true if string has valid integer characters
 */
inline __device__ bool is_integer(string_view const& d_str)
{
  if (d_str.empty()) return false;
  auto begin = d_str.begin();
  auto end   = d_str.end();
  if (*begin == '+' || *begin == '-') ++begin;
  return (thrust::distance(begin, end) > 0) &&
         thrust::all_of(
           thrust::seq, begin, end, [] __device__(auto chr) { return chr >= '0' && chr <= '9'; });
}

/**
 * @brief Returns `true` if all characters in the string
 * are valid for conversion to a float type.
 *
 * Valid characters are in [-+0-9eE.]. The sign character (+/-)
 * is optional but if present must be the first character.
 * The sign character may also optionally appear right after the 'e' or 'E'
 * if the string is formatted with scientific notation.
 * The decimal character can appear only once and never after the
 * 'e' or 'E' character.
 * An empty string returns `false`.
 * No bounds checking is performed to verify if the value would fit
 * within a specific float type.
 * The following strings are also allowed "NaN", "Inf" and, "-Inf"
 * and will return true.
 *
 * @param d_str String to check.
 * @return true if string has valid float characters
 */
inline __device__ bool is_float(string_view const& d_str)
{
  if (d_str.empty()) return false;
  // strings allowed by the converter
  if (d_str.compare("NaN", 3) == 0) return true;
  if (d_str.compare("Inf", 3) == 0) return true;
  if (d_str.compare("-Inf", 4) == 0) return true;
  bool decimal_found  = false;
  bool exponent_found = false;
  size_type bytes     = d_str.size_bytes();
  const char* data    = d_str.data();
  // sign character allowed at the beginning of the string
  size_type chidx = (*data == '-' || *data == '+') ? 1 : 0;
  bool result     = chidx < bytes;
  // check for float chars [0-9] and a single decimal '.'
  // and scientific notation [eE][+-][0-9]
  for (; chidx < bytes; ++chidx) {
    auto chr = data[chidx];
    if (chr >= '0' && chr <= '9') continue;
    if (!decimal_found && chr == '.') {
      decimal_found = true;  // no more decimals
      continue;
    }
    if (!exponent_found && (chr == 'e' || chr == 'E')) {
      if (chidx + 1 < bytes) chr = data[chidx + 1];
      if (chr == '-' || chr == '+') ++chidx;
      decimal_found  = true;  // no decimal allowed in exponent
      exponent_found = true;  // no more exponents
      continue;
    }
    return false;
  }
  return result;
}

/** @} */  // end of group
}  // namespace string
}  // namespace strings
}  // namespace cudf
