/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/string_view.cuh>

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
 * The following strings are also allowed and will return true:
 *  "NaN", "NAN", "Inf", "INF", "INFINITY"
 *
 * @param d_str String to check.
 * @return true if string has valid float characters
 */
inline __device__ bool is_float(string_view const& d_str)
{
  if (d_str.empty()) return false;
  bool decimal_found  = false;
  bool exponent_found = false;
  size_type bytes     = d_str.size_bytes();
  char const* data    = d_str.data();
  // sign character allowed at the beginning of the string
  size_type ch_idx = (*data == '-' || *data == '+') ? 1 : 0;

  bool result = ch_idx < bytes;
  // check for nan and infinity strings
  if (result && data[ch_idx] > '9') {
    auto const inf_nan = string_view(data + ch_idx, bytes - ch_idx);
    if (is_nan_str(inf_nan) || is_inf_str(inf_nan)) return true;
  }

  // check for float chars [0-9] and a single decimal '.'
  // and scientific notation [eE][+-][0-9]
  for (; ch_idx < bytes; ++ch_idx) {
    auto chr = data[ch_idx];
    if (chr >= '0' && chr <= '9') continue;
    if (!decimal_found && chr == '.') {
      decimal_found = true;  // no more decimals
      continue;
    }
    if (!exponent_found && (chr == 'e' || chr == 'E')) {
      if (ch_idx + 1 < bytes) chr = data[ch_idx + 1];
      if (chr == '-' || chr == '+') ++ch_idx;
      decimal_found  = true;  // no decimal allowed in exponent
      exponent_found = true;  // no more exponents
      continue;
    }
    return false;
  }
  return result;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
