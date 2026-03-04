/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/side_type.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Return the size in bytes of padding d_str to width characters using a fill character
 * with byte length of fill_char_size
 *
 * Pad does not perform truncation. That is, If `d_str.length() > width` then `d_str.size_bytes()`
 * is returned.
 *
 * @param d_str String to pad
 * @param width Number of characters for the padded string result
 * @param fill_char_size Size of the fill character in bytes
 * @return The number of bytes required for the pad
 */
__device__ size_type compute_padded_size(string_view d_str,
                                         size_type width,
                                         size_type fill_char_size)
{
  auto const length = d_str.length();
  auto bytes        = d_str.size_bytes();
  if (width > length)                            // no truncating;
    bytes += fill_char_size * (width - length);  // add padding
  return bytes;
}

/**
 * @brief Pad d_str with fill_char into output up to width characters
 *
 * Pad does not perform truncation. That is, If `d_str.length() > width` then
 * then d_str is copied into output.
 *
 * @tparam side Specifies where fill_char is added to d_str
 * @param d_str String to pad
 * @param width Number of characters for the padded string result
 * @param fill_char Size of the fill character in bytes
 * @param output Device memory to copy the padded string into
 */
template <side_type side = side_type::RIGHT>
__device__ void pad_impl(cudf::string_view d_str,
                         cudf::size_type width,
                         cudf::char_utf8 fill_char,
                         char* output)
{
  auto length = d_str.length();
  if constexpr (side == side_type::LEFT) {
    while (length++ < width) {
      output += from_char_utf8(fill_char, output);
    }
    copy_string(output, d_str);
  }
  if constexpr (side == side_type::RIGHT) {
    output = copy_string(output, d_str);
    while (length++ < width) {
      output += from_char_utf8(fill_char, output);
    }
  }
  if constexpr (side == side_type::BOTH) {
    auto const pad_size = width - length;
    // an odd width will right-justify
    auto right_pad = (width % 2) ? pad_size / 2 : (pad_size - pad_size / 2);
    auto left_pad  = pad_size - right_pad;  // e.g. width=7: "++foxx+"; width=6: "+fox++"
    while (left_pad-- > 0) {
      output += from_char_utf8(fill_char, output);
    }
    output = copy_string(output, d_str);
    while (right_pad-- > 0) {
      output += from_char_utf8(fill_char, output);
    }
  }
}

/**
 * @brief Prepend d_str with '0' into output up to width characters
 *
 * Pad does not perform truncation. That is, If `d_str.length() > width` then
 * then d_str is copied into output.
 *
 * If d_str starts with a sign character ('-' or '+') then '0' padding
 * starts after the sign.
 *
 * @param d_str String to pad
 * @param width Number of characters for the padded string result
 * @param output Device memory to copy the padded string into
 */
__device__ void zfill_impl(cudf::string_view d_str, cudf::size_type width, char* output)
{
  auto length = d_str.length();
  auto in_ptr = d_str.data();
  // if the string starts with a sign, output the sign first
  if (!d_str.empty() && (*in_ptr == '-' || *in_ptr == '+')) {
    *output++ = *in_ptr++;
    d_str     = cudf::string_view{in_ptr, d_str.size_bytes() - 1};
  }
  while (length++ < width)
    *output++ = '0';  // prepend zero char
  copy_string(output, d_str);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
