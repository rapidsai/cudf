
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "udf_string.cuh"

#include <cudf/strings/detail/pad_impl.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Pad beginning and/or end of a string with the given fill character
 *
 * The side_type::BOTH will attempt to center the text using the `fill_char`.
 * If `width <= d_str.length()` no change occurs and the input `d_str` is returned.
 *
 * @tparam side Specify where the padding should occur
 * @param d_str String to pad
 * @param width Minimum length in characters of the output string
 * @param fill_char Character used for padding
 */
template <side_type side = side_type::RIGHT>
__device__ udf_string pad(cudf::string_view const d_str,
                          cudf::size_type width,
                          cudf::string_view fill_char = cudf::string_view{" ", 1})
{
  if (fill_char.empty()) { return udf_string{d_str}; }

  udf_string result;
  result.resize(cudf::strings::detail::compute_padded_size(d_str, width, fill_char.size_bytes()));
  cudf::strings::detail::pad_impl<side>(d_str, width, *fill_char.begin(), result.data());
  return result;
}

/**
 * @brief Pad beginning of a string with zero '0'
 *
 * If the `width` is smaller than the length of `d_str` no change occurs.
 *
 * If `d_str` starts with a sign character ('-' or '+') then '0' padding
 * starts after the sign.
 *
 * @param d_str String to fill
 * @param width Minimum length in characters of the output string (including the sign character)
 */
__device__ udf_string zfill(cudf::string_view const d_str, cudf::size_type width)
{
  udf_string result;
  result.resize(cudf::strings::detail::compute_padded_size(d_str, width, 1));
  cudf::strings::detail::zfill_impl(d_str, width, result.data());
  return result;
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
