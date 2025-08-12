/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/fixed_point/detail/floating_conversion.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup fixed_point_classes
 * @{
 * @file
 * @brief Conversion functions for fixed-point numbers
 */

/**
 * @brief Convert a floating-point value to fixed point
 *
 * @note This conversion was moved from fixed-point member functions to free functions.
 * This is so that the complex conversion code is not included into many parts of the
 * code base that don't need it, and so that it's more obvious to pinpoint where these
 * conversions are occurring.
 *
 * @tparam Fixed The fixed-point type to convert to
 * @tparam Floating The floating-point type to convert from
 * @param floating The floating-point value to convert
 * @param scale The desired scale of the fixed-point value
 * @return The converted fixed-point value
 */
template <typename Fixed,
          typename Floating,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<Floating>&& is_fixed_point<Fixed>())>
CUDF_HOST_DEVICE Fixed convert_floating_to_fixed(Floating floating, numeric::scale_type scale)
{
  using Rep        = typename Fixed::rep;
  auto const value = [&]() {
    if constexpr (Fixed::rad == numeric::Radix::BASE_10) {
      return numeric::detail::convert_floating_to_integral<Rep>(floating, scale);
    } else {
      return static_cast<Rep>(numeric::detail::shift<Rep, Fixed::rad>(floating, scale));
    }
  }();

  return Fixed(numeric::scaled_integer<Rep>{value, scale});
}

/**
 * @brief Convert a fixed-point value to floating point
 *
 * @note This conversion was moved from fixed-point member functions to free functions.
 * This is so that the complex conversion code is not included into many parts of the
 * code base that don't need it, and so that it's more obvious to pinpoint where these
 * conversions are occurring.
 *
 * @tparam Floating The floating-point type to convert to
 * @tparam Fixed The fixed-point type to convert from
 * @param fixed The fixed-point value to convert
 * @return The converted floating-point value
 */
template <typename Floating,
          typename Fixed,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<Floating>&& is_fixed_point<Fixed>())>
CUDF_HOST_DEVICE Floating convert_fixed_to_floating(Fixed fixed)
{
  using Rep = typename Fixed::rep;
  if constexpr (Fixed::rad == numeric::Radix::BASE_10) {
    return numeric::detail::convert_integral_to_floating<Floating>(fixed.value(), fixed.scale());
  } else {
    auto const casted = static_cast<Floating>(fixed.value());
    auto const scale  = numeric::scale_type{-fixed.scale()};
    return numeric::detail::shift<Rep, Fixed::rad>(casted, scale);
  }
}

/**
 * @brief Convert a value to floating point
 *
 * @tparam Floating The floating-point type to convert to
 * @tparam Input The input type to convert from
 * @param input The input value to convert
 * @return The converted floating-point value
 */
template <typename Floating,
          typename Input,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<Floating>)>
CUDF_HOST_DEVICE Floating convert_to_floating(Input input)
{
  if constexpr (is_fixed_point<Input>()) {
    return convert_fixed_to_floating<Floating>(input);
  } else {
    return static_cast<Floating>(input);
  }
}

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
