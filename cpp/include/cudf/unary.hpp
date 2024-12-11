/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup transformation_unaryops
 * @{
 * @file
 * @brief Column APIs for unary ops
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

/**
 * @brief Types of unary operations that can be performed on data.
 */
enum class unary_operator : int32_t {
  SIN,         ///< Trigonometric sine
  COS,         ///< Trigonometric cosine
  TAN,         ///< Trigonometric tangent
  ARCSIN,      ///< Trigonometric sine inverse
  ARCCOS,      ///< Trigonometric cosine inverse
  ARCTAN,      ///< Trigonometric tangent inverse
  SINH,        ///< Hyperbolic sine
  COSH,        ///< Hyperbolic cosine
  TANH,        ///< Hyperbolic tangent
  ARCSINH,     ///< Hyperbolic sine inverse
  ARCCOSH,     ///< Hyperbolic cosine inverse
  ARCTANH,     ///< Hyperbolic tangent inverse
  EXP,         ///< Exponential (base e, Euler number)
  LOG,         ///< Natural Logarithm (base e)
  SQRT,        ///< Square-root (x^0.5)
  CBRT,        ///< Cube-root (x^(1.0/3))
  CEIL,        ///< Smallest integer value not less than arg
  FLOOR,       ///< largest integer value not greater than arg
  ABS,         ///< Absolute value
  RINT,        ///< Rounds the floating-point argument arg to an integer value
  BIT_INVERT,  ///< Bitwise Not (~)
  NOT,         ///< Logical Not (!)
};

/**
 * @brief Performs unary op on all values in column
 *
 * Note: For `decimal32` and `decimal64`, only `ABS`, `CEIL` and `FLOOR` are supported.
 *
 * @param input A `column_view` as input
 * @param op operation to perform
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns Column of same size as `input` containing result of the operation
 */
std::unique_ptr<cudf::column> unary_operation(
  cudf::column_view const& input,
  cudf::unary_operator op,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a column of `type_id::BOOL8` elements where for every element in `input` `true`
 * indicates the value is null and `false` indicates the value is valid.
 *
 * @param input A `column_view` as input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `true`
 * representing `null` values.
 */
std::unique_ptr<cudf::column> is_null(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a column of `type_id::BOOL8` elements where for every element in `input` `true`
 * indicates the value is valid and `false` indicates the value is null.
 *
 * @param input A `column_view` as input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `false`
 * representing `null` values.
 */
std::unique_ptr<cudf::column> is_valid(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Casts data from dtype specified in input to dtype specified in output.
 *
 * Supports only fixed-width types.
 *
 * @param input Input column
 * @param out_type Desired datatype of output column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns Column of same size as `input` containing result of the cast operation
 * @throw cudf::logic_error if `out_type` is not a fixed-width type
 */
std::unique_ptr<column> cast(
  column_view const& input,
  data_type out_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Check if a cast between two datatypes is supported.
 *
 * @param from source type
 * @param to   target type
 *
 * @returns true if the cast is supported.
 */
bool is_supported_cast(data_type from, data_type to) noexcept;

/**
 * @brief Creates a column of `type_id::BOOL8` elements indicating the presence of `NaN` values
 * in a column of floating point values.
 * The output element at row `i` is `true` if the element in `input` at row i is `NAN`, else `false`
 *
 * @throws cudf::logic_error if `input` is a non-floating point type
 *
 * @param input A column of floating-point elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `true` representing `NAN` values
 */
std::unique_ptr<column> is_nan(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a column of `type_id::BOOL8` elements indicating the absence of `NaN` values
 * in a column of floating point values.
 * The output element at row `i` is `false` if the element in `input` at row i is `NAN`, else `true`
 *
 * @throws cudf::logic_error if `input` is a non-floating point type
 *
 * @param input A column of floating-point elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `false` representing `NAN`
 * values
 */
std::unique_ptr<column> is_not_nan(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
