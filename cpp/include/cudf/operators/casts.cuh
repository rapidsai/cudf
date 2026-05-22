/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/conv.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Casts input values to bool.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_b8(bool* out, T const* a)
{
  *out = static_cast<bool>(*a);
}

/**
 * @brief Casts optional input values to optional bool.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_b8(cuda::std::optional<bool>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    bool r;
    cast_to_b8(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to int8_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_i8(int8_t* out, T const* a)
{
  *out = static_cast<int8_t>(*a);
}

/**
 * @brief Casts optional input values to optional int8_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_i8(cuda::std::optional<int8_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    int8_t r;
    cast_to_i8(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to int16_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_i16(int16_t* out, T const* a)
{
  *out = static_cast<int16_t>(*a);
}

/**
 * @brief Casts optional input values to optional int16_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_i16(cuda::std::optional<int16_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    int16_t r;
    cast_to_i16(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to int32_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_i32(int32_t* out, T const* a)
{
  *out = static_cast<int32_t>(*a);
}

/**
 * @brief Casts optional input values to optional int32_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_i32(cuda::std::optional<int32_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    int32_t r;
    cast_to_i32(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to int64_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_i64(int64_t* out, T const* a)
{
  *out = static_cast<int64_t>(*a);
}

/**
 * @brief Casts optional input values to optional int64_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_i64(cuda::std::optional<int64_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    int64_t r;
    cast_to_i64(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to uint8_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_u8(uint8_t* out, T const* a)
{
  *out = static_cast<uint8_t>(*a);
}

/**
 * @brief Casts optional input values to optional uint8_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_u8(cuda::std::optional<uint8_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    uint8_t r;
    cast_to_u8(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to uint16_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_u16(uint16_t* out, T const* a)
{
  *out = static_cast<uint16_t>(*a);
}

/**
 * @brief Casts optional input values to optional uint16_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_u16(cuda::std::optional<uint16_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    uint16_t r;
    cast_to_u16(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to uint32_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_u32(uint32_t* out, T const* a)
{
  *out = static_cast<uint32_t>(*a);
}

/**
 * @brief Casts optional input values to optional uint32_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_u32(cuda::std::optional<uint32_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    uint32_t r;
    cast_to_u32(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to uint64_t.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam T Source type.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_u64(uint64_t* out, T const* a)
{
  *out = static_cast<uint64_t>(*a);
}

/**
 * @brief Casts optional input values to optional uint64_t.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_u64(cuda::std::optional<uint64_t>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    uint64_t r;
    cast_to_u64(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to float.
 *
 * Overloads support integral, floating-point, fixed-point decimal, and optional inputs.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_f32(float* out, T const* a)
  requires(cuda::std::is_integral_v<T> || cuda::std::is_floating_point_v<T>)
{
  *out = static_cast<float>(*a);
}

/**
 * @brief Casts fixed-point decimal values to float.
 *
 * @tparam R Source decimal representation type.
 * @param out Destination cast value.
 * @param a Source decimal value.
 */
template <typename R>
__device__ void cast_to_f32(float* out, numeric::decimal<R> const* a)
{
  *out = convert_fixed_to_floating<float>(*a);
}

/**
 * @brief Casts optional input values to optional float.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_f32(cuda::std::optional<float>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    float r;
    cast_to_f32(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts input values to double.
 *
 * Overloads support integral, floating-point, fixed-point decimal, and optional inputs.
 * @param out Destination cast value.
 * @param a Input value.
 */
template <typename T>
__device__ void cast_to_f64(double* out, T const* a)
  requires(cuda::std::is_integral_v<T> || cuda::std::is_floating_point_v<T>)
{
  *out = static_cast<double>(*a);
}

/**
 * @brief Casts fixed-point decimal values to double.
 *
 * @tparam R Source decimal representation type.
 * @param out Destination cast value.
 * @param a Source decimal value.
 */
template <typename R>
__device__ void cast_to_f64(double* out, numeric::decimal<R> const* a)
{
  *out = convert_fixed_to_floating<double>(*a);
}

/**
 * @brief Casts optional input values to optional double.
 *
 * @tparam T Source type.
 * @param out Destination optional cast value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cast_to_f64(cuda::std::optional<double>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    double r;
    cast_to_f64(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

namespace detail {

/**
 * @brief Casts one fixed-point decimal representation to another.
 *
 * @tparam To Destination representation type.
 * @tparam From Source representation type.
 * @param out Destination decimal value.
 * @param a Source decimal value.
 */
template <typename To, typename From>
__device__ void decimal_cast(numeric::decimal<To>* out, numeric::decimal<From> const* a)
{
  auto rep = static_cast<To>(a->value());
  *out     = numeric::decimal<To>{numeric::scaled_integer<To>{rep, a->scale()}};
}

}  // namespace detail

// TODO(lamarrr): CAST_TO_DEC32 for int & float

/**
 * @brief Casts decimal input values to decimal32.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam R Source decimal representation type.
 * @param out Destination decimal32 value.
 * @param a Source decimal value.
 */
template <typename R>
__device__ void cast_to_dec32(numeric::decimal32* out, numeric::decimal<R> const* a)
{
  return detail::decimal_cast(out, a);
}

/**
 * @brief Casts optional decimal input values to optional decimal32.
 *
 * @tparam R Source decimal representation type.
 * @param out Destination optional decimal32 value.
 * @param a Optional decimal input value.
 */
template <typename R>
__device__ void cast_to_dec32(cuda::std::optional<numeric::decimal32>* out,
                              cuda::std::optional<numeric::decimal<R>> const* a)
{
  if (a->has_value()) {
    numeric::decimal32 r;
    cast_to_dec32(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts decimal input values to decimal64.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam R Source decimal representation type.
 * @param out Destination decimal64 value.
 * @param a Source decimal value.
 */
template <typename R>
__device__ void cast_to_dec64(numeric::decimal64* out, numeric::decimal<R> const* a)
{
  return detail::decimal_cast(out, a);
}

/**
 * @brief Casts optional decimal input values to optional decimal64.
 *
 * @tparam R Source decimal representation type.
 * @param out Destination optional decimal64 value.
 * @param a Optional decimal input value.
 */
template <typename R>
__device__ void cast_to_dec64(cuda::std::optional<numeric::decimal64>* out,
                              cuda::std::optional<numeric::decimal<R>> const* a)
{
  if (a->has_value()) {
    numeric::decimal64 r;
    cast_to_dec64(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Casts decimal input values to decimal128.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam R Source decimal representation type.
 * @param out Destination decimal128 value.
 * @param a Source decimal value.
 */
template <typename R>
__device__ void cast_to_dec128(numeric::decimal128* out, numeric::decimal<R> const* a)
{
  return detail::decimal_cast(out, a);
}

/**
 * @brief Casts optional decimal input values to optional decimal128.
 *
 * @tparam R Source decimal representation type.
 * @param out Destination optional decimal128 value.
 * @param a Optional decimal input value.
 */
template <typename R>
__device__ void cast_to_dec128(cuda::std::optional<numeric::decimal128>* out,
                               cuda::std::optional<numeric::decimal<R>> const* a)
{
  if (a->has_value()) {
    numeric::decimal128 r;
    cast_to_dec128(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Rescales fixed-point decimal values to a target scale.
 *
 * Scalar and optional overloads are provided; optional overloads propagate nulls.
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Source decimal value.
 * @param new_scale Target decimal scale.
 */
template <typename R>
__device__ void rescale(numeric::decimal<R>* out,
                        numeric::decimal<R> const* a,
                        int32_t const* new_scale)
{
  *out = a->rescaled(numeric::scale_type{*new_scale});
}

/**
 * @brief Rescales optional fixed-point decimal input values.
 *
 * @tparam R Decimal representation type.
 * @param out Destination optional decimal value.
 * @param a Optional source decimal value.
 * @param new_scale Optional target decimal scale.
 */
template <typename R>
__device__ void rescale(cuda::std::optional<numeric::decimal<R>>* out,
                        cuda::std::optional<numeric::decimal<R>> const* a,
                        cuda::std::optional<int32_t> const* new_scale)
{
  if (a->has_value() && new_scale->has_value()) {
    numeric::decimal<R> r;
    rescale(&r, &a->value(), &new_scale->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
