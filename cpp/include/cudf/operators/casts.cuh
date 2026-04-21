/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {

template <typename T>
__device__ inline errc cast_to_i32(int32_t* out, T const* a)
{
  *out = static_cast<int32_t>(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_i32(optional<int32_t>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    int32_t r;
    cast_to_i32(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_i64(int64_t* out, T const* a)
{
  *out = static_cast<int64_t>(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_i64(optional<int64_t>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    int64_t r;
    cast_to_i64(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_u32(uint32_t* out, T const* a)
{
  *out = static_cast<uint32_t>(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_u32(optional<uint32_t>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    uint32_t r;
    cast_to_u32(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_u64(uint64_t* out, T const* a)
{
  *out = static_cast<uint64_t>(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_u64(optional<uint64_t>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    uint64_t r;
    cast_to_u64(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_f32(float* out, T const* a)
{
  *out = static_cast<float>(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_f32(optional<float>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    float r;
    cast_to_f32(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_f64(double* out, T const* a)
{
  *out = static_cast<double>(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_f64(optional<double>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    double r;
    cast_to_f64(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

namespace detail {

template <typename T, typename U, numeric::Radix Radix>
__device__ inline errc fixed_point_cast(numeric::fixed_point<T, Radix>* out,
                                        numeric::fixed_point<U, Radix> const* a)
{
  auto rep = static_cast<T>(a->value());
  *out     = numeric::fixed_point<T, Radix>{numeric::scaled_integer<T>{rep, a->scale()}};
  return errc::OK;
}

}  // namespace detail

template <typename T>
__device__ inline errc cast_to_dec32(numeric::decimal32* out,
                                     numeric::fixed_point<T, numeric::Radix::BASE_10> const* a)
{
  return detail::fixed_point_cast(out, a);
}

template <typename T>
__device__ inline errc cast_to_dec32(
  optional<numeric::decimal32>* out,
  optional<numeric::fixed_point<T, numeric::Radix::BASE_10>> const* a)
{
  if (a->is_valid()) {
    numeric::decimal32 r;
    cast_to_dec32(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_dec64(numeric::decimal64* out,
                                     numeric::fixed_point<T, numeric::Radix::BASE_10> const* a)
{
  return detail::fixed_point_cast(out, a);
}

template <typename T>
__device__ inline errc cast_to_dec64(
  optional<numeric::decimal64>* out,
  optional<numeric::fixed_point<T, numeric::Radix::BASE_10>> const* a)
{
  if (a->is_valid()) {
    numeric::decimal64 r;
    cast_to_dec64(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cast_to_dec128(numeric::decimal128* out,
                                      numeric::fixed_point<T, numeric::Radix::BASE_10> const* a)
{
  return detail::fixed_point_cast(out, a);
}

template <typename T>
__device__ inline errc cast_to_dec128(
  optional<numeric::decimal128>* out,
  optional<numeric::fixed_point<T, numeric::Radix::BASE_10>> const* a)
{
  if (a->is_valid()) {
    numeric::decimal128 r;
    cast_to_dec128(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
