/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace jit {

enum errc : int {
  SUCCESS              = 0,
  ARITHMETIC_OVERFLOW  = 1,
  ARITHMETIC_UNDERFLOW = 2,
  DIVISION_BY_ZERO     = 3
};

namespace operators {

template <typename T>
using optional = cuda::std::optional<T>;

template <typename T>
struct promoted_t;

template <>
struct promoted_t<int8_t> {
  using type = int16_t;
};

template <>
struct promoted_t<uint8_t> {
  using type = uint16_t;
};

template <>
struct promoted_t<int16_t> {
  using type = int32_t;
};

template <>
struct promoted_t<uint16_t> {
  using type = uint32_t;
};

template <>
struct promoted_t<int32_t> {
  using type = int64_t;
};

template <>
struct promoted_t<uint32_t> {
  using type = uint64_t;
};

template <>
struct promoted_t<int64_t> {
  using type = __int128;
};

template <>
struct promoted_t<uint64_t> {
  using type = unsigned __int128;
};

template <typename T>
using promoted = typename promoted_t<T>::type;

template <typename T>
__device__ inline errc abs(T* out, T const* a)
{
  *out = (*a < 0) ? -*a : *a;
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc abs(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    abs(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc add(T* out, T const* a, T const* b)
{
  *out = (*a + *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    add(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arccos(T* out, T const* a);

template <>
__device__ inline errc arccos<float>(float* out, float const* a)
{
  *out = ::acosf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc arccos<double>(double* out, double const* a)
{
  *out = ::acos(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arccos(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arccos(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arccosh(T* out, T const* a);

template <>
__device__ inline errc arccosh<float>(float* out, float const* a)
{
  *out = ::acoshf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc arccosh<double>(double* out, double const* a)
{
  *out = ::acosh(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arccosh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arccosh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arcsin(T* out, T const* a);

template <>
__device__ inline errc arcsin<float>(float* out, float const* a)
{
  *out = ::asinf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc arcsin<double>(double* out, double const* a)
{
  *out = ::asin(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arcsin(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arcsin(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arcsinh(T* out, T const* a);

template <>
__device__ inline errc arcsinh<float>(float* out, float const* a)
{
  *out = ::asinhf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc arcsinh<double>(double* out, double const* a)
{
  *out = ::asinh(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arcsinh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arcsinh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arctan(T* out, T const* a);

template <>
__device__ inline errc arctan<float>(float* out, float const* a)
{
  *out = ::atanf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc arctan<double>(double* out, double const* a)
{
  *out = ::atan(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arctan(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arctan(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arctanh(T* out, T const* a);

template <>
__device__ inline errc arctanh<float>(float* out, float const* a)
{
  *out = ::atanhf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc arctanh<double>(double* out, double const* a)
{
  *out = ::atanh(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc arctanh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arctanh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_and(T* out, T const* a, T const* b)
{
  *out = (*a & *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    bit_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_invert(T* out, T const* a)
{
  *out = ~(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_invert(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    bit_invert(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_or(T* out, T const* a, T const* b)
{
  *out = (*a | *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    bit_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_xor(T* out, T const* a, T const* b)
{
  *out = (*a ^ *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc bit_xor(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    bit_xor(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cast_to_i32(int32_t* out, T const* a)
{
  *out = static_cast<int32_t>(*a);
  return errc::SUCCESS;
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
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cast_to_i64(int64_t* out, T const* a)
{
  *out = static_cast<int64_t>(*a);
  return errc::SUCCESS;
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
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cast_to_u32(uint32_t* out, T const* a)
{
  *out = static_cast<uint32_t>(*a);
  return errc::SUCCESS;
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
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cast_to_u64(uint64_t* out, T const* a)
{
  *out = static_cast<uint64_t>(*a);
  return errc::SUCCESS;
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
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cast_to_f32(float* out, T const* a)
{
  *out = static_cast<float>(*a);
  return errc::SUCCESS;
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
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cast_to_f64(double* out, T const* a)
{
  *out = static_cast<double>(*a);
  return errc::SUCCESS;
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
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cbrt(T* out, T const* a);

template <>
__device__ inline errc cbrt<float>(float* out, float const* a)
{
  *out = ::cbrtf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc cbrt<double>(double* out, double const* a)
{
  *out = ::cbrt(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cbrt(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cbrt(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ceil(T* out, T const* a);

template <>
__device__ inline errc ceil<float>(float* out, float const* a)
{
  *out = ::ceilf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc ceil<double>(double* out, double const* a)
{
  *out = ::ceil(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ceil(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    ceil(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cos(T* out, T const* a);

template <>
__device__ inline errc cos<float>(float* out, float const* a)
{
  *out = ::cosf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc cos<double>(double* out, double const* a)
{
  *out = ::cos(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cos(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cos(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cosh(T* out, T const* a);

template <>
__device__ inline errc cosh<float>(float* out, float const* a)
{
  *out = ::coshf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc cosh<double>(double* out, double const* a)
{
  *out = ::cosh(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc cosh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cosh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc div(T* out, T const* a, T const* b)
{
  *out = (*a / *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    bool r;
    equal(&r, &a->value(), &b->value());
    *out = r;
  } else if (a->is_null() && b->is_null()) {
    *out = true;
  } else {
    *out = false;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc exp(T* out, T const* a);

template <>
__device__ inline errc exp<float>(float* out, float const* a)
{
  *out = ::expf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc exp<double>(double* out, double const* a)
{
  *out = ::exp(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc exp(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    exp(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc floor(T* out, T const* a);

template <>
__device__ inline errc floor<float>(float* out, float const* a)
{
  *out = ::floorf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc floor<double>(double* out, double const* a)
{
  *out = ::floor(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc floor(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    floor(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc greater(bool* out, T const* a, T const* b)
{
  *out = (*a > *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc greater(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    bool r;
    greater(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc greater_equal(bool* out, T const* a, T const* b)
{
  *out = (*a >= *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc greater_equal(optional<bool>* out,
                                     optional<T> const* a,
                                     optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    bool r;
    greater_equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc identity(T* out, T const* a)
{
  *out = *a;
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc identity(optional<T>* out, optional<T> const* a)
{
  *out = *a;
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc is_null(bool* out, T const* a)
{
  *out = false;
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc is_null(optional<bool>* out, optional<T> const* a)
{
  *out = a->is_null();
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc less(bool* out, T const* a, T const* b)
{
  *out = (*a < *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc less(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    bool r;
    less(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc less_equal(bool* out, T const* a, T const* b)
{
  *out = (*a <= *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc less_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    bool r;
    less_equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc log(T* out, T const* a);

template <>
__device__ inline errc log<float>(float* out, float const* a)
{
  *out = ::logf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc log<double>(double* out, double const* a)
{
  *out = ::log(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc log(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    log(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc logical_and(T* out, T const* a, T const* b)
{
  *out = (*a && *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc logical_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc logical_or(T* out, T const* a, T const* b)
{
  *out = (*a || *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc logical_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc mod(T* out, T const* a, T const* b)
{
  *out = (*a % *b);
  return errc::SUCCESS;
}

template <>
__device__ inline errc mod<float>(float* out, float const* a, float const* b)
{
  *out = ::fmodf(*a, *b);
  return errc::SUCCESS;
}

template <>
__device__ inline errc mod<double>(double* out, double const* a, double const* b)
{
  *out = ::fmod(*a, *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    mod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc mul(T* out, T const* a, T const* b)
{
  *out = (*a * *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    mul(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc null_equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc null_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    *out = (*(*a) == *(*b));
  } else if (a->is_null() && b->is_null()) {
    *out = true;
  } else {
    *out = false;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc null_logical_and(T* out, T const* a, T const* b)
{
  *out = (*a && *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc null_logical_and(optional<T>* out,
                                        optional<T> const* a,
                                        optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    bool r;
    null_logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else if (a->is_null() && b->is_null()) {
    *out = nullopt;
  } else {
    if (a->is_valid() ? *(*a) : *(*b)) {
      *out = nullopt;
    } else {
      *out = false;
    }
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc null_logical_or(T* out, T const* a, T const* b)
{
  *out = (*a || *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc null_logical_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    bool r;
    null_logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else if (a->is_null() && b->is_null()) {
    *out = nullopt;
  } else {
    if (a->is_valid() ? *(*a) : *(*b)) {
      *out = true;
    } else {
      *out = nullopt;
    }
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc pow(T* out, T const* a, T const* b);

template <>
__device__ inline errc pow<float>(float* out, float const* a, float const* b)
{
  *out = ::powf(*a, *b);
  return errc::SUCCESS;
}

template <>
__device__ inline errc pow<double>(double* out, double const* a, double const* b)
{
  *out = ::pow(*a, *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc pow(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    pow(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc pymod(T* out, T const* a, T const* b)
{
  *out = (*a % *b + *b) % *b;
  return errc::SUCCESS;
}

template <>
__device__ inline errc pymod<float>(float* out, float const* a, float const* b)
{
  *out = ::fmodf(::fmodf(*a, *b) + *b, *b);
  return errc::SUCCESS;
}

template <>
__device__ inline errc pymod<double>(double* out, double const* a, double const* b)
{
  *out = ::fmod(::fmod(*a, *b) + *b, *b);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc pymod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    pymod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc rint(T* out, T const* a);

template <>
__device__ inline errc rint<float>(float* out, float const* a)
{
  *out = ::rintf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc rint<double>(double* out, double const* a)
{
  *out = ::rint(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc rint(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    rint(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc sin(T* out, T const* a);

template <>
__device__ inline errc sin<float>(float* out, float const* a)
{
  *out = ::sinf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc sin<double>(double* out, double const* a)
{
  *out = ::sin(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc sin(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    sin(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc sinh(T* out, T const* a);

template <>
__device__ inline errc sinh<float>(float* out, float const* a)
{
  *out = ::sinhf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc sinh<double>(double* out, double const* a)
{
  *out = ::sinh(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc sinh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    sinh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc sub(T* out, T const* a, T const* b)
{
  *out = *a - *b;
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    sub(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc tanh(T* out, T const* a);

template <>
__device__ inline errc tanh<float>(float* out, float const* a)
{
  *out = ::tanhf(*a);
  return errc::SUCCESS;
}

template <>
__device__ inline errc tanh<double>(double* out, double const* a)
{
  *out = ::tanh(*a);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc tanh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    tanh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc if_else(T* out,
                               bool const* condition,
                               T const* true_value,
                               T const* false_value)
{
  *out = *condition ? *true_value : *false_value;
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc if_else(optional<T>* out,
                               optional<bool> const* condition,
                               optional<T> const* true_value,
                               optional<T> const* false_value)
{
  if (condition->is_valid() && true_value->is_valid() && false_value->is_valid()) {
    if_else<T>(&out->value(), &condition->value(), &true_value->value(), &false_value->value());
  } else {
    *out = nullopt;
  }
  return errc::SUCCESS;
}

namespace detail {

template <typename T>
__device__ inline errc ansi_add_unsigned(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) + static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max())) { return errc::ARITHMETIC_OVERFLOW; }
  *out = static_cast<T>(r);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_add_signed(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) + static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::ARITHMETIC_OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_sub_unsigned(T* out, T const* a, T const* b)
{
  if (*a < *b) { return errc::ARITHMETIC_UNDERFLOW; }
  auto r = *a - *b;
  *out   = static_cast<T>(r);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_sub_signed(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) - static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::ARITHMETIC_UNDERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_mul_unsigned(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r = static_cast<P>(*a) * static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max())) { return errc::ARITHMETIC_OVERFLOW; }
  *out = static_cast<T>(r);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_mul_signed(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) * static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::ARITHMETIC_OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_div_unsigned(T* out, T const* a, T const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = static_cast<T>(static_cast<T>(*a) / static_cast<T>(*b));
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_div_signed(T* out, T const* a, T const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  if (*a == cuda::std::numeric_limits<T>::min() && *b == -1) { return errc::ARITHMETIC_OVERFLOW; }
  *out = static_cast<T>(static_cast<T>(*a) / static_cast<T>(*b));
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc ansi_div_float(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  auto r = static_cast<P>(*a) / static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::lowest())) {
    return errc::ARITHMETIC_OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::SUCCESS;
}

}  // namespace detail

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
{
  return detail::ansi_add_unsigned(out, a, b);
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
{
  return detail::ansi_add_signed(out, a, b);
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
{
  *out = *a + *b;
  return errc::SUCCESS;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
__device__ inline errc ansi_sub(T* out, T const* a, T const* b)
{
  return detail::ansi_sub_unsigned(out, a, b);
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_sub(T* out, T const* a, T const* b)
{
  return detail::ansi_sub_signed(out, a, b);
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_sub(T* out, T const* a, T const* b)
{
  *out = *a - *b;
  return errc::SUCCESS;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
__device__ inline errc ansi_mul(T* out, T const* a, T const* b)
{
  return detail::ansi_mul_unsigned(out, a, b);
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_mul(T* out, T const* a, T const* b)
{
  return detail::ansi_mul_signed(out, a, b);
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_mul(T* out, T const* a, T const* b)
{
  *out = *a * *b;
  return errc::SUCCESS;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
__device__ inline errc ansi_div(T* out, T const* a, T const* b)
{
  return detail::ansi_div_unsigned(out, a, b);
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_div(T* out, T const* a, T const* b)
{
  return detail::ansi_div_signed(out, a, b);
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_div(T* out, T const* a, T const* b)
{
  return detail::ansi_div_float(out, a, b);
}

template <typename T>
__device__ inline errc try_add(optional<T>* out, T const* a, T const* b)
{
  auto e = ansi_add(out, a, b);
  if (e != errc::SUCCESS) { *out = nullopt; }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc try_sub(optional<T>* out, T const* a, T const* b)
{
  auto e = ansi_sub(out, a, b);
  if (e != errc::SUCCESS) { *out = nullopt; }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc try_mul(optional<T>* out, T const* a, T const* b)
{
  auto e = ansi_mul(out, a, b);
  if (e != errc::SUCCESS) { *out = nullopt; }
  return errc::SUCCESS;
}

template <typename T>
__device__ inline errc try_div(optional<T>* out, T const* a, T const* b)
{
  auto e = ansi_div(out, a, b);
  if (e != errc::SUCCESS) { *out = nullopt; }
  return errc::SUCCESS;
}

// TODO: overloads for optional

}  // namespace operators
}  // namespace jit
}  // namespace CUDF_EXPORT cudf
