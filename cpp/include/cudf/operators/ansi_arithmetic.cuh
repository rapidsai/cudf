/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {
namespace detail {

template <typename T>
__device__ inline errc ansi_add_unsigned(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) + static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max())) { return errc::OVERFLOW; }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_add_signed(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) + static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_sub_unsigned(T* out, T const* a, T const* b)
{
  if (*a < *b) { return errc::OVERFLOW; }
  auto r = *a - *b;
  *out   = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_sub_signed(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) - static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_mul_unsigned(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) * static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max())) { return errc::OVERFLOW; }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_mul_signed(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  auto r  = static_cast<P>(*a) * static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_div_unsigned(T* out, T const* a, T const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = static_cast<T>(static_cast<T>(*a) / static_cast<T>(*b));
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_div_signed(T* out, T const* a, T const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  if (*a == cuda::std::numeric_limits<T>::min() && *b == -1) { return errc::OVERFLOW; }
  *out = static_cast<T>(static_cast<T>(*a) / static_cast<T>(*b));
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_div_float(T* out, T const* a, T const* b)
{
  using P = promoted<T>;
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  auto r = static_cast<P>(*a) / static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::lowest())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
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
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_add(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* a,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());
  auto sum   = a->rescaled(scale).value() + b->rescaled(scale).value();

  if (numeric::addition_overflow(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = numeric::fixed_point<R, numeric::Radix::BASE_10>{numeric::scaled_integer<R>{sum, scale}};
  return errc::OK;
}

template <typename R, typename Ratio>
__device__ inline errc ansi_add(cuda::std::chrono::duration<R, Ratio>* out,
                                cuda::std::chrono::duration<R, Ratio> const* a,
                                cuda::std::chrono::duration<R, Ratio> const* b)
{
  using P = promoted<R>;
  auto r  = static_cast<P>(a->count()) + static_cast<P>(b->count());
  if (r > static_cast<P>(cuda::std::numeric_limits<R>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<R>::min())) {
    return errc::OVERFLOW;
  }
  *out = cuda::std::chrono::duration<R, Ratio>{static_cast<R>(r)};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_add(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }

  return errc::OK;
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
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_sub(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* a,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());
  auto sum   = a->rescaled(scale).value() - b->rescaled(scale).value();

  if (numeric::subtraction_overflow(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = numeric::fixed_point<R, numeric::Radix::BASE_10>{numeric::scaled_integer<R>{sum, scale}};
  return errc::OK;
}

template <typename R, typename Ratio>
__device__ inline errc ansi_sub(cuda::std::chrono::duration<R, Ratio>* out,
                                cuda::std::chrono::duration<R, Ratio> const* a,
                                cuda::std::chrono::duration<R, Ratio> const* b)
{
  using P = promoted<R>;
  auto r  = static_cast<P>(a->count()) - static_cast<P>(b->count());
  if (r > static_cast<P>(cuda::std::numeric_limits<R>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<R>::min())) {
    return errc::OVERFLOW;
  }
  *out = cuda::std::chrono::duration<R, Ratio>{static_cast<R>(r)};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_sub(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
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
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_mul(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* a,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* b)
{
  if (numeric::multiplication_overflow(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = numeric::fixed_point<R, numeric::Radix::BASE_10>{
    numeric::scaled_integer<R>{a->value() * b->value(), a->scale() + b->scale()}};
  return errc::OK;
}

template <typename R, typename Ratio>
__device__ inline errc ansi_mul(cuda::std::chrono::duration<R, Ratio>* out,
                                cuda::std::chrono::duration<R, Ratio> const* a,
                                cuda::std::chrono::duration<R, Ratio> const* b)
{
  using P = promoted<R>;
  auto r  = static_cast<P>(a->count()) * static_cast<P>(b->count());
  if (r > static_cast<P>(cuda::std::numeric_limits<R>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<R>::min())) {
    return errc::OVERFLOW;
  }
  *out = cuda::std::chrono::duration<R, Ratio>{static_cast<R>(r)};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_mul(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
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

template <typename R>
__device__ inline errc ansi_div(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* a,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* b)
{
  if (numeric::division_overflow(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = numeric::fixed_point<R, numeric::Radix::BASE_10>{
    numeric::scaled_integer<R>{a->value() / b->value(), a->scale() - b->scale()}};
  return errc::OK;
}

template <typename R, typename Ratio>
__device__ inline errc ansi_div(cuda::std::chrono::duration<R, Ratio>* out,
                                cuda::std::chrono::duration<R, Ratio> const* a,
                                cuda::std::chrono::duration<R, Ratio> const* b)
{
  if (b->count() == 0) { return errc::DIVISION_BY_ZERO; }
  using P = promoted<R>;
  auto r  = static_cast<P>(a->count()) / static_cast<P>(b->count());
  if (r > static_cast<P>(cuda::std::numeric_limits<R>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<R>::min())) {
    return errc::OVERFLOW;
  }
  *out = cuda::std::chrono::duration<R, Ratio>{static_cast<R>(r)};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_div(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_abs(T* out, T const* a)
{
  if (*a == cuda::std::numeric_limits<T>::min()) { return errc::OVERFLOW; }
  *out = (*a < 0) ? -(*a) : *a;
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
__device__ inline errc ansi_abs(T* out, T const* a)
{
  *out = *a;
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_abs(T* out, T const* a)
{
  *out = (*a < 0) ? -(*a) : *a;
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_abs(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  out->value() = (a->value() < 0) ? -a->value() : a->value();
  return errc::OK;
}

template <typename R, typename Ratio>
__device__ inline errc ansi_abs(cuda::std::chrono::duration<R, Ratio>* out,
                                cuda::std::chrono::duration<R, Ratio> const* a)
{
  if (a->count() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  *out = (*a < cuda::std::chrono::duration<R, Ratio>{0}) ? -(*a) : *a;
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_abs(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    if (errc e = ansi_abs(&r, &a->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_signed_v<T>)
__device__ inline errc ansi_neg(T* out, T const* a)
{
  if (*a == cuda::std::numeric_limits<T>::min()) { return errc::OVERFLOW; }
  *out = -(*a);
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_neg(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                                numeric::fixed_point<R, numeric::Radix::BASE_10> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  auto rep = -a->value();
  *out =
    numeric::fixed_point<R, numeric::Radix::BASE_10>{numeric::scaled_integer<R>{rep, a->scale()}};
  return errc::OK;
}

template <typename R, typename Ratio>
__device__ inline errc ansi_neg(cuda::std::chrono::duration<R, Ratio>* out,
                                cuda::std::chrono::duration<R, Ratio> const* a)
{
  if (a->count() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  auto rep = -a->count();
  *out     = cuda::std::chrono::duration<R, Ratio>{rep};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_neg(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    if (errc e = ansi_neg(&r, &a->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_try_add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_add(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_try_sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_sub(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_try_mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_mul(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_try_div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    if (errc e = ansi_div(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_try_abs(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    if (errc e = ansi_abs(&r, a); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_try_neg(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    if (errc e = ansi_neg(&r, &a->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

// TODO: IMPLEMENT
// exponent should not exceed 38 thus 10^exponent should fit in __int128_t

// TODO: check if a's precision is less than or equal to the provided precision, if not return
// errc::OVERFLOW
template <typename R>
__device__ inline errc assert_precise(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                                      numeric::fixed_point<R, numeric::Radix::BASE_10> const* a,
                                      int32_t precision)
{
}

template <typename T>
__device__ inline errc assert_precise(optional<T>* out, optional<T> const* a, int32_t precision)
{
  if (a->is_valid()) {
    return assert_precise(&out->value(), &a->value(), precision);
  } else {
    *out = nullopt;
    return errc::OK;
  }
}

template <typename R>
__device__ inline errc try_precise(
  optional<numeric::fixed_point<R, numeric::Radix::BASE_10>>* out,
  optional<numeric::fixed_point<R, numeric::Radix::BASE_10>> const* a,
  int32_t precision)
{
  if (a->is_valid()) {
    if (errc e = assert_precise(&out->value(), &a->value(), precision); e != errc::OK) {
      *out = nullopt;
      return errc::OK;
    } else {
      *out = a->value();
      return errc::OK;
    }
  } else {
    *out = nullopt;
    return errc::OK;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
