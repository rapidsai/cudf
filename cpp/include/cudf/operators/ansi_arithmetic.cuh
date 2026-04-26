/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/math.cuh>
#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {
namespace detail {

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
using promote = typename promoted_t<T>::type;

}  // namespace detail

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
{
  using P = detail::promote<T>;
  auto r  = static_cast<P>(*a) + static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max())) { return errc::OVERFLOW; }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
{
  using P = detail::promote<T>;
  auto r  = static_cast<P>(*a) + static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
{
  *out = *a + *b;
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_add(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());
  auto sum   = a->rescaled(scale).value() + b->rescaled(scale).value();

  if (numeric::addition_overflow(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = decimal<R>{numeric::scaled_integer<R>{sum, scale}};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
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
  if (*a < *b) { return errc::OVERFLOW; }
  auto r = *a - *b;
  *out   = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_sub(T* out, T const* a, T const* b)
{
  using P = detail::promote<T>;
  auto r  = static_cast<P>(*a) - static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_sub(T* out, T const* a, T const* b)
{
  *out = *a - *b;
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_sub(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());
  auto sum   = a->rescaled(scale).value() - b->rescaled(scale).value();

  if (numeric::subtraction_overflow(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = decimal<R>{numeric::scaled_integer<R>{sum, scale}};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
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
  using P = detail::promote<T>;
  auto r  = static_cast<P>(*a) * static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max())) { return errc::OVERFLOW; }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_mul(T* out, T const* a, T const* b)
{
  using P = detail::promote<T>;
  auto r  = static_cast<P>(*a) * static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::min())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_mul(T* out, T const* a, T const* b)
{
  *out = *a * *b;
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_mul(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  if (numeric::multiplication_overflow(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = decimal<R>{numeric::scaled_integer<R>{a->value() * b->value(), a->scale() + b->scale()}};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
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
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = static_cast<T>(static_cast<T>(*a) / static_cast<T>(*b));
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
__device__ inline errc ansi_div(T* out, T const* a, T const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  if (*a == cuda::std::numeric_limits<T>::min() && *b == -1) { return errc::OVERFLOW; }
  *out = static_cast<T>(static_cast<T>(*a) / static_cast<T>(*b));
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
__device__ inline errc ansi_div(T* out, T const* a, T const* b)
{
  using P = detail::promote<T>;
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  auto r = static_cast<P>(*a) / static_cast<P>(*b);
  if (r > static_cast<P>(cuda::std::numeric_limits<T>::max()) ||
      r < static_cast<P>(cuda::std::numeric_limits<T>::lowest())) {
    return errc::OVERFLOW;
  }
  *out = static_cast<T>(r);
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_div(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  if (numeric::division_overflow(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = decimal<R>{numeric::scaled_integer<R>{a->value() / b->value(), a->scale() - b->scale()}};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
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
__device__ inline errc ansi_mod(T* out, T const* a, T const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  T r = *a % *b;
  if (r != 0 && ((r > 0) != (*b > 0))) { r += *b; }
  *out = r;
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
__device__ inline errc ansi_mod(T* out, T const* a, T const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = *a % *b;
  return errc::OK;
}

__device__ inline errc ansi_mod(float* out, float const* a, float const* b)
{
  *out = (*a) - (*b) * ::floorf((*a) / (*b));
  return errc::OK;
}

__device__ inline errc ansi_mod(double* out, double const* a, double const* b)
{
  *out = (*a) - (*b) * ::floor((*a) / (*b));
  return errc::OK;
}

template <typename R>
__device__ inline errc ansi_mod(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  if (b->value() == 0) { return errc::DIVISION_BY_ZERO; }

  decimal<R> div;

  if (errc e = ansi_div(&div, a, b); e != errc::OK) { return e; }

  decimal<R> quotient;
  floor(&quotient, &div);
  *out = *a - *b * quotient;
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mod(&r, &a->value(), &b->value()); e != errc::OK) {
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
__device__ inline errc ansi_abs(decimal<R>* out, decimal<R> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  out->value() = (a->value() < 0) ? -a->value() : a->value();
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_abs(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
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
__device__ inline errc ansi_neg(decimal<R>* out, decimal<R> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  auto rep = -a->value();
  *out     = decimal<R>{numeric::scaled_integer<R>{rep, a->scale()}};
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_neg(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
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

template <typename R>
__device__ inline errc ansi_precision_cast(decimal<R>* out,
                                           decimal<R> const* a,
                                           int32_t const* precision)
{
  auto current_scale = static_cast<int32_t>(a->scale());
  auto allowed_scale = -(*precision);

  if (current_scale >= allowed_scale) {
    *out = *a;
    return errc::OK;
  }

  auto extra_digits = allowed_scale - current_scale;

  auto factor = detail::ipow10(static_cast<R>(extra_digits));

  if (a->value() % factor != 0) { return errc::OVERFLOW; }

  *out = *a;
  return errc::OK;
}

template <typename T>
__device__ inline errc ansi_precision_cast(optional<T>* out,
                                           optional<T> const* a,
                                           optional<int32_t> const* precision)
{
  if (a->has_value()) {
    return ansi_precision_cast(&out->value(), &a->value(), &precision->value());
  } else {
    *out = nullopt;
    return errc::OK;
  }
}

template <typename T>
__device__ inline errc ansi_try_add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
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
  if (a->has_value() && b->has_value()) {
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
  if (a->has_value() && b->has_value()) {
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
  if (a->has_value() && b->has_value()) {
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
__device__ inline errc ansi_try_mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mod(&r, &a->value(), &b->value()); e != errc::OK) {
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
  if (a->has_value()) {
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
  if (a->has_value()) {
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

template <typename R>
__device__ inline errc ansi_try_precision_cast(optional<decimal<R>>* out,
                                               optional<decimal<R>> const* a,
                                               optional<int32_t> const* precision)
{
  if (a->has_value() && precision->has_value()) {
    if (errc e = ansi_precision_cast(&out->value(), &a->value(), &precision->value());
        e != errc::OK) {
      *out = nullopt;
    } else {
      *out = a->value();
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
