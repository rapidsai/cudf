/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {
namespace ops {

template <typename T>
__device__ inline errc bit_and(T* out, T const* a, T const* b)
{
  *out = (*a & *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc bit_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc bit_invert(T* out, T const* a)
{
  *out = ~(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc bit_invert(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    bit_invert(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc bit_or(T* out, T const* a, T const* b)
{
  *out = (*a | *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc bit_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc bit_xor(T* out, T const* a, T const* b)
{
  *out = (*a ^ *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc bit_xor(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_xor(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
