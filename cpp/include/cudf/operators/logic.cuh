/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {
namespace ops {

template <typename T>
__device__ inline errc null_logical_and(T* out, T const* a, T const* b)
{
  *out = (*a && *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc null_logical_and(optional<T>* out,
                                        optional<T> const* a,
                                        optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    null_logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = nullopt;
  } else {
    if (a->has_value() ? *(*a) : *(*b)) {
      *out = nullopt;
    } else {
      *out = false;
    }
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc null_logical_or(T* out, T const* a, T const* b)
{
  *out = (*a || *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc null_logical_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    null_logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = nullopt;
  } else {
    if (a->has_value() ? *(*a) : *(*b)) {
      *out = true;
    } else {
      *out = nullopt;
    }
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc logical_and(T* out, T const* a, T const* b)
{
  *out = (*a && *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc logical_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc logical_or(T* out, T const* a, T const* b)
{
  *out = (*a || *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc logical_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc logical_not(T* out, T const* a)
{
  *out = !(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc logical_not(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    logical_not(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc if_else(T* out, T const* true_value, T const* false_value, bool const* pred)
{
  *out = *pred ? *true_value : *false_value;
  return errc::OK;
}

template <typename T>
__device__ inline errc if_else(optional<T>* out,
                               optional<T> const* true_value,
                               optional<T> const* false_value,
                               optional<bool> const* pred)
{
  if (pred->has_value() && true_value->has_value() && false_value->has_value()) {
    if_else<T>(&out->value(), &pred->value(), &true_value->value(), &false_value->value());
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
