/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {

template <typename T>
__device__ inline errc equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
  return errc::OK;
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
  return errc::OK;
}

template <typename T>
__device__ inline errc greater(bool* out, T const* a, T const* b)
{
  *out = (*a > *b);
  return errc::OK;
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
  return errc::OK;
}

template <typename T>
__device__ inline errc greater_equal(bool* out, T const* a, T const* b)
{
  *out = (*a >= *b);
  return errc::OK;
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
  return errc::OK;
}

template <typename T>
__device__ inline errc less(bool* out, T const* a, T const* b)
{
  *out = (*a < *b);
  return errc::OK;
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
  return errc::OK;
}

template <typename T>
__device__ inline errc less_equal(bool* out, T const* a, T const* b)
{
  *out = (*a <= *b);
  return errc::OK;
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
  return errc::OK;
}

template <typename T>
__device__ inline errc null_equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
  return errc::OK;
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
  return errc::OK;
}

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
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
