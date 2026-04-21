/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {

template <typename T>
__device__ inline errc logical_and(T* out, T const* a, T const* b)
{
  *out = (*a && *b);
  return errc::OK;
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
  if (a->is_valid() && b->is_valid()) {
    T r;
    logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc if_else(T* out, bool const* pred, T const* true_value, T const* false_value)
{
  *out = *pred ? *true_value : *false_value;
  return errc::OK;
}

template <typename T>
__device__ inline errc if_else(optional<T>* out,
                               optional<bool> const* pred,
                               optional<T> const* true_value,
                               optional<T> const* false_value)
{
  if (pred->is_valid() && true_value->is_valid() && false_value->is_valid()) {
    if_else<T>(&out->value(), &pred->value(), &true_value->value(), &false_value->value());
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
