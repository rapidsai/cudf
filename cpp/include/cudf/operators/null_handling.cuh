/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/operators/types.cuh>

#include <format>

namespace CUDF_EXPORT cudf {
namespace ops {

template <typename T>
__device__ inline errc is_null(bool* out, T const* a)
{
  *out = false;
  return errc::OK;
}

template <typename T>
__device__ inline errc is_null(optional<bool>* out, optional<T> const* a)
{
  *out = !a->has_value();
  return errc::OK;
}

template <typename T>
__device__ inline errc nullify_if(optional<T>* out,
                                  optional<bool> const* condition,
                                  optional<T> const* a)
{
  if (condition->has_value() && a->has_value()) {
    if (condition->value()) {
      *out = nullopt;
    } else {
      *out = a->value();
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc coalesce(T* out, T const* a, T const* b)
{
  *out = *a;
  return errc::OK;
}

template <typename T>
__device__ inline errc coalesce(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value()) {
    *out = a->value();
  } else if (b->has_value()) {
    *out = b->value();
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
