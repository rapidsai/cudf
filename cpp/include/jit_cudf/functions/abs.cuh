/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

template <typename T>
__device__ inline void abs(T* out, T const& a)
{
  *out = (a < 0) ? -a : a;
}

template <typename T>
__device__ inline void abs(optional<T>* out, optional<T> const& a)
{
  if (a.has_value()) {
    T r;
    abs(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
