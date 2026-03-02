/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

__device__ inline void cos(f32* out, f32 const& a) { *out = __builtin_cosf(a); }

__device__ inline void cos(f64* out, f64 const& a) { *out = __builtin_cos(a); }

template <typename T>
__device__ inline void cos(optional<T>* out, optional<T> const& a)
{
  if (a.has_value()) {
    T r;
    cos(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
