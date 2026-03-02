/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

__device__ inline void pow(f32* out, f32 const& a, f32 const& b) { *out = __builtin_powf(a, b); }

__device__ inline void pow(f64* out, f64 const& a, f64 const& b) { *out = __builtin_pow(a, b); }

template <typename T>
__device__ inline void pow(optional<T>* out, optional<T> const& a, optional<T> const& b)
{
  if (a.has_value() && b.has_value()) {
    T r;
    pow(&r, *a, *b);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
