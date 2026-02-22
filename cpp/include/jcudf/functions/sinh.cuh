/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jcudf {
namespace functions {

__device__ inline void sinh(f32* out, f32 const& a) { *out = __builtin_sinhf(a); }

__device__ inline void sinh(f64* out, f64 const& a) { *out = __builtin_sinh(a); }

template <typename T>
__device__ inline void sinh(optional<T>* out, optional<T> const& a)
{
  if (a.has_value()) {
    T r;
    sinh(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace jcudf
