/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jcudf {
namespace functions {

template <typename T>
__device__ inline void mod(T* out, T const& a, T const& b)
{
  *out = a % b;
}

__device__ inline void mod(f32* out, f32 const& a, f32 const& b) { *out = __builtin_fmodf(a, b); }

__device__ inline void mod(f64* out, f64 const& a, f64 const& b) { *out = __builtin_fmod(a, b); }

template <typename T>
__device__ inline void mod(optional<T>* out, optional<T> const& a, optional<T> const& b)
{
  if (a.has_value() && b.has_value()) {
    T r;
    mod(&r, *a, *b);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace jcudf
