/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

template <typename T>
__device__ inline void pymod(T* out, T const& a, T const& b)
{
  *out = (a % b + b) % b;
}

__device__ inline void pymod(f32* out, f32 const& a, f32 const& b)
{
  *out = __builtin_fmodf(__builtin_fmodf(a, b) + b, b);
}

__device__ inline void pymod(f64* out, f64 const& a, f64 const& b)
{
  *out = __builtin_fmod(__builtin_fmod(a, b) + b, b);
}

template <typename T>
__device__ inline void pymod(optional<T>* out, optional<T> const& a, optional<T> const& b)
{
  if (a.has_value() && b.has_value()) {
    T r;
    pymod(&r, *a, *b);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
