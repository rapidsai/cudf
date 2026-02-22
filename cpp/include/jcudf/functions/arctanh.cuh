/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jcudf {
namespace functions {

__device__ inline void arctanh(f32* out, f32 const& a) { *out = __builtin_atanhf(a); }

__device__ inline void arctanh(f64* out, f64 const& a) { *out = __builtin_atanh(a); }

template <typename T>
__device__ inline void arctanh(optional<T>* out, optional<T> const& a)
{
  if (a.has_value()) {
    T r;
    arctanh(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace jcudf
