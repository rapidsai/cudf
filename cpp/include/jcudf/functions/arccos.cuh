/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jcudf {
namespace functions {

__device__ inline void arccos(f32* out, f32 const& a) { *out = __builtin_acosf(a); }

__device__ inline void arccos(f64* out, f64 const& a) { *out = __builtin_acos(a); }

template <typename T>
__device__ inline void arccos(optional<T>* out, optional<T> const& a)
{
  if (a.has_value()) {
    T r;
    arccos(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace jcudf
