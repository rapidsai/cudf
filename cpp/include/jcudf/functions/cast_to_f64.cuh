/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jcudf {
namespace functions {

template <typename T>
__device__ inline void cast_to_f64(f64* out, T const& a)
{
  *out = static_cast<f64>(a);
}

template <typename T>
__device__ inline void cast_to_f64(optional<f64>* out, optional<T> const& a)
{
  if (a.has_value()) {
    f64 r;
    cast_to_f64(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace jcudf
