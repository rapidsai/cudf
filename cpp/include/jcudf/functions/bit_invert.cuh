/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jcudf {
namespace functions {

template <typename T>
__device__ inline void bit_invert(T* out, T const& a)
{
  *out = ~a;
}

template <typename T>
__device__ inline void bit_invert(optional<T>* out, optional<T> const& a)
{
  if (a.has_value()) {
    T r;
    bit_invert(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace jcudf
