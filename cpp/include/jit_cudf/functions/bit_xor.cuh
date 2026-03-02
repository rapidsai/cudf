/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

template <typename T>
__device__ inline void bit_xor(T* out, T const& a, T const& b)
{
  *out = a ^ b;
}

template <typename T>
__device__ inline void bit_xor(optional<T>* out, optional<T> const& a, optional<T> const& b)
{
  if (a.has_value() && b.has_value()) {
    T r;
    bit_xor(&r, *a, *b);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
