/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

template <typename T>
__device__ inline void cast_to_u64(u64* out, T const& a)
{
  *out = static_cast<u64>(a);
}

template <typename T>
__device__ inline void cast_to_u64(optional<u64>* out, optional<T> const& a)
{
  if (a.has_value()) {
    u64 r;
    cast_to_u64(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
