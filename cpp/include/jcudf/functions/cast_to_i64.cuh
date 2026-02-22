/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jcudf {
namespace functions {

template <typename T>
__device__ inline void cast_to_i64(i64* out, T const& a)
{
  *out = static_cast<i64>(a);
}

template <typename T>
__device__ inline void cast_to_i64(optional<i64>* out, optional<T> const& a)
{
  if (a.has_value()) {
    i64 r;
    cast_to_i64(&r, *a);
    *out = r;
  } else {
    *out = nullopt;
  }
}

}  // namespace functions
}  // namespace jcudf
