/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

template <typename T>
__device__ inline void is_null(bool* out, T const& a)
{
  *out = false;
}

template <typename T>
__device__ inline void is_null(optional<bool>* out, optional<T> const& a)
{
  *out = a.has_null();
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
