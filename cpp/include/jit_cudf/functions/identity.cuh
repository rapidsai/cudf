/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

template <typename T>
__device__ inline void identity(T* out, T const& a)
{
  *out = a;
}

template <typename T>
__device__ inline void identity(optional<T>* out, optional<T> const& a)
{
  *out = a;
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
