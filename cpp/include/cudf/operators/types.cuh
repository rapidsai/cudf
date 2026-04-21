/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/operators/int256.cuh>
#include <cudf/operators/optional.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/durations.hpp>

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {

namespace ops {

enum errc : int { OK = 0, OVERFLOW = 1, DIVISION_BY_ZERO = 2 };

template <typename T>
struct promoted_t;

template <>
struct promoted_t<int8_t> {
  using type = int16_t;
};

template <>
struct promoted_t<uint8_t> {
  using type = uint16_t;
};

template <>
struct promoted_t<int16_t> {
  using type = int32_t;
};

template <>
struct promoted_t<uint16_t> {
  using type = uint32_t;
};

template <>
struct promoted_t<int32_t> {
  using type = int64_t;
};

template <>
struct promoted_t<uint32_t> {
  using type = uint64_t;
};

template <>
struct promoted_t<int64_t> {
  using type = __int128;
};

template <>
struct promoted_t<uint64_t> {
  using type = unsigned __int128;
};

template <>
struct promoted_t<__int128> {
  using type = int256_t;
};

template <>
struct promoted_t<unsigned __int128> {
  using type = uint256_t;
};

template <typename T>
using promoted = typename promoted_t<T>::type;

template <typename T>
__device__ inline errc identity(T* out, T const* a)
{
  *out = *a;
  return errc::OK;
}

template <typename T>
__device__ inline errc identity(optional<T>* out, optional<T> const* a)
{
  *out = *a;
  return errc::OK;
}

template <typename T>
__device__ inline errc is_null(bool* out, T const* a)
{
  *out = false;
  return errc::OK;
}

template <typename T>
__device__ inline errc is_null(optional<bool>* out, optional<T> const* a)
{
  *out = a->is_null();
  return errc::OK;
}

// TODO: decimal ansi operators(precision and scale-oriented non-templated arguments)
// TODO: cast operators to match AST
// TODO: decimal cast operators to match AST
// TODO: datetime cast operators & arithmetic
// TODO: decimal ansi cast
// TODO: ansi_mod, div operations for fixed-point and duration types

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
