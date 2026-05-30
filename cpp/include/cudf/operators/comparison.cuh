/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/concepts.cuh>
#include <cudf/operators/types.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Tests `a == b`.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool equal(T a, T b)
  requires(!nullable<T> && cuda::std::equality_comparable<T>)
{
  return (a == b);
}

/**
 * @brief Tests `a != b`.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool not_equal(T a, T b)
  requires(!nullable<T> && cuda::std::equality_comparable<T>)
{
  return (a != b);
}

/**
 * @brief Tests `a > b`.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool greater(T a, T b)
  requires(!nullable<T> && cuda::std::totally_ordered<T>)
{
  return (a > b);
}

/**
 * @brief Tests `a >= b`.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool greater_equal(T a, T b)
  requires(!nullable<T> && cuda::std::totally_ordered<T>)
{
  return (a >= b);
}

/**
 * @brief Tests `a < b`.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool less(T a, T b)
  requires(!nullable<T> && cuda::std::totally_ordered<T>)
{
  return (a < b);
}

/**
 * @brief Tests `a <= b`.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool less_equal(T a, T b)
  requires(!nullable<T> && cuda::std::totally_ordered<T>)
{
  return (a <= b);
}

/**
 * @brief Tests equality between two values for null-aware equality semantics.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool null_equal(T a, T b)
  requires(!nullable<T> && cuda::std::equality_comparable<T>)
{
  return (a == b);
}

/**
 * @brief Tests equality between two values for null-aware equality semantics.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ bool null_equal(optional<T> a, optional<T> b)
  requires(!nullable<T> && cuda::std::equality_comparable<T>)
{
  if (a.has_value() && b.has_value()) {
    return null_equal(a.value(), b.value());
  } else if (!a.has_value() && !b.has_value()) {
    return true;
  } else {
    return false;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
