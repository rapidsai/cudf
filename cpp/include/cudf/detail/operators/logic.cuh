/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Computes logical AND.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <cuda::std::integral A, cuda::std::integral B>
__device__ bool logical_and(A a, B b)
{
  return a && b;
}

/**
 * @brief Computes logical OR.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <cuda::std::integral A, cuda::std::integral B>
__device__ bool logical_or(A a, B b)
{
  return a || b;
}

/**
 * @brief Computes logical NOT.
 *
 * @tparam T Value type.
 * @param a Input operand.
 */
template <cuda::std::integral T>
__device__ bool logical_not(T a)
{
  return !a;
}

/**
 * @brief Computes logical AND with null-aware semantics.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <cuda::std::integral A, cuda::std::integral B>
__device__ bool null_logical_and(A a, B b)
{
  return logical_and(a, b);
}

template <cuda::std::integral A, cuda::std::integral B>
__device__ cuda::std::optional<bool> null_logical_and(cuda::std::optional<A> a,
                                                      cuda::std::optional<B> b)
{
  if (a.has_value() && b.has_value()) {
    return null_logical_and(a.value(), b.value());
  } else if (!a.has_value() && !b.has_value()) {
    return {};
  } else {
    if (a.has_value() ? *a : *b) {
      return {};
    } else {
      return false;
    }
  }
}

/**
 * @brief Computes logical OR with null-aware semantics.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <cuda::std::integral A, cuda::std::integral B>
__device__ bool null_logical_or(A a, B b)
{
  return logical_or(a, b);
}

template <cuda::std::integral A, cuda::std::integral B>
__device__ cuda::std::optional<bool> null_logical_or(cuda::std::optional<A> a,
                                                     cuda::std::optional<B> b)
{
  if (a.has_value() && b.has_value()) {
    return null_logical_or(a.value(), b.value());
  } else if (!a.has_value() && !b.has_value()) {
    return {};
  } else {
    if (a.has_value() ? *a : *b) {
      return true;
    } else {
      return {};
    }
  }
}

/**
 * @brief Selects one of two values based on a predicate.
 *
 * @tparam T Selected value type.
 * @param true_value Value selected when @p pred is true.
 * @param false_value Value selected when @p pred is false.
 * @param pred Selection predicate.
 */
template <typename T>
__device__ T if_else(T true_value, T false_value, bool pred)
{
  return pred ? true_value : false_value;
}

template <typename T>
__device__ cuda::std::optional<T> if_else(cuda::std::optional<T> true_value,
                                          cuda::std::optional<T> false_value,
                                          cuda::std::optional<bool> pred)
{
  return pred.value_or(false) ? true_value : false_value;
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
