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
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  return a == b;
}

/**
 * @brief Tests `a != b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool not_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a != b; })
{
  return a != b;
}

/**
 * @brief Tests `a > b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool greater(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a > b; })
{
  return a > b;
}

/**
 * @brief Tests `a >= b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool greater_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a >= b; })
{
  return a >= b;
}

/**
 * @brief Tests `a < b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool less(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a < b; })
{
  return a < b;
}

/**
 * @brief Tests `a <= b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool less_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a <= b; })
{
  return a <= b;
}

/**
 * @brief Tests equality between two values for null-aware equality semantics.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool null_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  return a == b;
}

template <typename A, typename B>
__device__ bool null_equal(optional<A> a, optional<B> b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
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
