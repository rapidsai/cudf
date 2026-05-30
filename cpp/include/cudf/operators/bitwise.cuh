/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/concepts.cuh>
#include <cudf/operators/types.cuh>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes bitwise AND of two values.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <integer T>
__device__ T bit_and(T a, T b)
{
  return (a & b);
}

/**
 * @brief Computes bitwise NOT of one value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <integer T>
__device__ T bit_invert(T a)
{
  return ~a;
}

/**
 * @brief Computes bitwise OR of two values.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <integer T>
__device__ T bit_or(T a, T b)
{
  return (a | b);
}

/**
 * @brief Computes bitwise XOR of two values.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <integer T>
__device__ T bit_xor(T a, T b)
{
  return (a ^ b);
}

/**
 * @brief Shifts a value left by a bit count.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @param b Shift count.
 */
template <integer T>
__device__ T bit_shift_left(T a, T b)
{
  return (a << b);
}

/**
 * @brief Shifts a value right by a bit count.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @param b Shift count.
 */
template <integer T>
__device__ T bit_shift_right(T a, T b)
{
  return (a >> b);
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
