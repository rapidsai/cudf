/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Tests equality between two values.
 *
 * @tparam T Operand type.
 * @param out Destination for the comparison result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
  return errc::OK;
}

/**
 * @brief Tests equality between optional operands.
 *
 * @tparam T Operand type.
 * @param out Destination optional boolean result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    equal(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = true;
  } else {
    *out = false;
  }
  return errc::OK;
}

/**
 * @brief Tests inequality between two values.
 *
 * @tparam T Operand type.
 * @param out Destination for the comparison result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc not_equal(bool* out, T const* a, T const* b)
{
  *out = (*a != *b);
  return errc::OK;
}

/**
 * @brief Tests inequality between optional operands.
 *
 * @tparam T Operand type.
 * @param out Destination optional boolean result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc not_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    not_equal(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = false;
  } else {
    *out = true;
  }
  return errc::OK;
}

/**
 * @brief Tests whether the left operand is greater than the right operand.
 *
 * @tparam T Operand type.
 * @param out Destination for the comparison result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc greater(bool* out, T const* a, T const* b)
{
  *out = (*a > *b);
  return errc::OK;
}

/**
 * @brief Tests whether one optional operand is greater than another.
 *
 * @tparam T Operand type.
 * @param out Destination optional boolean result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc greater(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    greater(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::OK;
}

/**
 * @brief Tests whether the left operand is greater than or equal to the right operand.
 *
 * @tparam T Operand type.
 * @param out Destination for the comparison result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc greater_equal(bool* out, T const* a, T const* b)
{
  *out = (*a >= *b);
  return errc::OK;
}

/**
 * @brief Tests whether one optional operand is greater than or equal to another.
 *
 * @tparam T Operand type.
 * @param out Destination optional boolean result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc greater_equal(optional<bool>* out,
                                     optional<T> const* a,
                                     optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    greater_equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::OK;
}

/**
 * @brief Tests whether the left operand is less than the right operand.
 *
 * @tparam T Operand type.
 * @param out Destination for the comparison result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc less(bool* out, T const* a, T const* b)
{
  *out = (*a < *b);
  return errc::OK;
}

/**
 * @brief Tests whether one optional operand is less than another.
 *
 * @tparam T Operand type.
 * @param out Destination optional boolean result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc less(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    less(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::OK;
}

/**
 * @brief Tests whether the left operand is less than or equal to the right operand.
 *
 * @tparam T Operand type.
 * @param out Destination for the comparison result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc less_equal(bool* out, T const* a, T const* b)
{
  *out = (*a <= *b);
  return errc::OK;
}

/**
 * @brief Tests whether one optional operand is less than or equal to another.
 *
 * @tparam T Operand type.
 * @param out Destination optional boolean result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc less_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    less_equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = false;
  }
  return errc::OK;
}

/**
 * @brief Tests equality between two non-optional values for null-aware equality semantics.
 *
 * @tparam T Operand type.
 * @param out Destination for the comparison result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc null_equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
  return errc::OK;
}

/**
 * @brief Tests null-aware equality between optional operands.
 *
 * @tparam T Operand type.
 * @param out Destination optional boolean result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc null_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    *out = (*(*a) == *(*b));
  } else if (!a->has_value() && !b->has_value()) {
    *out = true;
  } else {
    *out = false;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
