/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes logical AND for non-optional operands with null-aware semantics.
 *
 * @tparam T Operand type.
 * @param out Destination for the logical result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc null_logical_and(bool* out, T const* a, T const* b)
{
  *out = (*a && *b);
  return errc::OK;
}

/**
 * @brief Computes logical AND for optional operands with three-valued semantics.
 *
 * @tparam T Operand type.
 * @param out Destination optional logical result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc null_logical_and(optional<bool>* out,
                                        optional<T> const* a,
                                        optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    null_logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = nullopt;
  } else {
    if (a->has_value() ? *(*a) : *(*b)) {
      *out = nullopt;
    } else {
      *out = false;
    }
  }
  return errc::OK;
}

/**
 * @brief Computes logical OR for non-optional operands with null-aware semantics.
 *
 * @tparam T Operand type.
 * @param out Destination for the logical result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc null_logical_or(bool* out, T const* a, T const* b)
{
  *out = (*a || *b);
  return errc::OK;
}

/**
 * @brief Computes logical OR for optional operands with three-valued semantics.
 *
 * @tparam T Operand type.
 * @param out Destination optional logical result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc null_logical_or(optional<bool>* out,
                                       optional<T> const* a,
                                       optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    null_logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = nullopt;
  } else {
    if (a->has_value() ? *(*a) : *(*b)) {
      *out = true;
    } else {
      *out = nullopt;
    }
  }
  return errc::OK;
}

/**
 * @brief Computes logical AND for non-optional operands.
 *
 * @tparam T Operand type.
 * @param out Destination for the logical result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc logical_and(bool* out, T const* a, T const* b)
{
  *out = (*a && *b);
  return errc::OK;
}

/**
 * @brief Computes logical AND for optional operands.
 *
 * @tparam T Operand type.
 * @param out Destination optional logical result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc logical_and(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes logical OR for non-optional operands.
 *
 * @tparam T Operand type.
 * @param out Destination for the logical result.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc logical_or(bool* out, T const* a, T const* b)
{
  *out = (*a || *b);
  return errc::OK;
}

/**
 * @brief Computes logical OR for optional operands.
 *
 * @tparam T Operand type.
 * @param out Destination optional logical result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc logical_or(optional<bool>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes logical NOT for a non-optional operand.
 *
 * @tparam T Operand type.
 * @param out Destination for the logical result.
 * @param a Input operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc logical_not(bool* out, T const* a)
{
  *out = !(*a);
  return errc::OK;
}

/**
 * @brief Computes logical NOT for an optional operand.
 *
 * @tparam T Operand type.
 * @param out Destination optional logical result.
 * @param a Optional input operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc logical_not(optional<bool>* out, optional<T> const* a)
{
  if (a->has_value()) {
    bool r;
    logical_not(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Selects one of two values based on a boolean predicate.
 *
 * @tparam T Selected value type.
 * @param out Destination for the selected value.
 * @param true_value Value selected when @p pred is true.
 * @param false_value Value selected when @p pred is false.
 * @param pred Selection predicate.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc if_else(T* out, T const* true_value, T const* false_value, bool const* pred)
{
  *out = *pred ? *true_value : *false_value;
  return errc::OK;
}

/**
 * @brief Selects one of two optional values based on an optional predicate.
 *
 * @tparam T Selected value type.
 * @param out Destination optional selected value.
 * @param true_value Optional value selected when @p pred is true.
 * @param false_value Optional value selected when @p pred is false.
 * @param pred Optional selection predicate.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc if_else(optional<T>* out,
                               optional<T> const* true_value,
                               optional<T> const* false_value,
                               optional<bool> const* pred)
{
  if (pred->has_value() && true_value->has_value() && false_value->has_value()) {
    T r;
    if_else<T>(&r, &true_value->value(), &false_value->value(), &pred->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
