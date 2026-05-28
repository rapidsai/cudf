/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes logical AND with null-aware semantics.
 *
 * @tparam T Operand type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void null_logical_and(bool* out, T const* a, T const* b)
{
  *out = (*a && *b);
}

/**
 * @brief Computes logical AND with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void null_logical_and(cuda::std::optional<bool>* out,
                                 cuda::std::optional<T> const* a,
                                 cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    null_logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = cuda::std::nullopt;
  } else {
    if (a->has_value() ? *(*a) : *(*b)) {
      *out = cuda::std::nullopt;
    } else {
      *out = false;
    }
  }
}

/**
 * @brief Computes logical OR with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void null_logical_or(bool* out, T const* a, T const* b)
{
  *out = (*a || *b);
}

/**
 * @brief Computes logical OR with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void null_logical_or(cuda::std::optional<bool>* out,
                                cuda::std::optional<T> const* a,
                                cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    null_logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = cuda::std::nullopt;
  } else {
    if (a->has_value() ? *(*a) : *(*b)) {
      *out = true;
    } else {
      *out = cuda::std::nullopt;
    }
  }
}

/**
 * @brief Computes logical AND with null-aware semantics.
 *
 * @tparam T Operand type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void logical_and(bool* out, T const* a, T const* b)
{
  *out = (*a && *b);
}

/**
 * @brief Computes logical AND with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void logical_and(cuda::std::optional<bool>* out,
                            cuda::std::optional<T> const* a,
                            cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    logical_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes logical OR with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void logical_or(bool* out, T const* a, T const* b)
{
  *out = (*a || *b);
}

/**
 * @brief Computes logical OR with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void logical_or(cuda::std::optional<bool>* out,
                           cuda::std::optional<T> const* a,
                           cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    logical_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes logical NOT with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input operand.
 */
template <typename T>
__device__ void logical_not(bool* out, T const* a)
{
  *out = !(*a);
}

/**
 * @brief Computes logical NOT with null-aware semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input operand.
 */
template <typename T>
__device__ void logical_not(cuda::std::optional<bool>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    bool r;
    logical_not(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Selects one of two values based on a predicate.
 *
 * @tparam T Selected value type.
 * @param out Result destination.
 * @param true_value Value selected when @p pred is true.
 * @param false_value Value selected when @p pred is false.
 * @param pred Selection predicate.
 */
template <typename T>
__device__ void if_else(T* out, T const* true_value, T const* false_value, bool const* pred)
{
  *out = *pred ? *true_value : *false_value;
}

/**
 * @brief Selects one of two values based on a predicate.
 *
 * @tparam T Selected value type.
 * @param out Result destination.
 * @param true_value Optional value selected when @p pred is true.
 * @param false_value Optional value selected when @p pred is false or null.
 * @param pred Selection predicate.
 */
template <typename T>
__device__ void if_else(cuda::std::optional<T>* out,
                        cuda::std::optional<T> const* true_value,
                        cuda::std::optional<T> const* false_value,
                        cuda::std::optional<bool> const* pred)
{
  *out = pred->value_or(false) ? *true_value : *false_value;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
