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
 * @brief Tests `a == b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
}

/**
 * @brief Tests `a == b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void equal(cuda::std::optional<bool>* out,
                      cuda::std::optional<T> const* a,
                      cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Tests `a != b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void not_equal(bool* out, T const* a, T const* b)
{
  *out = (*a != *b);
}

/**
 * @brief Tests `a != b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void not_equal(cuda::std::optional<bool>* out,
                          cuda::std::optional<T> const* a,
                          cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    not_equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Tests `a > b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void greater(bool* out, T const* a, T const* b)
{
  *out = (*a > *b);
}

/**
 * @brief Tests `a > b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void greater(cuda::std::optional<bool>* out,
                        cuda::std::optional<T> const* a,
                        cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    greater(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Tests `a >= b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void greater_equal(bool* out, T const* a, T const* b)
{
  *out = (*a >= *b);
}

/**
 * @brief Tests `a >= b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void greater_equal(cuda::std::optional<bool>* out,
                              cuda::std::optional<T> const* a,
                              cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    greater_equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Tests `a < b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void less(bool* out, T const* a, T const* b)
{
  *out = (*a < *b);
}

/**
 * @brief Tests `a < b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void less(cuda::std::optional<bool>* out,
                     cuda::std::optional<T> const* a,
                     cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    less(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Tests `a <= b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void less_equal(bool* out, T const* a, T const* b)
{
  *out = (*a <= *b);
}

/**
 * @brief Tests `a <= b`.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void less_equal(cuda::std::optional<bool>* out,
                           cuda::std::optional<T> const* a,
                           cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    less_equal(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Tests equality between two values for null-aware equality semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void null_equal(bool* out, T const* a, T const* b)
{
  *out = (*a == *b);
}

/**
 * @brief Tests equality between two values for null-aware equality semantics.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void null_equal(cuda::std::optional<bool>* out,
                           cuda::std::optional<T> const* a,
                           cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    bool r;
    null_equal(&r, &a->value(), &b->value());
    *out = r;
  } else if (!a->has_value() && !b->has_value()) {
    *out = true;
  } else {
    *out = false;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
