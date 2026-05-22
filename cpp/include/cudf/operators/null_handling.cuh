/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Returns false for non-optional inputs.
 *
 * @tparam T Input type.
 * @param out Destination for the null test result.
 * @param a Input value.
 */
template <typename T>
__device__ void is_null(bool* out, T const* a)
{
  *out = false;
}

/**
 * @brief Tests whether an optional input is null.
 *
 * @tparam T Input value type.
 * @param out Destination optional boolean result.
 * @param a Optional input value.
 */
template <typename T>
__device__ void is_null(cuda::std::optional<bool>* out, cuda::std::optional<T> const* a)
{
  *out = !a->has_value();
}

/**
 * @brief Sets the output to null when the condition is true.
 *
 * @tparam T Value type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @param condition Optional boolean condition.
 */
template <typename T>
__device__ void nullify_if(cuda::std::optional<T>* out,
                           cuda::std::optional<T> const* a,
                           cuda::std::optional<bool> const* condition)
{
  if (condition->has_value() && a->has_value()) {
    if (condition->value()) {
      *out = cuda::std::nullopt;
    } else {
      *out = a->value();
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Returns the first non-null input value of two non-nullable values.
 *
 * @tparam T Value type.
 * @param out Destination value.
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__device__ void coalesce(T* out, T const* a, T const* b)
{
  *out = *a;
}

/**
 * @brief Returns the first non-null optional value of two optional values, otherwise null.
 *
 * @tparam T Value type.
 * @param out Destination optional value.
 * @param a First optional value.
 * @param b Second optional value.
 */
template <typename T>
__device__ void coalesce(cuda::std::optional<T>* out,
                         cuda::std::optional<T> const* a,
                         cuda::std::optional<T> const* b)
{
  if (a->has_value()) {
    *out = a->value();
  } else if (b->has_value()) {
    *out = b->value();
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Returns the input boolean predicate unchanged.
 *
 * @param out Destination boolean predicate.
 * @param a Input boolean predicate.
 */
__device__ inline void predicate(bool* out, bool const* a) { *out = *a; }

/**
 * @brief Converts an optional predicate to a non-nullable predicate.
 *
 * @param out Destination optional boolean predicate.
 * @param a Optional input boolean predicate.
 */
__device__ inline void predicate(cuda::std::optional<bool>* out, cuda::std::optional<bool> const* a)
{
  if (a->has_value()) {
    *out = a->value();
  } else {
    *out = false;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
