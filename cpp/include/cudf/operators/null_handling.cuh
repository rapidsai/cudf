/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Returns false for non-optional inputs.
 *
 * @tparam T Input type.
 * @param out Destination for the null test result.
 * @param a Input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc is_null(bool* out, T const* a)
{
  *out = false;
  return errc::OK;
}

/**
 * @brief Tests whether an optional input is null.
 *
 * @tparam T Input value type.
 * @param out Destination optional boolean result.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc is_null(optional<bool>* out, optional<T> const* a)
{
  *out = !a->has_value();
  return errc::OK;
}

/**
 * @brief Sets the output to null when the condition is true.
 *
 * @tparam T Value type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @param condition Optional boolean condition.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc nullify_if(optional<T>* out,
                                  optional<T> const* a,
                                  optional<bool> const* condition)
{
  if (condition->has_value() && a->has_value()) {
    if (condition->value()) {
      *out = nullopt;
    } else {
      *out = a->value();
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Returns the first non-null input value of two non-nullable values.
 *
 * @tparam T Value type.
 * @param out Destination value.
 * @param a First value.
 * @param b Second value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc coalesce(T* out, T const* a, T const* b)
{
  *out = *a;
  return errc::OK;
}

/**
 * @brief Returns the first non-null optional value of two optional values, otherwise null.
 *
 * @tparam T Value type.
 * @param out Destination optional value.
 * @param a First optional value.
 * @param b Second optional value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc coalesce(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value()) {
    *out = a->value();
  } else if (b->has_value()) {
    *out = b->value();
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Returns the input boolean predicate unchanged.
 *
 * @param out Destination boolean predicate.
 * @param a Input boolean predicate.
 * @return errc::OK.
 */
__device__ inline errc predicate(bool* out, bool const* a)
{
  *out = *a;
  return errc::OK;
}

/**
 * @brief Converts an optional predicate to a non-nullable predicate.
 *
 * @param out Destination optional boolean predicate.
 * @param a Optional input boolean predicate.
 * @return errc::OK.
 */
__device__ inline errc predicate(optional<bool>* out, optional<bool> const* a)
{
  if (a->has_value()) {
    *out = a->value();
  } else {
    *out = false;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
