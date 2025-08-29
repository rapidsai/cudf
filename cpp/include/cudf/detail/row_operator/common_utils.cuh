/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <cmath>

namespace cudf::detail {

/**
 * @brief Result type of comparison operations.
 *
 * Indicates how two elements `a` and `b` compare with one and another.
 *
 * Equivalence is defined as `not (a<b) and not (b<a)`. Elements that are
 * EQUIVALENT may not necessarily be *equal*.
 */
enum class weak_ordering {
  LESS,        ///< Indicates `a` is less than (ordered before) `b`
  EQUIVALENT,  ///< Indicates `a` is ordered neither before nor after `b`
  GREATER      ///< Indicates `a` is greater than (ordered after) `b`
};

/**
 * @brief Compare the elements ordering with respect to `lhs`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return Indicates the relationship between the elements in
 * the `lhs` and `rhs` columns.
 */
template <typename Element>
__device__ weak_ordering compare_elements(Element lhs, Element rhs)
{
  if (lhs < rhs) {
    return weak_ordering::LESS;
  } else if (rhs < lhs) {
    return weak_ordering::GREATER;
  }
  return weak_ordering::EQUIVALENT;
}

/**
 * @brief A specialization for floating-point `Element` type relational comparison
 * to derive the order of the elements with respect to `lhs`.
 *
 * This specialization handles `nan` in the following order:
 * `[-Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN, null] (for null_order::AFTER)`
 * `[null, -Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN] (for null_order::BEFORE)`
 *
 * @param lhs The first element
 * @param rhs The second element
 * @return Indicates the relationship between the elements
 */
template <typename Element>
__device__ weak_ordering relational_compare(Element lhs, Element rhs)
  requires(std::is_floating_point_v<Element>)
{
  if (isnan(lhs) and isnan(rhs)) {
    return weak_ordering::EQUIVALENT;
  } else if (isnan(rhs)) {
    return weak_ordering::LESS;
  } else if (isnan(lhs)) {
    return weak_ordering::GREATER;
  }

  return detail::compare_elements(lhs, rhs);
}

/**
 * @brief Compare the nulls according to null order.
 *
 * @param lhs_is_null boolean representing if lhs is null
 * @param rhs_is_null boolean representing if rhs is null
 * @param null_precedence null order
 * @return Indicates the relationship between null in lhs and rhs columns.
 */
inline __device__ auto null_compare(bool lhs_is_null, bool rhs_is_null, null_order null_precedence)
{
  if (lhs_is_null and rhs_is_null) {  // null <? null
    return weak_ordering::EQUIVALENT;
  } else if (lhs_is_null) {  // null <? x
    return (null_precedence == null_order::BEFORE) ? weak_ordering::LESS : weak_ordering::GREATER;
  } else if (rhs_is_null) {  // x <? null
    return (null_precedence == null_order::AFTER) ? weak_ordering::LESS : weak_ordering::GREATER;
  }
  return weak_ordering::EQUIVALENT;
}

/**
 * @brief A specialization for non-floating-point `Element` type relational
 * comparison to derive the order of the elements with respect to `lhs`.
 *
 * @param lhs The first element
 * @param rhs The second element
 * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns
 */
template <typename Element>
__device__ weak_ordering relational_compare(Element lhs, Element rhs)
  requires(not std::is_floating_point_v<Element>)
{
  return detail::compare_elements(lhs, rhs);
}

/**
 * @brief A specialization for floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`. `nan == nan`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element>
__device__ bool equality_compare(Element lhs, Element rhs)
  requires(std::is_floating_point_v<Element>)
{
  if (isnan(lhs) and isnan(rhs)) { return true; }
  return lhs == rhs;
}

/**
 * @brief A specialization for non-floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element>
__device__ bool equality_compare(Element const lhs, Element const rhs)
  requires(not std::is_floating_point_v<Element>)
{
  return lhs == rhs;
}

}  // namespace cudf::detail
