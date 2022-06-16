/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <cudf/types.hpp>

#include <limits>

/**
 * @file
 * @brief Concrete type definition for dictionary columns.
 */

namespace cudf {
/**
 * @addtogroup dictionary_classes
 * @{
 * @file
 */

/**
 * @brief A strongly typed wrapper for indices in a DICTIONARY type column.
 *
 * IndexType will be integer types like int32_t.
 *
 * For example, `dictionary32` is a strongly typed wrapper around an `int32_t`
 * value that holds the offset into the dictionary keys for a specific element.
 *
 * This wrapper provides common conversion and comparison operations for
 * the IndexType.
 */
template <typename IndexType>
struct dictionary_wrapper {
  using value_type = IndexType;  ///< The underlying type of the dictionary

  dictionary_wrapper()                          = default;
  ~dictionary_wrapper()                         = default;
  dictionary_wrapper(dictionary_wrapper&&)      = default;  ///< Move constructor
  dictionary_wrapper(dictionary_wrapper const&) = default;  ///< Copy constructor

  /**
   * @brief Move assignment operator
   *
   * @return The reference to this dictionary wrapper object
   */
  dictionary_wrapper& operator=(dictionary_wrapper&&) = default;

  /**
   * @brief Copy assignment operator
   *
   * @return The reference to this dictionary wrapper object
   */
  dictionary_wrapper& operator=(const dictionary_wrapper&) = default;

  /**
   * @brief Construct dictionary_wrapper from a value
   *
   * @param v The value to construct the dictionary_wrapper from
   */
  CUDF_HOST_DEVICE inline constexpr explicit dictionary_wrapper(value_type v) : _value{v} {}

  /**
   * @brief Conversion operator
   *
   * @return The value of this dictionary wrapper
   */
  CUDF_HOST_DEVICE inline explicit operator value_type() const { return _value; }

  /**
   * @brief Simple accessor
   *
   * @return The value of this dictionary wrapper
   */
  CUDF_HOST_DEVICE inline value_type value() const { return _value; }

  /**
   * @brief Returns the maximum value of the value type.
   *
   * @return The maximum value of the value type
   */
  static CUDF_HOST_DEVICE inline constexpr value_type max_value()
  {
    return std::numeric_limits<value_type>::max();
  }

  /**
   * @brief Returns the minimum value of the value type.
   *
   * @return The minimum value of the value type
   */
  static CUDF_HOST_DEVICE inline constexpr value_type min_value()
  {
    return std::numeric_limits<value_type>::min();
  }

  /**
   * @brief Returns the lowest value of the value type.
   *
   * @return The lowest value of the value type
   */
  static CUDF_HOST_DEVICE inline constexpr value_type lowest_value()
  {
    return std::numeric_limits<value_type>::lowest();
  }

 private:
  value_type _value;
};

// comparison operators
/**
 * @brief Wqual to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator==(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() == rhs.value();
}

/**
 * @brief Not equal to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is not equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator!=(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() != rhs.value();
}

/**
 * @brief Less than or equal to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is less than or equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator<=(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() <= rhs.value();
}

/**
 * @brief Greater than or equal to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is greater than or equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator>=(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() >= rhs.value();
}

/**
 * @brief Less than operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is less than rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline constexpr bool operator<(dictionary_wrapper<Integer> const& lhs,
                                                 dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() < rhs.value();
}

/**
 * @brief Greater than operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is greater than rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator>(dictionary_wrapper<Integer> const& lhs,
                                       dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() > rhs.value();
}

using dictionary32 = dictionary_wrapper<int32_t>;  ///< 32-bit integer indexed dictionary wrapper

/** @} */  // end of group
}  // namespace cudf
