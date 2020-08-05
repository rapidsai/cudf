/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

/**
 * @file dictionary.hpp
 * @brief Concrete type definition for dictionary columns.
 */

namespace cudf {
/**
 * @addtogroup dictionary_classes
 * @{
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
  using value_type = IndexType;

  dictionary_wrapper()                            = default;
  ~dictionary_wrapper()                           = default;
  dictionary_wrapper(dictionary_wrapper&&)        = default;
  dictionary_wrapper(dictionary_wrapper const& v) = default;
  dictionary_wrapper& operator=(dictionary_wrapper&&) = default;
  dictionary_wrapper& operator=(const dictionary_wrapper&) = default;

  // construct object from type
  CUDA_HOST_DEVICE_CALLABLE constexpr explicit dictionary_wrapper(value_type v) : _value{v} {}

  // conversion operator
  CUDA_HOST_DEVICE_CALLABLE explicit operator value_type() const { return _value; }
  // simple accessor
  CUDA_HOST_DEVICE_CALLABLE value_type value() const { return _value; }

 private:
  value_type _value;
};

// comparison operators
template <typename Integer>
CUDA_HOST_DEVICE_CALLABLE bool operator==(dictionary_wrapper<Integer> const& lhs,
                                          dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() == rhs.value();
}

template <typename Integer>
CUDA_HOST_DEVICE_CALLABLE bool operator!=(dictionary_wrapper<Integer> const& lhs,
                                          dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() != rhs.value();
}

template <typename Integer>
CUDA_HOST_DEVICE_CALLABLE bool operator<=(dictionary_wrapper<Integer> const& lhs,
                                          dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() <= rhs.value();
}

template <typename Integer>
CUDA_HOST_DEVICE_CALLABLE bool operator>=(dictionary_wrapper<Integer> const& lhs,
                                          dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() >= rhs.value();
}

template <typename Integer>
CUDA_HOST_DEVICE_CALLABLE constexpr bool operator<(dictionary_wrapper<Integer> const& lhs,
                                                   dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() < rhs.value();
}

template <typename Integer>
CUDA_HOST_DEVICE_CALLABLE bool operator>(dictionary_wrapper<Integer> const& lhs,
                                         dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() > rhs.value();
}

using dictionary32 = dictionary_wrapper<int32_t>;

/** @} */  // end of group
}  // namespace cudf
