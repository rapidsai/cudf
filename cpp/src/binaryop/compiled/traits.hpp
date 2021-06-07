/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <type_traits>

namespace cudf {

// has common type
template <typename AlwaysVoid, typename... Ts>
struct has_common_type_impl : std::false_type {
};

template <typename... Ts>
struct has_common_type_impl<std::void_t<std::common_type_t<Ts...>>, Ts...> : std::true_type {
};

template <typename... Ts>
using has_common_type = typename has_common_type_impl<void, Ts...>::type;

template <typename... Ts>
constexpr inline bool has_common_type_v = has_common_type_impl<void, Ts...>::value;

namespace binops::compiled {

template <typename BinaryOperator>
struct is_binary_operation_supported {
  // For types where Out type is fixed. (eg. comparison types)
  template <typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()(void)
  {
    if constexpr (column_device_view::has_element_accessor<TypeLhs>() and
                  column_device_view::has_element_accessor<TypeRhs>()) {
      if constexpr (has_common_type_v<TypeLhs, TypeRhs>) {
        using common_t = std::common_type_t<TypeLhs, TypeRhs>;
        return std::is_invocable_v<BinaryOperator, common_t, common_t>;
      } else
        return std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>;
    } else {
      return false;
    }
  }
  template <typename TypeOut, typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()(void)
  {
    if constexpr (column_device_view::has_element_accessor<TypeLhs>() and
                  column_device_view::has_element_accessor<TypeRhs>() and
                  (mutable_column_device_view::has_element_accessor<TypeOut>() or
                   is_fixed_point<TypeOut>())) {
      if constexpr (has_common_type_v<TypeLhs, TypeRhs>) {
        using common_t = std::common_type_t<TypeLhs, TypeRhs>;
        if constexpr (std::is_invocable_v<BinaryOperator, common_t, common_t>) {
          using ReturnType = std::invoke_result_t<BinaryOperator, common_t, common_t>;
          return std::is_constructible_v<TypeOut, ReturnType>;
        }
      } else {
        if constexpr (std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>) {
          using ReturnType = std::invoke_result_t<BinaryOperator, TypeLhs, TypeRhs>;
          return std::is_constructible_v<TypeOut, ReturnType>;
        }
      }
    }
    return false;
  }
};

}  // namespace binops::compiled
}  // namespace cudf
