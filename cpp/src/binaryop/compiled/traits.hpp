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
//#include <cuda/std/type_traits>

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

template <typename T1, typename BinaryFunctor>
constexpr bool is_op_supported(void)
{
  return std::is_invocable_v<BinaryFunctor, T1>;
}

template <typename T1, typename T2, typename BinaryFunctor>
constexpr bool is_op_supported(void)
{
  return std::is_invocable_v<BinaryFunctor, T1, T2>;
}

// has ::operate

template <typename T, typename T1, typename T2, typename T3, typename = T>
struct has_operate : std::false_type {
};
template <typename T, typename T1, typename T2, typename T3>
struct has_operate<
  T,
  T1,
  T2,
  T3,
  std::enable_if_t<std::is_same<decltype(T::template operate<T1, T2, T3>), T1(T2, T3)>::value, T>>
  : std::true_type {
};

template <class T, typename T1, typename T2, typename T3>
constexpr inline bool has_operate_v = has_operate<T, T1, T2, T3>::value;

}  // namespace binops::compiled
}  // namespace cudf
