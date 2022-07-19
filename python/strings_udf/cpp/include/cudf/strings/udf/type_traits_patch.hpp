/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

// Patch for the <type_traits> that is setup by jitify.hpp.
// The source below supports the std::underlying_type_t which is
// missing from the jitify.hpp implementation of <type_traits>

#ifdef CUDF_JIT_UDF

namespace std {
/// integral_constant
// the 'udf' prefix prevents collision with jitify.hpp definition
// which is incompatible with how is-enum and underlying_type needs
template <typename _Tp, _Tp __v>
struct udf_integral_constant {
  static constexpr _Tp value = __v;
  typedef _Tp value_type;
  typedef udf_integral_constant<_Tp, __v> type;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};
template <typename _Tp, _Tp __v>
constexpr _Tp udf_integral_constant<_Tp, __v>::value;

/// is_enum
template <typename _Tp>
struct is_enum : public udf_integral_constant<bool, __is_enum(_Tp)> {
};

template <typename _Tp, bool = is_enum<_Tp>::value>
struct __underlying_type_impl {
  using type = __underlying_type(_Tp);
};

template <typename _Tp>
struct __underlying_type_impl<_Tp, false> {
};

/// The underlying type of an enum.
template <typename _Tp>
struct underlying_type : public __underlying_type_impl<_Tp> {
};

/// Alias template for underlying_type
template <typename _Tp>
using underlying_type_t = typename underlying_type<_Tp>::type;
}  // namespace std

#endif