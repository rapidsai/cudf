/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/utils/type_dispatcher.hpp>

#include <type_traits>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Indicates if the type `T` is a numeric type.
 *
 * @tparam T  The type to verify
 * @return true `T` is numeric
 * @return false  `T` is not numeric
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_numeric() {
  return std::is_integral<T>::value or std::is_floating_point<T>::value;
}

namespace detail {
struct numeric {
  template <typename T>
  constexpr bool operator()() const noexcept {
    return is_numeric<T>();
  }
};
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Indicates if `type` is a numeric `data_type`.
 *
 * "Numeric" types are fundamental integral/floating point types such as `INT*`
 * or `FLOAT*`. Types that wrap a numeric type are not considered numeric, e.g.,
 *`TIMESTAMP` or `DATE32`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is numeric
 * @return false `type` is not numeric
 *---------------------------------------------------------------------------**/
constexpr inline bool is_numeric(data_type type) {
  return cudf::exp::type_dispatcher(type, detail::numeric{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates whether the concerete C++ type `T` corresponds to a
 * fixed-width element type.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @tparam T The C++ type to verify
 * @return true `T` corresponds to a fixed-width element type
 * @return false `T` corresponds to a variable-width element type
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_fixed_width() {
  // TODO Add fixed width wrapper types
  // Is a category fixed width?
  return std::is_integral<T>::value or std::is_floating_point<T>::value;
}

namespace detail {
struct fixed_width {
  template <typename T>
  constexpr bool operator()() const noexcept {
    return is_fixed_width<T>();
  }
};
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Indicates if elements of the specified `data_type` have a fixed-width.
 *
 * @param type The `data_type` to verify
 * @return true `type` is fixed-width
 * @return false  `type` is variable-width
 *---------------------------------------------------------------------------**/
constexpr inline bool is_fixed_width(data_type type) {
  return cudf::exp::type_dispatcher(type, detail::fixed_width{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates if the concrete C++ type `T` corresponds to a compound
 * element type.
 *
 * `column`s with "compound" elements are logically a single column of elements,
 * but may be concretely implemented with two or more `column`s. For example, a
 * `STRING` column could contain a `column` of offsets and a child `column` of
 * characters.
 *
 * @tparam T The C++ type to verify
 * @return true `T` corresponds to a "compound" type
 * @return false `T` corresponds to a "simple" type
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_compound() {
  // TODO Implement with checks for the compound wrapper types
  return false;
}

/**---------------------------------------------------------------------------*
 * @brief Indicates if the concrete C++ type `T` corresponds to a "simple"
 * element type.
 *
 * "Simple" element types are implemented with only a single column, i.e.,
 * `num_children() == 0` for columns of "simple" elements
 *
 * @tparam T The C++ type to verify
 * @return true `T` corresponds to a simple type
 * @return false `T` corresponds to a compound type
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_simple() {
  return not is_compound<T>();
}

namespace detail {
struct compound {
  template <typename T>
  constexpr bool operator()() const noexcept {
    return is_compound<T>();
  }
};
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Indicates if elements of the sepcified `data_type` are a "compound"
 * type.
 *
 * `column`s with "compound" elements are logically a single column of elements,
 * but may be concretely implemented with two or more `column`s. For example, a
 * `STRING` column could contain a `column` of offsets and a child `column` of
 * characters.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a "compound" type
 * @return false `type` is a "simple" type
 *---------------------------------------------------------------------------**/
constexpr inline bool is_compound(data_type type) {
  return cudf::exp::type_dispatcher(type, detail::compound{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates if elements of the specified `data_type` are a "simple"
 *type.
 *
 * "Simple" element types are implemented with only a single column, i.e.,
 * `num_children() == 0` for columns of "simple" elements
 *
 * @param type The `data_type` to verify
 * @return true `type` is a simple type
 * @return false `type` is a compound type
 *---------------------------------------------------------------------------**/
constexpr inline bool is_simple(data_type type) {
  return not is_compound(type);
}

}  // namespace cudf