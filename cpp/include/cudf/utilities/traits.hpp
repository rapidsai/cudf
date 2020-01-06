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
#include <cudf/wrappers/timestamps.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/strings/string_view.cuh>

#include <type_traits>

namespace cudf {

template <typename...>
using void_t = void;

template <typename L, typename R, typename = void>
struct is_relationally_comparable_impl : std::false_type {};

template <typename L, typename R>
using less_comparable = decltype(std::declval<L>() < std::declval<R>());

template <typename L, typename R>
using greater_comparable = decltype(std::declval<L>() > std::declval<R>());

template <typename L, typename R>
struct is_relationally_comparable_impl<
    L, R, void_t<less_comparable<L, R>, greater_comparable<L, R> > >
    : std::true_type {};

template <typename T>
using is_timestamp_t = simt::std::disjunction<
  std::is_same<cudf::timestamp_D, T>,
  std::is_same<cudf::timestamp_s, T>,
  std::is_same<cudf::timestamp_ms, T>,
  std::is_same<cudf::timestamp_us, T>,
  std::is_same<cudf::timestamp_ns, T>
>;

/**---------------------------------------------------------------------------*
 * @brief Indicates whether objects of types `L` and `R` can be relationally
 *compared.
 *
 * Given two objects `L l`, and `R r`, returns true if `l < r` and `l > r` are
 * well-formed expressions.
 *
 * @tparam L Type of the first object
 * @tparam R Type of the second object
 * @return true Objects of types `L` and `R` can be relationally be compared
 * @return false Objects of types `L` and `R` cannot be compared
 *---------------------------------------------------------------------------**/
template <typename L, typename R>
constexpr inline bool is_relationally_comparable() {
  return is_relationally_comparable_impl<L, R>::value;
}

/**---------------------------------------------------------------------------*
 * @brief Indicates whether the type `T` is a numeric type.
 *
 * @tparam T  The type to verify
 * @return true `T` is numeric
 * @return false  `T` is not numeric
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_numeric() {
  return std::is_integral<T>::value or std::is_floating_point<T>::value;
}

struct is_numeric_impl {
  template <typename T>
  bool operator()() {
    return is_numeric<T>();
  }
};

/**---------------------------------------------------------------------------*
 * @brief Indicates whether `type` is a numeric `data_type`.
 *
 * "Numeric" types are fundamental integral/floating point types such as `INT*`
 * or `FLOAT*`. Types that wrap a numeric type are not considered numeric, e.g.,
 *`TIMESTAMP`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is numeric
 * @return false `type` is not numeric
 *---------------------------------------------------------------------------**/
constexpr inline bool is_numeric(data_type type) {
  return cudf::experimental::type_dispatcher(type, is_numeric_impl{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates whether `T` is a Boolean type.
 *
 * `cudf::bool8` is cudf's Boolean type.
 *
 * @param type The `data_type` to verify
 * @return true `type` is Boolean
 * @return false `type` is not Boolean
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_boolean() {
  return std::is_same<T, cudf::experimental::bool8>::value ||
         std::is_same<T, bool>::value;
}

struct is_boolean_impl {
  template <typename T>
  bool operator()() {
    return is_boolean<T>();
  }
};

/**---------------------------------------------------------------------------*
 * @brief Indicates whether `type` is a Boolean `data_type`.
 *
 * `cudf::bool8` is cudf's Boolean type.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a Boolean
 * @return false `type` is not a Boolean
 *---------------------------------------------------------------------------**/
constexpr inline bool is_boolean(data_type type) {
  return cudf::experimental::type_dispatcher(type, is_boolean_impl{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates whether the type `T` is a timestamp type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a timestamp
 * @return false  `T` is not a timestamp
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_timestamp() {
  return is_timestamp_t<T>::value;
}

struct is_timestamp_impl {
  template <typename T>
  bool operator()() {
    return is_timestamp<T>();
  }
};

/**---------------------------------------------------------------------------*
 * @brief Indicates whether `type` is a timestamp `data_type`.
 *
 * "Timestamp" types are int32_t or int64_t durations since the unix epoch.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a timestamp
 * @return false `type` is not a timestamp
 *---------------------------------------------------------------------------**/
constexpr inline bool is_timestamp(data_type type) {
  return cudf::experimental::type_dispatcher(type, is_timestamp_impl{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates whether elements of type `T` are fixed-width.
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
  return cudf::is_numeric<T>() || cudf::is_timestamp<T>();
}

struct is_fixed_width_impl {
  template <typename T>
  bool operator()() {
    return is_fixed_width<T>();
  }
};

/**---------------------------------------------------------------------------*
 * @brief Indicates whether elements of `type` are fixed-width.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @param type The `data_type` to verify
 * @return true `type` is fixed-width
 * @return false  `type` is variable-width
 *---------------------------------------------------------------------------**/
constexpr inline bool is_fixed_width(data_type type) {
  return cudf::experimental::type_dispatcher(type, is_fixed_width_impl{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates whether the type `T` is a compound type.
 *
 * `column`s with "compound" elements are logically a single column of elements,
 * but may be concretely implemented with two or more `column`s. For example, a
 * `STRING` column could contain a `column` of offsets and a child `column` of
 * characters.
 *
 * @tparam T The type to verify
 * @return true `T` corresponds to a "compound" type
 * @return false `T` corresponds to a "simple" type
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_compound() {
  return std::is_same<T, cudf::string_view>::value;
}

struct is_compound_impl {
  template <typename T>
  bool operator()() {
    return is_compound<T>();
  }
};

/**---------------------------------------------------------------------------*
 * @brief Indicates whether elements of `type` are compound.
 *
 * `column`s with "compound" elements are logically a single column of elements,
 * but may be concretely implemented with two or more `column`s. For example, a
 * `STRING` column could contain a `column` of offsets and a child `column` of
 * characters.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a compound type
 * @return false `type` is a simple type
 *---------------------------------------------------------------------------**/
constexpr inline bool is_compound(data_type type) {
  return cudf::experimental::type_dispatcher(type, is_compound_impl{});
}

/**---------------------------------------------------------------------------*
 * @brief Indicates whether the type `T` is a simple type.
 *
 * "Simple" element types are implemented with only a single column, i.e.,
 * `num_children() == 0` for columns of "simple" elements
 *
 * @tparam T The type to verify
 * @return true `T` corresponds to a simple type
 * @return false `T` corresponds to a compound type
 *---------------------------------------------------------------------------**/
template <typename T>
constexpr inline bool is_simple() {
  return not is_compound<T>();
}

struct is_simple_impl {
  template <typename T>
  bool operator()() {
    return is_simple<T>();
  }
};

/**---------------------------------------------------------------------------*
 * @brief Indicates whether elements of `type` are simple.
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
