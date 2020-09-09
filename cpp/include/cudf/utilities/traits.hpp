/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/lists/list_view.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <simt/type_traits>
#include "cudf/structs/struct_view.hpp"

namespace cudf {

/**
 * @addtogroup utility_types
 * @{
 */

template <typename...>
using void_t = void;

template <typename L, typename R, typename = void>
struct is_relationally_comparable_impl : std::false_type {
};

template <typename L, typename R>
using less_comparable = decltype(std::declval<L>() < std::declval<R>());

template <typename L, typename R>
using greater_comparable = decltype(std::declval<L>() > std::declval<R>());

template <typename L, typename R>
struct is_relationally_comparable_impl<L,
                                       R,
                                       void_t<less_comparable<L, R>, greater_comparable<L, R>>>
  : std::true_type {
};

template <typename L, typename R, typename = void>
struct is_equality_comparable_impl : std::false_type {
};

template <typename L, typename R>
using equality_comparable = decltype(std::declval<L>() == std::declval<R>());

template <typename L, typename R>
struct is_equality_comparable_impl<L, R, void_t<equality_comparable<L, R>>> : std::true_type {
};

template <typename T>
using is_timestamp_t = simt::std::disjunction<std::is_same<cudf::timestamp_D, T>,
                                              std::is_same<cudf::timestamp_s, T>,
                                              std::is_same<cudf::timestamp_ms, T>,
                                              std::is_same<cudf::timestamp_us, T>,
                                              std::is_same<cudf::timestamp_ns, T>>;

template <typename T>
using is_duration_t = simt::std::disjunction<std::is_same<cudf::duration_D, T>,
                                             std::is_same<cudf::duration_s, T>,
                                             std::is_same<cudf::duration_ms, T>,
                                             std::is_same<cudf::duration_us, T>,
                                             std::is_same<cudf::duration_ns, T>>;

/**
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
 **/
template <typename L, typename R>
constexpr inline bool is_relationally_comparable()
{
  return is_relationally_comparable_impl<L, R>::value;
}

/**
 * @brief Indicates whether objects of types `L` and `R` can be compared
 * for equality.
 *
 * Given two objects `L l`, and `R r`, returns true if `l == r` is a
 * well-formed expression.
 *
 * @tparam L Type of the first object
 * @tparam R Type of the second object
 * @return true Objects of types `L` and `R` can be compared for equality
 * @return false Objects of types `L` and `R` cannot be compared
 */
template <typename L, typename R>
constexpr inline bool is_equality_comparable()
{
  return is_equality_comparable_impl<L, R>::value;
}

/**
 * @brief Indicates whether the type `T` is a numeric type.
 *
 * @tparam T  The type to verify
 * @return true `T` is numeric
 * @return false  `T` is not numeric
 **/
template <typename T>
constexpr inline bool is_numeric()
{
  return std::is_integral<T>::value or std::is_floating_point<T>::value;
}

struct is_numeric_impl {
  template <typename T>
  bool operator()()
  {
    return is_numeric<T>();
  }
};

/**
 * @brief Indicates whether the type `T` is a unsigned numeric type.
 *
 * @tparam T  The type to verify
 * @return true `T` is unsigned numeric
 * @return false  `T` is signed numeric
 **/
template <typename T>
constexpr inline bool is_unsigned()
{
  return std::is_unsigned<T>::value;
}

struct is_unsigned_impl {
  template <typename T>
  bool operator()()
  {
    return is_unsigned<T>();
  }
};

/**
 * @brief Indicates whether `type` is a numeric `data_type`.
 *
 * "Numeric" types are fundamental integral/floating point types such as `INT*`
 * or `FLOAT*`. Types that wrap a numeric type are not considered numeric, e.g.,
 *`TIMESTAMP`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is numeric
 * @return false `type` is not numeric
 **/
constexpr inline bool is_numeric(data_type type)
{
  return cudf::type_dispatcher(type, is_numeric_impl{});
}

/**
 * @brief Indicates whether the type `T` is a index type.
 *
 * A type `T` is considered an index type if it is valid to use
 * elements of type `T` to index into a column. I.e.,
 * index types are integral types such as 'INT*' apart from 'bool'.
 *
 * @tparam T  The type to verify
 * @return true `T` is index type
 * @return false  `T` is not index type
 **/
template <typename T>
constexpr inline bool is_index_type()
{
  return std::is_integral<T>::value and not std::is_same<T, bool>::value;
}

struct is_index_type_impl {
  template <typename T>
  bool operator()()
  {
    return is_index_type<T>();
  }
};

/**
 * @brief Indicates whether the type `type` is a index type.
 *
 * A type `T` is considered an index type if it is valid to use
 * elements of type `T` to index into a column. I.e.,
 * index types are integral types such as 'INT*' apart from 'bool'.
 *
 * @param type The `data_type` to verify
 * @return true `type` is index type
 * @return false `type` is not index type
 **/
constexpr inline bool is_index_type(data_type type)
{
  return cudf::type_dispatcher(type, is_index_type_impl{});
}

/**
 * @brief Indicates whether `type` is a unsigned numeric `data_type`.
 *
 * "Unsigned Numeric" types are fundamental integral types such as `UINT*`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is unsigned numeric
 * @return false `type` is signed numeric
 **/
constexpr inline bool is_unsigned(data_type type)
{
  return cudf::type_dispatcher(type, is_unsigned_impl{});
}

/**
 * @brief Indicates whether the type `T` is a floating point type.
 *
 * @tparam T  The type to verify
 * @return true `T` is floating point
 * @return false  `T` is not floating point
 **/
template <typename T>
constexpr inline bool is_floating_point()
{
  return std::is_floating_point<T>::value;
}

struct is_floating_point_impl {
  template <typename T>
  bool operator()()
  {
    return is_floating_point<T>();
  }
};

/**
 * @brief Indicates whether `type` is a floating point `data_type`.
 *
 * "Floating point" types are fundamental floating point types such as `FLOAT*`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is floating point
 * @return false `type` is not floating point
 **/
constexpr inline bool is_floating_point(data_type type)
{
  return cudf::type_dispatcher(type, is_floating_point_impl{});
}

/**
 * @brief Indicates whether `T` is a Boolean type.
 *
 * @param type The `data_type` to verify
 * @return true `type` is Boolean
 * @return false `type` is not Boolean
 **/
template <typename T>
constexpr inline bool is_boolean()
{
  return std::is_same<T, bool>::value;
}

struct is_boolean_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_boolean<T>();
  }
};

/**
 * @brief Indicates whether `type` is a Boolean `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a Boolean
 * @return false `type` is not a Boolean
 **/
constexpr inline bool is_boolean(data_type type)
{
  return cudf::type_dispatcher(type, is_boolean_impl{});
}

/**
 * @brief Indicates whether the type `T` is a timestamp type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a timestamp
 * @return false  `T` is not a timestamp
 **/
template <typename T>
constexpr inline bool is_timestamp()
{
  return is_timestamp_t<T>::value;
}

struct is_timestamp_impl {
  template <typename T>
  bool operator()()
  {
    return is_timestamp<T>();
  }
};

/**
 * @brief Indicates whether `type` is a timestamp `data_type`.
 *
 * "Timestamp" types are int32_t or int64_t durations since the unix epoch.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a timestamp
 * @return false `type` is not a timestamp
 **/
constexpr inline bool is_timestamp(data_type type)
{
  return cudf::type_dispatcher(type, is_timestamp_impl{});
}

/**
 * @brief Indicates whether the type `T` is a fixed-point type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a fixed-point type
 * @return false  `T` is not a fixed-point type
 **/
template <typename T>
constexpr inline bool is_fixed_point()
{
  return std::is_same<numeric::decimal32, T>::value || std::is_same<numeric::decimal64, T>::value;
}

struct is_fixed_point_impl {
  template <typename T>
  bool operator()()
  {
    return is_fixed_point<T>();
  }
};

/**
 * @brief Indicates whether `type` is a fixed point `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a fixed point type
 * @return false `type` is not a fixed point type
 **/
constexpr inline bool is_fixed_point(data_type type)
{
  return cudf::type_dispatcher(type, is_fixed_point_impl{});
}

/**
 * @brief Indicates whether the type `T` is a duration type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a duration
 * @return false  `T` is not a duration
 **/
template <typename T>
constexpr inline bool is_duration()
{
  return is_duration_t<T>::value;
}

struct is_duration_impl {
  template <typename T>
  bool operator()()
  {
    return is_duration<T>();
  }
};

/**
 * @brief Indicates whether `type` is a duration `data_type`.
 *
 * "Duration" types are int32_t or int64_t tick counts representing a time interval.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a duration
 * @return false `type` is not a duration
 **/
constexpr inline bool is_duration(data_type type)
{
  return cudf::type_dispatcher(type, is_duration_impl{});
}

/**
 * @brief Indicates whether the type `T` is a chrono type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a duration or a timestamp type
 * @return false  `T` is neither a duration nor a timestamp type
 **/
template <typename T>
constexpr inline bool is_chrono()
{
  return is_duration<T>() || is_timestamp<T>();
}

struct is_chrono_impl {
  template <typename T>
  bool operator()()
  {
    return is_chrono<T>();
  }
};

/**
 * @brief Indicates whether `type` is a chrono `data_type`.
 *
 * Chrono types include cudf timestamp types, which represent a point in time, and cudf
 * duration types that represent a time interval.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a chrono type
 * @return false `type` is not a chrono type
 **/
constexpr inline bool is_chrono(data_type type)
{
  return cudf::type_dispatcher(type, is_chrono_impl{});
}

/**
 * @brief Indicates whether elements of type `T` are fixed-width.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @tparam T The C++ type to verify
 * @return true `T` corresponds to a fixed-width element type
 * @return false `T` corresponds to a variable-width element type
 **/
template <typename T>
constexpr inline bool is_fixed_width()
{
  // TODO Add fixed width wrapper types
  // Is a category fixed width?
  return cudf::is_numeric<T>() || cudf::is_chrono<T>() || cudf::is_fixed_point<T>();
}

struct is_fixed_width_impl {
  template <typename T>
  bool operator()()
  {
    return is_fixed_width<T>();
  }
};

/**
 * @brief Indicates whether elements of `type` are fixed-width.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @param type The `data_type` to verify
 * @return true `type` is fixed-width
 * @return false  `type` is variable-width
 **/
constexpr inline bool is_fixed_width(data_type type)
{
  return cudf::type_dispatcher(type, is_fixed_width_impl{});
}

/**
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
 */
template <typename T>
constexpr inline bool is_compound()
{
  return std::is_same<T, cudf::string_view>::value or std::is_same<T, cudf::dictionary32>::value or
         std::is_same<T, cudf::list_view>::value or std::is_same<T, cudf::struct_view>::value;
}

struct is_compound_impl {
  template <typename T>
  bool operator()()
  {
    return is_compound<T>();
  }
};

/**
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
 **/
constexpr inline bool is_compound(data_type type)
{
  return cudf::type_dispatcher(type, is_compound_impl{});
}

/**
 * @brief Indicates whether `T` is a nested type.
 *
 * "Nested" types are distinct from compound types in that they
 * can have an arbitrarily deep list of descendants of the same
 * type. Strings are not a nested type, but lists are.
 *
 * @param T The type to verify
 * @return true T is a nested type
 * @return false T is not a nested type
 **/
template <typename T>
constexpr inline bool is_nested()
{
  return std::is_same<T, cudf::list_view>::value || std::is_same<T, cudf::struct_view>::value;
}

struct is_nested_impl {
  template <typename T>
  bool operator()()
  {
    return is_nested<T>();
  }
};

/**
 * @brief Indicates whether `type` is a nested type
 *
 * "Nested" types are distinct from compound types in that they
 * can have an arbitrarily deep list of descendants of the same
 * type. Strings are not a nested type, but lists are.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a nested type
 * @return false `type` is not a nested type
 **/
constexpr inline bool is_nested(data_type type)
{
  return cudf::type_dispatcher(type, is_nested_impl{});
}

template <typename FromType, typename ToType>
struct is_logically_castable_impl : std::false_type {
};

// Allow cast to same type
template <typename Type>
struct is_logically_castable_impl<Type, Type> : std::true_type {
};

#ifndef MAP_CASTABLE_TYPES
#define MAP_CASTABLE_TYPES(Type1, Type2)                             \
  template <>                                                        \
  struct is_logically_castable_impl<Type1, Type2> : std::true_type { \
  };                                                                 \
  template <>                                                        \
  struct is_logically_castable_impl<Type2, Type1> : std::true_type { \
  };
#endif

// Allow cast between timestamp and integer representation
MAP_CASTABLE_TYPES(cudf::timestamp_D, cudf::timestamp_D::duration::rep);
MAP_CASTABLE_TYPES(cudf::timestamp_s, cudf::timestamp_s::duration::rep);
MAP_CASTABLE_TYPES(cudf::timestamp_ms, cudf::timestamp_ms::duration::rep);
MAP_CASTABLE_TYPES(cudf::timestamp_us, cudf::timestamp_us::duration::rep);
MAP_CASTABLE_TYPES(cudf::timestamp_ns, cudf::timestamp_ns::duration::rep);
// Allow cast between durations and integer representation
MAP_CASTABLE_TYPES(cudf::duration_D, cudf::duration_D::rep);
MAP_CASTABLE_TYPES(cudf::duration_s, cudf::duration_s::rep);
MAP_CASTABLE_TYPES(cudf::duration_ms, cudf::duration_ms::rep);
MAP_CASTABLE_TYPES(cudf::duration_us, cudf::duration_us::rep);
MAP_CASTABLE_TYPES(cudf::duration_ns, cudf::duration_ns::rep);

template <typename FromType>
struct is_logically_castable_to_impl {
  template <typename ToType>
  constexpr bool operator()()
  {
    return is_logically_castable_impl<FromType, ToType>::value;
  }
};

struct is_logically_castable_from_impl {
  template <typename FromType>
  constexpr bool operator()(data_type to)
  {
    return type_dispatcher(to, is_logically_castable_to_impl<FromType>{});
  }
};

/**
 * @brief Indicates whether `from` is logically castable to `to`.
 *
 * Data types that have the same size and underlying representation, e.g. INT32 and TIMESTAMP_DAYS
 * which are both represented as 32-bit integers in memory, are eligible for logical cast.
 *
 * See `cudf::logical_cast()` which returns a zero-copy `column_view` when casting between
 * logically castable types.
 *
 * @param from The `data_type` to convert from
 * @param to The `data_type` to convert to
 * @return `true` if the types are logically castable
 */
constexpr bool is_logically_castable(data_type from, data_type to)
{
  return type_dispatcher(from, is_logically_castable_from_impl{}, to);
}

/** @} */

template <typename From, typename To>
struct is_convertible : std::is_convertible<From, To> {
};

// This will ensure that timestamps can be promoted to a higher precision. Presently, they can't
// do that due to nvcc/gcc compiler issues
template <typename Duration1, typename Duration2>
struct is_convertible<cudf::detail::timestamp<Duration1>, cudf::detail::timestamp<Duration2>>
  : std::is_convertible<typename cudf::detail::time_point<Duration1>::duration,
                        typename cudf::detail::time_point<Duration2>::duration> {
};

}  // namespace cudf
