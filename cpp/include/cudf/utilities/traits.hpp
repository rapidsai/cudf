/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/dictionary.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/type_traits>

namespace cudf {

/**
 * @addtogroup utility_types
 * @{
 * @file
 */

/// Utility metafunction that maps a sequence of any types to the type void.
template <typename...>
using void_t = void;

/**
 * @brief Convenience macro for SFINAE as an unnamed template parameter.
 *
 * Example:
 * \code{cpp}
 * // This function will participate in overload resolution only if T is an integral type
 * template <typename T, CUDF_ENABLE_IF(std::is_integral_v<T> )>
 * void foo();
 * \endcode
 *
 */
#define CUDF_ENABLE_IF(...) std::enable_if_t<(__VA_ARGS__)>* = nullptr

/// Checks if two types are comparable using less operator (i.e. <).
template <typename L, typename R>
using less_comparable = decltype(std::declval<L>() < std::declval<R>());

/// Checks if two types are comparable using greater operator (i.e. >).
template <typename L, typename R>
using greater_comparable = decltype(std::declval<L>() > std::declval<R>());

/// Checks if two types are comparable using equality operator (i.e. ==).
template <typename L, typename R>
using equality_comparable = decltype(std::declval<L>() == std::declval<R>());

namespace detail {
template <typename L, typename R, typename = void>
struct is_relationally_comparable_impl : std::false_type {
};

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
struct is_equality_comparable_impl<L, R, void_t<equality_comparable<L, R>>> : std::true_type {
};

// has common type
template <typename AlwaysVoid, typename... Ts>
struct has_common_type_impl : std::false_type {
};

template <typename... Ts>
struct has_common_type_impl<void_t<std::common_type_t<Ts...>>, Ts...> : std::true_type {
};
}  // namespace detail

/// Checks if types have a common type
template <typename... Ts>
using has_common_type = typename detail::has_common_type_impl<void, Ts...>::type;

/// Helper variable template for has_common_type<>::value
template <typename... Ts>
constexpr inline bool has_common_type_v = detail::has_common_type_impl<void, Ts...>::value;

/// Checks if a type is a timestamp type.
template <typename T>
using is_timestamp_t = cuda::std::disjunction<std::is_same<cudf::timestamp_D, T>,
                                              std::is_same<cudf::timestamp_s, T>,
                                              std::is_same<cudf::timestamp_ms, T>,
                                              std::is_same<cudf::timestamp_us, T>,
                                              std::is_same<cudf::timestamp_ns, T>>;

/// Checks if a type is a duration type.
template <typename T>
using is_duration_t = cuda::std::disjunction<std::is_same<cudf::duration_D, T>,
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
 */
template <typename L, typename R>
constexpr inline bool is_relationally_comparable()
{
  return detail::is_relationally_comparable_impl<L, R>::value;
}

/**
 * @brief Checks whether `data_type` `type` supports relational comparisons.
 *
 * @param type Data_type for comparison.
 * @return true If `type` supports relational comparisons.
 * @return false If `type` does not support relational comparisons.
 */
bool is_relationally_comparable(data_type type);

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
  return detail::is_equality_comparable_impl<L, R>::value;
}

/**
 * @brief Checks whether `data_type` `type` supports equality comparisons.
 *
 * @param type Data_type for comparison.
 * @return true If `type` supports equality comparisons.
 * @return false If `type` does not support equality comparisons.
 */
bool is_equality_comparable(data_type type);

/**
 * @brief Indicates whether the type `T` is a numeric type.
 *
 * @tparam T  The type to verify
 * @return true `T` is numeric
 * @return false  `T` is not numeric
 */
template <typename T>
constexpr inline bool is_numeric()
{
  return cuda::std::is_arithmetic<T>();
}

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
 */
bool is_numeric(data_type type);

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
 */
template <typename T>
constexpr inline bool is_index_type()
{
  return std::is_integral_v<T> and not std::is_same_v<T, bool>;
}

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
 */
bool is_index_type(data_type type);

/**
 * @brief Indicates whether the type `T` is a unsigned numeric type.
 *
 * @tparam T  The type to verify
 * @return true `T` is unsigned numeric
 * @return false  `T` is signed numeric
 */
template <typename T>
constexpr inline bool is_unsigned()
{
  return std::is_unsigned_v<T>;
}

/**
 * @brief Indicates whether `type` is a unsigned numeric `data_type`.
 *
 * "Unsigned Numeric" types are fundamental integral types such as `UINT*`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is unsigned numeric
 * @return false `type` is signed numeric
 */
bool is_unsigned(data_type type);

/**
 * @brief Indicates whether the `Iterator` value type is unsigned.
 *
 * @tparam Iterator  The type to verify
 * @return true if the iterator's value type is unsigned
 */
template <typename Iterator>
constexpr inline bool is_signed_iterator()
{
  return std::is_signed_v<typename std::iterator_traits<Iterator>::value_type>;
}

/**
 * @brief Indicates whether the type `T` is an integral type.
 *
 * @tparam T  The type to verify
 * @return true `T` is integral
 * @return false  `T` is not integral
 */
template <typename T>
constexpr inline bool is_integral()
{
  return cuda::std::is_integral_v<T>;
}

/**
 * @brief Indicates whether `type` is a integral `data_type`.
 *
 * "Integral" types are fundamental integer types such as `INT*` and `UINT*`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is integral
 * @return false `type` is integral
 */
bool is_integral(data_type type);

/**
 * @brief Indicates whether the type `T` is a floating point type.
 *
 * @tparam T  The type to verify
 * @return true `T` is floating point
 * @return false  `T` is not floating point
 */
template <typename T>
constexpr inline bool is_floating_point()
{
  return std::is_floating_point_v<T>;
}

/**
 * @brief Indicates whether `type` is a floating point `data_type`.
 *
 * "Floating point" types are fundamental floating point types such as `FLOAT*`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is floating point
 * @return false `type` is not floating point
 */
bool is_floating_point(data_type type);

/**
 * @brief Indicates whether `T` is a std::byte type.
 *
 * @tparam T The type to verify
 * @return true `type` is std::byte
 * @return false `type` is not std::byte
 */
template <typename T>
constexpr inline bool is_byte()
{
  return std::is_same_v<std::remove_cv_t<T>, std::byte>;
}

/**
 * @brief Indicates whether `T` is a Boolean type.
 *
 * @param type The `data_type` to verify
 * @return true `type` is Boolean
 * @return false `type` is not Boolean
 */
template <typename T>
constexpr inline bool is_boolean()
{
  return std::is_same_v<T, bool>;
}

/**
 * @brief Indicates whether `type` is a Boolean `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a Boolean
 * @return false `type` is not a Boolean
 */
bool is_boolean(data_type type);

/**
 * @brief Indicates whether the type `T` is a timestamp type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a timestamp
 * @return false  `T` is not a timestamp
 */
template <typename T>
constexpr inline bool is_timestamp()
{
  return is_timestamp_t<T>::value;
}

/**
 * @brief Indicates whether `type` is a timestamp `data_type`.
 *
 * "Timestamp" types are int32_t or int64_t durations since the unix epoch.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a timestamp
 * @return false `type` is not a timestamp
 */
bool is_timestamp(data_type type);

/**
 * @brief Indicates whether the type `T` is a fixed-point type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a fixed-point type
 * @return false  `T` is not a fixed-point type
 */
template <typename T>
constexpr inline bool is_fixed_point()
{
  return std::is_same_v<numeric::decimal32, T> || std::is_same_v<numeric::decimal64, T> ||
         std::is_same_v<numeric::decimal128, T>;
}

/**
 * @brief Indicates whether `type` is a fixed point `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a fixed point type
 * @return false `type` is not a fixed point type
 */
bool is_fixed_point(data_type type);

/**
 * @brief Indicates whether the type `T` is a duration type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a duration
 * @return false  `T` is not a duration
 */
template <typename T>
constexpr inline bool is_duration()
{
  return is_duration_t<T>::value;
}

/**
 * @brief Indicates whether `type` is a duration `data_type`.
 *
 * "Duration" types are int32_t or int64_t tick counts representing a time interval.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a duration
 * @return false `type` is not a duration
 */
bool is_duration(data_type type);

/**
 * @brief Indicates whether the type `T` is a chrono type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a duration or a timestamp type
 * @return false  `T` is neither a duration nor a timestamp type
 */
template <typename T>
constexpr inline bool is_chrono()
{
  return is_duration<T>() || is_timestamp<T>();
}

/**
 * @brief Indicates whether `type` is a chrono `data_type`.
 *
 * Chrono types include cudf timestamp types, which represent a point in time, and cudf
 * duration types that represent a time interval.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a chrono type
 * @return false `type` is not a chrono type
 */
bool is_chrono(data_type type);

/**
 * @brief Indicates whether `T` is layout compatible with its "representation" type.
 *
 * For example, in a column, a `decimal32` is concretely represented by a single `int32_t`, but the
 * `decimal32` type itself contains both the integer representation and the scale. Therefore,
 * `decimal32` is _not_ layout compatible with `int32_t`.
 *
 * As further example, `duration_ns` is distinct from its concrete `int64_t` representation type,
 * but they are layout compatible.
 *
 * @return true if `T` is layout compatible with its "representation" type
 */
template <typename T>
constexpr bool is_rep_layout_compatible()
{
  return cudf::is_numeric<T>() or cudf::is_chrono<T>() or cudf::is_boolean<T>() or
         cudf::is_byte<T>();
}

/**
 * @brief Indicates whether the type `T` is a dictionary type.
 *
 * @tparam T  The type to verify
 * @return true `T` is a dictionary-type
 * @return false  `T` is not dictionary-type
 */
template <typename T>
constexpr inline bool is_dictionary()
{
  return std::is_same_v<dictionary32, T>;
}

/**
 * @brief Indicates whether `type` is a dictionary `data_type`.
 *
 * @param type The `data_type` to verify
 * @return true `type` is a dictionary type
 * @return false `type` is not a dictionary type
 */
bool is_dictionary(data_type type);

/**
 * @brief Indicates whether elements of type `T` are fixed-width.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @tparam T The C++ type to verify
 * @return true `T` corresponds to a fixed-width element type
 * @return false `T` corresponds to a variable-width element type
 */
template <typename T>
constexpr inline bool is_fixed_width()
{
  // TODO Add fixed width wrapper types
  // Is a category fixed width?
  return cudf::is_numeric<T>() || cudf::is_chrono<T>() || cudf::is_fixed_point<T>();
}

/**
 * @brief Indicates whether elements of `type` are fixed-width.
 *
 * Elements of a fixed-width type all have the same size in bytes.
 *
 * @param type The `data_type` to verify
 * @return true `type` is fixed-width
 * @return false  `type` is variable-width
 */
bool is_fixed_width(data_type type);

class string_view;

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
  return std::is_same_v<T, cudf::string_view> or std::is_same_v<T, cudf::dictionary32> or
         std::is_same_v<T, cudf::list_view> or std::is_same_v<T, cudf::struct_view>;
}

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
 */
bool is_compound(data_type type);

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
 */
template <typename T>
constexpr inline bool is_nested()
{
  return std::is_same_v<T, cudf::list_view> || std::is_same_v<T, cudf::struct_view>;
}

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
 */
bool is_nested(data_type type);

/**
 * @brief Indicates whether `from` is bit-castable to `to`.
 *
 * This casting is based on std::bit_cast. Data types that have the same size and are trivially
 * copyable are eligible for this casting.
 *
 * See `cudf::bit_cast()` which returns a zero-copy `column_view` when casting between
 * bit-castable types.
 *
 * @param from The `data_type` to convert from
 * @param to The `data_type` to convert to
 * @return `true` if the types are castable
 */
bool is_bit_castable(data_type from, data_type to);

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

/** @} */

}  // namespace cudf
