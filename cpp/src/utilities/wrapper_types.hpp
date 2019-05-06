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
#ifndef GDF_CPPTYPES_H
#define GDF_CPPTYPES_H

#include <cudf/types.h>
#include "cudf_utils.h"

#include <cub/util_type.cuh>

#include <iosfwd>
#include <type_traits>
#include <limits>

/* --------------------------------------------------------------------------*/
/** 
 * @file wrapper_types.hpp
 * @brief  Wrapper structs for for the non-fundamental gdf_dtype types.
 *
 * These structs simply wrap a single member variable of a fundamental type 
 * called "value". 
 * 
 * These wrapper structures are used in conjunction with the type_dispatcher to
 * emulate "strong typedefs", i.e., provide opaque types that allow template 
 * specialization. A normal C++ typedef is simply an alias and does not allow
 * for specializing a template or overloading a function.
 * 
 * The purpose of these "strong typedefs" is to provide a one-to-one mapping between
 * gdf_dtype enum values and concrete C++ types and allow distinguishing columns with
 * different gdf_dtype types, but have the same underlying type. For example,
 * the underlying type of both GDF_DATE32 and GDF_INT32 is int32_t. However, if
 * one wished to specialize a functor invoked with the type_dispatcher to handle
 * GDF_DATE32 different from GDF_INT32, that would not be possible with aliases.
 * 
 * The standard arithmetic operators are defined for these wrapper structs such
 * that they can be used as if they were fundamental arithmetic types.
 * 
 * In general, interacting with the wrapper structs should be done via the defined 
 * operators. However, if one needs to directly access the underlying value, the
 * "unwrap" function may be used. Calling `unwrap` on an instance of a wrapper struct
 * will return a reference to the underlying value. Calling `unwrap` on an instance
 * of a fundamental type will return a reference to that instance (effectively a no-op).
 *
 */
/* ----------------------------------------------------------------------------*/
namespace cudf
{
namespace detail
{
/**
     * @brief Base wrapper structure to emulate "strong typedefs" for gdf_dtype values 
     * that do not correspond to fundamental types.
     * 
     * Implements operators that allow the wrapper to be used as if it were a fundamental
     * type.
     * 
     * @tparam T  The type of the wrapped value, i.e., the "underlying type" of the wrapper
     * @tparam type_id  The wrapped gdf_dtype
     */
template <typename T, gdf_dtype type_id>
struct wrapper
{
  static constexpr gdf_dtype corresponding_column_type{type_id}; ///< The wrapped gdf_dtype
  using value_type = T;                                          ///< The underlying fundamental type of the wrapper
  value_type value;                                              ///< The wrapped value

  CUDA_HOST_DEVICE_CALLABLE
  constexpr explicit wrapper(T v) : value{v} {}

  CUDA_HOST_DEVICE_CALLABLE
  explicit operator value_type() const { return this->value; }

  // enable conversion to arithmetic types *only* for the cudf::bool8 wrapper
  // (defined later in this file as wrapper<gdf_bool8, GDF_BOOL8>)
  template <gdf_dtype the_type = type_id,
            typename T_out,
            typename std::enable_if<(the_type == GDF_BOOL8) &&
                                     std::is_arithmetic<T_out>::value,
                                     int>::type* = nullptr >
  CUDA_HOST_DEVICE_CALLABLE
  explicit operator T_out() const { 
    // Casting a cudf::bool8 to arithmetic type should always be the same as
    // casting a bool to arithmetic type, and not the same as casting the
    // underlying type to arithmetic type. Therefore we cast the value to bool
    // first, then the output type
    return static_cast<T_out>(static_cast<bool>(this->value));
  }

  wrapper(wrapper const& w) = default;

  wrapper() = default;
};

template <typename T, gdf_dtype type_id>
std::ostream& operator<<(std::ostream& os, wrapper<T, type_id> const& w) 
{
  return os << w.value;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator==(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs) 
{ 
  return lhs.value == rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator!=(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs) 
{ 
  return lhs.value != rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator<=(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs) 
{ 
  return lhs.value <= rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
bool operator>=(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{ 
  return lhs.value >= rhs.value; 
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE 
bool operator<(wrapper<T, type_id> const &lhs, wrapper<T, type_id> const &rhs)
{
  return lhs.value < rhs.value;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE 
bool operator>(wrapper<T, type_id> const &lhs, wrapper<T, type_id> const &rhs)
{
  return lhs.value > rhs.value;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator+=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value += rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator-=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value -= rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator*=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value *= rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id>& operator/=(wrapper<T,type_id> & lhs, wrapper<T,type_id> const& rhs)
{
  lhs.value /= rhs.value;
  return lhs;
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator+(wrapper<T, type_id> const &lhs, wrapper<T, type_id> const &rhs)
{
  return wrapper<T, type_id>{ static_cast<T>(lhs.value + rhs.value) };
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator-(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{
  return wrapper<T, type_id>{ static_cast<T>(lhs.value - rhs.value) };
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator*(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{
  return wrapper<T, type_id>{ static_cast<T>(lhs.value * rhs.value) };
}

template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
wrapper<T,type_id> operator/(wrapper<T,type_id> const& lhs, wrapper<T,type_id> const& rhs)
{
  return wrapper<T, type_id>{ static_cast<T>(lhs.value / rhs.value) };
}

/* --------------------------------------------------------------------------*/
/** 
     * @brief  Returns a reference to the underlying "value" member of a wrapper struct
     * 
     * @param[in] wrapped A non-const reference to the wrapper struct to unwrap
     * 
     * @returns A reference to the underlying wrapped value  
     */
/* ----------------------------------------------------------------------------*/
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
typename wrapper<T, type_id>::value_type &
unwrap(wrapper<T, type_id> &wrapped)
{
  return wrapped.value;
}

/* --------------------------------------------------------------------------*/
/** 
     * @brief  Returns a reference to the underlying "value" member of a wrapper struct
     * 
     * @param[in] wrapped A const reference to the wrapper struct to unwrap
     * 
     * @returns A const reference to the underlying wrapped value  
     */
/* ----------------------------------------------------------------------------*/
template <typename T, gdf_dtype type_id>
CUDA_HOST_DEVICE_CALLABLE
typename wrapper<T, type_id>::value_type const &
unwrap(wrapper<T, type_id> const &wrapped)
{
  return wrapped.value;
}

/* --------------------------------------------------------------------------*/
/** 
     * @brief Passthrough function for fundamental types
     *
     * This specialization of "unwrap" is provided such that it can be used in generic
     * code that is agnostic to whether or not the type being operated on is a wrapper
     * struct or a fundamental type
     * 
     * @param[in] value Reference to a fundamental type to passthrough
     * 
     * @returns Reference to the value passed in
     */
/* ----------------------------------------------------------------------------*/
template <typename T>
CUDA_HOST_DEVICE_CALLABLE
typename std::enable_if_t<
  std::is_fundamental<typename std::decay<T>::type>::value, T> &
unwrap(T &value)
{
  return value;
}

/* --------------------------------------------------------------------------*/
/** 
     * @brief Passthrough function for fundamental types
     *
     * This specialization of "unwrap" is provided such that it can be used in generic
     * code that is agnostic to whether or not the type being operated on is a wrapper
     * struct or a fundamental type
     * 
     * @param[in] value const reference to a fundamental type to passthrough
     * 
     * @returns const reference to the value passed in
     */
/* ----------------------------------------------------------------------------*/
template <typename T>
CUDA_HOST_DEVICE_CALLABLE
typename std::enable_if_t<
  std::is_fundamental<typename std::decay<T>::type>::value, T> const &
unwrap(T const &value)
{
  return value;
}

/**---------------------------------------------------------------------------*
 * @brief Trait to use to get underlying type of wrapped object
 * 
 * This struct can be used with either a fundamental type or a wrapper type and
 * it uses unwrap to get the underlying type.
 * 
 * Example use case: 
 *  Making a functor to use with a `type_dispatcher` that works on the
 *  underlying type of all `gdf_dtype`
 *  
 * ```c++
 * struct example_functor{
 *  template <typename T>
 *  int operator()(){
 *    using T1 = cudf::detail::unwrapped_type<T>::type;
 *    return sizeof(T1);
 *  }
 * };
 * ```
 * 
 * @tparam T Either wrapped object type or fundamental type
 *---------------------------------------------------------------------------**/
template <typename T>
struct unwrapped_type {
  using type = std::decay_t<decltype(unwrap(std::declval<T&>()))>;
};

/**---------------------------------------------------------------------------*
 * @brief Helper type for `unwrapped_type`
 * 
 * Example:
 * ```c++
 * using T1 = cudf::detail::unwrapped_type_t<date32>; // T1 = int 
 * using T2 = cudf::detail::unwrapped_type_t<float>;  // T2 = float 
 * ```
 * 
 * @tparam T Either wrapped object type or fundamental type
 *---------------------------------------------------------------------------**/
template <typename T>
using unwrapped_type_t = typename unwrapped_type<T>::type;

} // namespace detail

using category = detail::wrapper<gdf_category, GDF_CATEGORY>;

using nvstring_category = detail::wrapper<gdf_nvstring_category, GDF_STRING_CATEGORY>;

using timestamp = detail::wrapper<gdf_timestamp, GDF_TIMESTAMP>;

using date32 = detail::wrapper<gdf_date32, GDF_DATE32>;

using date64 = detail::wrapper<gdf_date64, GDF_DATE64>;

using bool8 = detail::wrapper<gdf_bool8, GDF_BOOL8>;

// This is necessary for global, constant, non-fundamental types
// We can't rely on --expt-relaxed-constexpr here because `bool8` is not a
// scalar type. See CUDA Programming guide
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables
#define HOST_DEVICE_CONSTANT
#ifdef __CUDA_ARCH__
__device__ __constant__ static bool8 true_v{gdf_bool8{1}};

__device__ __constant__ static bool8 false_v{gdf_bool8{0}};
#else
static constexpr bool8 true_v{gdf_bool8{1}};
static constexpr bool8 false_v{gdf_bool8{0}};
#endif

// Wrapper operator overloads for cudf::bool8 
namespace detail {

inline
std::ostream& operator<<(std::ostream& os, cudf::bool8 const& w) 
{
  return os << static_cast<bool>(w);
}

CUDA_HOST_DEVICE_CALLABLE
bool operator==(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<bool>(lhs) == static_cast<bool>(rhs);
}

CUDA_HOST_DEVICE_CALLABLE
bool operator!=(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<bool>(lhs) != static_cast<bool>(rhs);
}

CUDA_HOST_DEVICE_CALLABLE
bool operator<=(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{ 
  return static_cast<bool>(lhs) <= static_cast<bool>(rhs); 
}

CUDA_HOST_DEVICE_CALLABLE
bool operator>=(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{ 
  return static_cast<bool>(lhs) >= static_cast<bool>(rhs); 
}

CUDA_HOST_DEVICE_CALLABLE 
bool operator<(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<bool>(lhs) < static_cast<bool>(rhs);
}

CUDA_HOST_DEVICE_CALLABLE 
bool operator>(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<bool>(lhs) > static_cast<bool>(rhs);
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8 operator+(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<cudf::bool8>(static_cast<bool>(lhs) +
                                  static_cast<bool>(rhs));
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8 operator-(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<cudf::bool8>(static_cast<bool>(lhs) -
                                  static_cast<bool>(rhs));
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8 operator*(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<cudf::bool8>(static_cast<bool>(lhs) *
                                  static_cast<bool>(rhs));
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8 operator/(cudf::bool8 const &lhs, cudf::bool8 const &rhs)
{
  return static_cast<cudf::bool8>(static_cast<bool>(lhs) /
                                  static_cast<bool>(rhs));
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8& operator+=(cudf::bool8 &lhs, cudf::bool8 const &rhs)
{
  lhs = lhs + rhs;
  return lhs;
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8& operator-=(cudf::bool8 &lhs, cudf::bool8 const &rhs)
{
  lhs = lhs - rhs;
  return lhs;
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8& operator*=(cudf::bool8 &lhs, cudf::bool8 const &rhs)
{
  lhs = lhs * rhs;
  return lhs;
}

CUDA_HOST_DEVICE_CALLABLE
cudf::bool8& operator/=(cudf::bool8 &lhs, cudf::bool8 const &rhs)
{
  lhs = lhs / rhs;
  return lhs;
}

} // namespace detail

} // namespace cudf

namespace std
{

/**---------------------------------------------------------------------------*
 * @brief Specialization of std::numeric_limits for wrapper types
 *---------------------------------------------------------------------------**/
template <typename T, gdf_dtype type_id>
struct numeric_limits< cudf::detail::wrapper<T, type_id> > {
  
  using wrapper_t = cudf::detail::wrapper<T, type_id>;

  /**---------------------------------------------------------------------------*
   * @brief Returns the maximum finite value representable by the numeric type T
   *---------------------------------------------------------------------------**/
  static constexpr wrapper_t max() noexcept {
    return wrapper_t{ std::numeric_limits<T>::max() };
  }
  
  /**---------------------------------------------------------------------------*
   * @brief Returns the lowest finite value representable by the numeric type T
   * 
   * Returns a finite value x such that there is no other finite value y where y < x
   *---------------------------------------------------------------------------**/
  static constexpr wrapper_t lowest() noexcept {
    return wrapper_t{ std::numeric_limits<T>::lowest() };
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the minimum finite value representable by the numeric type T
   * 
   * For floating-point types with denormalization, min returns the minimum
   * positive normalized value.
   *---------------------------------------------------------------------------**/
  static constexpr wrapper_t min() noexcept {
    return wrapper_t{ std::numeric_limits<T>::min() };
  }

};

/** --------------------------------------------------------------------------*
  * @brief Specialization of std::numeric_limits for cudf::bool8
  *
  * Required since the underlying type, int8_t, has different limits than bool
  * --------------------------------------------------------------------------**/
template <>
struct numeric_limits< cudf::bool8 > {
  
  static constexpr cudf::bool8 max() noexcept {
    // tried using `return cudf::true_v` but it causes a compiler segfault!
    return cudf::bool8{true};
  }
  
  static constexpr cudf::bool8 lowest() noexcept {
    return cudf::bool8{false};
  }

  static constexpr cudf::bool8 min() noexcept {
    return cudf::bool8{false};
  }
};

} // std

namespace cub
{

template <> struct NumericTraits<cudf::date32> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::date32>>,
    cudf::detail::unwrapped_type_t<cudf::date32>> {};

template <> struct NumericTraits<cudf::timestamp> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::timestamp>>,
    cudf::detail::unwrapped_type_t<cudf::timestamp>> {};

template <> struct NumericTraits<cudf::date64> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::date64>>,
    cudf::detail::unwrapped_type_t<cudf::date64>> {};

template <> struct NumericTraits<cudf::category> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::category>>,
    cudf::detail::unwrapped_type_t<cudf::category>> {};

template <> struct NumericTraits<cudf::nvstring_category> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::nvstring_category>>,
    cudf::detail::unwrapped_type_t<cudf::nvstring_category>> {};

template <> struct NumericTraits<cudf::bool8> :
  BaseTraits<SIGNED_INTEGER, true, false,
    std::make_unsigned_t<cudf::detail::unwrapped_type_t<cudf::bool8>>,
    cudf::detail::unwrapped_type_t<cudf::bool8>> {};

} // cub

#endif
