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
#include <cudf/utilities/error.hpp>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/wrappers/bool.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <string>

/**---------------------------------------------------------------------------*
 * @file type_dispatcher.hpp
 * @brief Defines the mapping between `cudf::type_id` runtime type information
 * and concrete C++ types.
 *---------------------------------------------------------------------------**/
namespace cudf {
namespace experimental {
/**---------------------------------------------------------------------------*
 * @brief Maps a C++ type to it's corresponding `cudf::type_id`
 *
 * When explicitly passed a template argument of a given type, returns the
 * appropriate `type_id` enum for the specified C++ type.
 *
 * For example:
 *
 * ```
 * return cudf::type_to_id<int32_t>();        // Returns INT32
 * ```
 *
 * @tparam T The type to map to a `cudf::type_id`
 *---------------------------------------------------------------------------**/
template <typename T>
inline constexpr type_id type_to_id() {
  return EMPTY;
};

struct type_to_name {
  template <typename T>
  inline std::string operator()() {
    return "void";
  }
};

template <cudf::type_id t>
struct id_to_type_impl {
  using type = void;
};
/**---------------------------------------------------------------------------*
 * @brief Maps a `cudf::type_id` to it's corresponding concrete C++ type
 *
 * Example:
 * ```
 * static_assert(std::is_same<int32_t, id_to_type<INT32>);
 * ```
 * @tparam t The `cudf::type_id` to map
 *---------------------------------------------------------------------------**/
template <cudf::type_id Id>
using id_to_type = typename id_to_type_impl<Id>::type;

/**---------------------------------------------------------------------------*
 * @brief Macro used to define a mapping between a concrete C++ type and a
 *`cudf::type_id` enum.

 * @param Type The concrete C++ type
 * @param Id The `cudf::type_id` enum
 *---------------------------------------------------------------------------**/
#ifndef CUDF_TYPE_MAPPING
#define CUDF_TYPE_MAPPING(Type, Id)             \
  template <>                                   \
  constexpr inline type_id type_to_id<Type>() { \
    return Id;                                  \
  }                                             \
  template <>                                   \
  inline std::string                            \
  type_to_name::operator()<Type>() {            \
    return CUDF_STRINGIFY(Type);                \
  }                                             \
  template <>                                   \
  struct id_to_type_impl<Id> {                  \
    using type = Type;                          \
  }
#endif

/**---------------------------------------------------------------------------*
 * @brief Defines all of the mappings between C++ types and their corresponding
 * `cudf::type_id` values.
 *---------------------------------------------------------------------------**/
CUDF_TYPE_MAPPING(cudf::experimental::bool8, type_id::BOOL8);
CUDF_TYPE_MAPPING(int8_t, type_id::INT8);
CUDF_TYPE_MAPPING(int16_t, type_id::INT16);
CUDF_TYPE_MAPPING(int32_t, type_id::INT32);
CUDF_TYPE_MAPPING(int64_t, type_id::INT64);
CUDF_TYPE_MAPPING(float, type_id::FLOAT32);
CUDF_TYPE_MAPPING(double, type_id::FLOAT64);
CUDF_TYPE_MAPPING(cudf::string_view, type_id::STRING);
CUDF_TYPE_MAPPING(cudf::timestamp_D, type_id::TIMESTAMP_DAYS);
CUDF_TYPE_MAPPING(cudf::timestamp_s, type_id::TIMESTAMP_SECONDS);
CUDF_TYPE_MAPPING(cudf::timestamp_ms, type_id::TIMESTAMP_MILLISECONDS);
CUDF_TYPE_MAPPING(cudf::timestamp_us, type_id::TIMESTAMP_MICROSECONDS);
CUDF_TYPE_MAPPING(cudf::timestamp_ns, type_id::TIMESTAMP_NANOSECONDS);


template <typename T>
struct type_to_scalar_type_impl {
  using ScalarType = cudf::scalar;
};

#ifndef MAP_NUMERIC_SCALAR
#define MAP_NUMERIC_SCALAR(Type)                    \
template <>                                         \
struct type_to_scalar_type_impl<Type> {             \
  using ScalarType = cudf::numeric_scalar<Type>;    \
  using ScalarDeviceType = cudf::numeric_scalar_device_view<Type>; \
};
#endif

MAP_NUMERIC_SCALAR(int8_t)
MAP_NUMERIC_SCALAR(int16_t)
MAP_NUMERIC_SCALAR(int32_t)
MAP_NUMERIC_SCALAR(int64_t)
MAP_NUMERIC_SCALAR(float)
MAP_NUMERIC_SCALAR(double)
MAP_NUMERIC_SCALAR(cudf::experimental::bool8)

template <>
struct type_to_scalar_type_impl<cudf::string_view> {
  using ScalarType = cudf::string_scalar;
  using ScalarDeviceType = cudf::string_scalar_device_view;
};

#ifndef MAP_TIMESTAMP_SCALAR
#define MAP_TIMESTAMP_SCALAR(Type)                  \
template <>                                         \
struct type_to_scalar_type_impl<Type> {             \
  using ScalarType = cudf::timestamp_scalar<Type>;  \
  using ScalarDeviceType = cudf::timestamp_scalar_device_view<Type>;       \
};
#endif

MAP_TIMESTAMP_SCALAR(timestamp_D)
MAP_TIMESTAMP_SCALAR(timestamp_s)
MAP_TIMESTAMP_SCALAR(timestamp_ms)
MAP_TIMESTAMP_SCALAR(timestamp_us)
MAP_TIMESTAMP_SCALAR(timestamp_ns)

/**
 * @brief Maps a C++ type to the scalar type required to hold its value
 * 
 * @tparam T The concrete C++ type to map
 */
template <typename T>
using scalar_type_t = typename type_to_scalar_type_impl<T>::ScalarType;

template <typename T>
using scalar_device_type_t = typename type_to_scalar_type_impl<T>::ScalarDeviceType;

/**---------------------------------------------------------------------------*
 * @brief Invokes an `operator()` template with the type instantiation based on
 * the specified `cudf::data_type`'s `id()`.
 *
 * Example usage with a functor that returns the size of the dispatched type:
 *
 * ```
 * struct size_of_functor{
 *  template <typename T>
 *  int operator()(){
 *    return sizeof(T);
 *  }
 * };
 * cudf::data_type t{INT32};
 * cudf::type_dispatcher(t, size_of_functor{});  // returns 4
 * ```
 *
 * The `type_dispatcher` uses `cudf::type_to_id<t>` to provide a default mapping
 * of `cudf::type_id`s to dispatched C++ types. However, this mapping may be
 * customized by explicitly specifying a user-defined trait struct for the
 * `IdTypeMap`. For example, to always dispatch `int32_t`
 *
 * ```
 * template<cudf::type_id t> struct always_int{ using type = int32_t; }
 *
 * // This will always invoke `operator()<int32_t>`
 * cudf::type_dispatcher<always_int>(data_type, f);
 * ```
 *
 * It is sometimes necessary to customize the dispatched functor's
 * `operator()` for different types.  This can be done in several ways.
 *
 * The first method is to use explicit template specialization. This is useful
 * for specializing behavior for single types. For example, a functor that
 * prints `int32_t` or `double` when invoked with either of those types, else it
 * prints `unhandled type`:
 *
 * ```
 * struct type_printer {
 *   template <typename ColumnType>
 *   void operator()() { std::cout << "unhandled type\n"; }
 * };
 *
 * // Due to a bug in g++, explicit member function specializations need to be
 * // defined outside of the class definition
 * template <>
 * void type_printer::operator()<int32_t>() { std::cout << "int32_t\n"; }
 *
 * template <>
 * void type_printer::operator()<double>() { std::cout << "double\n"; }
 * ```
 *
 * A second method is to use SFINAE with `std::enable_if_t`. This is useful for
 * specializing for a set of types that share some property. For example, a
 * functor that prints `integral` or `floating point` for integral or floating
 * point types:
 *
 * ```
 * struct integral_or_floating_point {
 *   template <typename ColumnType,
 *             std::enable_if_t<not std::is_integral<ColumnType>::value and
 *                              not std::is_floating_point<ColumnType>::value>*
 *= nullptr> void operator()() { std::cout << "neither integral nor floating
 *point\n"; }
 *
 *   template <typename ColumnType,
 *             std::enable_if_t<std::is_integral<ColumnType>::value>* = nullptr>
 *   void operator()() { std::cout << "integral\n"; }
 *
 *   template < typename ColumnType,
 *              std::enable_if_t<std::is_floating_point<ColumnType>::value>* =
 *nullptr> void operator()() { std::cout << "floating point\n"; }
 * };
 * ```
 *
 * For more info on SFINAE and `std::enable_if`, see
 * https://eli.thegreenplace.net/2014/sfinae-and-enable_if/
 *
 * The return type for all template instantiations of the functor's "operator()"
 * lambda must be the same, else there will be a compiler error as you would be
 * trying to return different types from the same function.
 *
 * @tparam id_to_type_impl Maps a `cudf::type_id` its dispatched C++ type
 * @tparam Functor The callable object's type
 * @tparam Ts Variadic parameter pack type
 * @param dtype The `cudf::data_type` whose `id()` determines which template
 * instantiation is invoked
 * @param f The callable whose `operator()` template is invoked
 * @param args Parameter pack of arguments forwarded to the `operator()`
 * invocation
 * @return Whatever is returned by the callable's `operator()`
 *---------------------------------------------------------------------------**/
// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#pragma nv_exec_check_disable
template <template <cudf::type_id> typename IdTypeMap = id_to_type_impl,
          typename Functor, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr decltype(auto) type_dispatcher(
    cudf::data_type dtype, Functor f, Ts&&... args) {
  switch (dtype.id()) {
    case BOOL8:
      return f.template operator()<typename IdTypeMap<BOOL8>::type>(
          std::forward<Ts>(args)...);
    case INT8:
      return f.template operator()<typename IdTypeMap<INT8>::type>(
          std::forward<Ts>(args)...);
    case INT16:
      return f.template operator()<typename IdTypeMap<INT16>::type>(
          std::forward<Ts>(args)...);
    case INT32:
      return f.template operator()<typename IdTypeMap<INT32>::type>(
          std::forward<Ts>(args)...);
    case INT64:
      return f.template operator()<typename IdTypeMap<INT64>::type>(
          std::forward<Ts>(args)...);
    case FLOAT32:
      return f.template operator()<typename IdTypeMap<FLOAT32>::type>(
          std::forward<Ts>(args)...);
    case FLOAT64:
      return f.template operator()<typename IdTypeMap<FLOAT64>::type>(
          std::forward<Ts>(args)...);
    case STRING:
      return f.template operator()<typename IdTypeMap<STRING>::type>(
          std::forward<Ts>(args)...);
    case TIMESTAMP_DAYS:
      return f.template operator()<typename IdTypeMap<TIMESTAMP_DAYS>::type>(
          std::forward<Ts>(args)...);
    case TIMESTAMP_SECONDS:
      return f.template operator()<typename IdTypeMap<TIMESTAMP_SECONDS>::type>(
          std::forward<Ts>(args)...);
    case TIMESTAMP_MILLISECONDS:
      return f.template operator()<typename IdTypeMap<TIMESTAMP_MILLISECONDS>::type>(
          std::forward<Ts>(args)...);
    case TIMESTAMP_MICROSECONDS:
      return f.template operator()<typename IdTypeMap<TIMESTAMP_MICROSECONDS>::type>(
          std::forward<Ts>(args)...);
    case TIMESTAMP_NANOSECONDS:
      return f.template operator()<typename IdTypeMap<TIMESTAMP_NANOSECONDS>::type>(
          std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Unsupported type_id.");
#else
      release_assert(false && "Unsuported type_id.");

      // The following code will never be reached, but the compiler generates a
      // warning if there isn't a return value.

      // Need to find out what the return type is in order to have a default
      // return value and solve the compiler warning for lack of a default
      // return
      using return_type =
          decltype(f.template operator()<int8_t>(std::forward<Ts>(args)...));
      return return_type();
#endif
    }
  }
}

}  // namespace experimental
}  // namespace cudf
