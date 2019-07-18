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

#ifndef CUDA_HOST_DEVICE_CALLABLE
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_CALLABLE __host__ __device__ inline
#define CUDA_DEVICE_CALLABLE __device__ inline
#else
#define CUDA_HOST_DEVICE_CALLABLE inline
#define CUDA_DEVICE_CALLABLE inline
#endif
#endif

/**
 *
 */
namespace cudf {
namespace exp {


/**---------------------------------------------------------------------------*
 * @brief Maps a C++ type to it's corresponding `cudf::type` id
 *
 * When explicitly passed a template argument of a given type, returns the
 * appropriate `type` enum for the specified C++ type.
 *
 * For example:
 *
 * ```
 * return cudf::type_to_idk<int32_t>();        // Returns INT32
 * ```
 *
 * @tparam T The type to map to a `cudf::type`
 *---------------------------------------------------------------------------**/
template <typename T>
inline constexpr type_id type_to_id() { return EMPTY; };


template <cudf::type_id t> struct id_to_type_impl { using type = void; };
/**---------------------------------------------------------------------------*
 * @brief Maps a `cudf::type_id` to it's corresponding concrete C++ type
 *
 * Example:
 * ```
 * static_assert(std::is_same<int32_t, id_to_type<INT32>);
 * ```
 * @tparam t The `cudf::type_id` to map
 *---------------------------------------------------------------------------**/
template <cudf::type_id t>
using id_to_type = typename id_to_type_impl<t>::type;

/**---------------------------------------------------------------------------*
 * @brief Macro used to define a mapping between a concrete C++ type and a
 *`cudf::type_id` enum.

 * @param Type The concrete C++ type
 * @param Id The `cudf::type_id` enum
 *---------------------------------------------------------------------------**/
#ifndef ADD_MAPPING
#define ADD_MAPPING(Type, Id) \
template<> constexpr inline type_id type_to_id<Type>() { return Id; } \
template<> struct id_to_type_impl<Id> { using type = Type; }
#endif

/**---------------------------------------------------------------------------*
 * @brief Defines all of the mappings between C++ types and their corresponding
 * `cudf::type_id` values.
 *---------------------------------------------------------------------------**/
ADD_MAPPING(int8_t, INT8);
ADD_MAPPING(int16_t, INT16);
ADD_MAPPING(int32_t, INT32);
ADD_MAPPING(int64_t, INT64);
ADD_MAPPING(float, FLOAT32);
ADD_MAPPING(double, FLOAT64);

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
 * The `type_dispatcher` provides a default mapping of `cudf::type_id`s to
 * dispatched C++ types. However, this mapping may be customized by explicitly
 * specifying a user-defined trait struct for the `IdTypeMap`. For example, to
 * always dispatch `int32_t`
 * 
 * ```
 * template<cudf::type_id t> struct always_int{ using type = int32_t; }
 * 
 * // This will always invoke `operator()<int32_t>`
 * cudf::type_dispatcher<always_int>(data_type, f); 
 * ```
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
    default: { assert(false && "Unsupported type"); }
  }
}

}  // namespace exp
}  // namespace cudf
