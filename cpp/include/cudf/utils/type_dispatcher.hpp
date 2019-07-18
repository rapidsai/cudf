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
 * return cudf::type_of<int32_t>();        // Returns INT32
 * ```
 *
 * @tparam T The type to map to a `cudf::type`
 *---------------------------------------------------------------------------**/
template <typename T>
inline constexpr type type_of() { return EMPTY; };
template <> inline constexpr type type_of<int8_t>() { return INT8; };
template <> inline constexpr type type_of<int16_t>() { return INT16; };
template <> inline constexpr type type_of<int32_t>() { return INT32; };
template <> inline constexpr type type_of<int64_t>() { return INT64; };
template <> inline constexpr type type_of<float>() { return FLOAT32; };
template <> inline constexpr type type_of<double>() { return FLOAT64; };

/**---------------------------------------------------------------------------*
 * @brief Maps a `cudf::type` to it's corresponding concrete C++ type
 * 
 * Example:
 * ```
 * static_assert(std::is_same<int32_t, typename corresponding_type<INT32>::type>);
 * ```
 *
 * @tparam t The `cudf::type` to map
 *---------------------------------------------------------------------------**/
template <cudf::type t> struct corresponding_type { using type = void; };
template<> struct corresponding_type<INT8> { using type = int8_t; };
template<> struct corresponding_type<INT16> { using type = int16_t; };
template<> struct corresponding_type<INT32> { using type = int32_t; };
template<> struct corresponding_type<INT64> { using type = int64_t; };
template<> struct corresponding_type<FLOAT32> { using type = float; };
template<> struct corresponding_type<FLOAT64> { using type = double; };

/**---------------------------------------------------------------------------*
 * @brief Helper alias for `corresponding_type<t>::type`
 * 
 * Example:
 * ```
 * static_assert(std::is_same<int32_t, corresponding_type_t<INT32>);
 * ```
 * 
 * @tparam t The `cudf::type` to map
 *---------------------------------------------------------------------------**/
template <cudf::type t>
using corresponding_type_t = typename corresponding_type<t>::type;

// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#pragma nv_exec_check_disable
template <class Functor, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE
constexpr decltype(auto) type_dispatcher(cudf::data_type dtype, Functor f,
                                         Ts&&... args) {
  switch (dtype.id()) {
    case INT8:    return f.template operator()<corresponding_type_t<INT8>>( std::forward<Ts>(args)...);
    case INT16:   return f.template operator()<corresponding_type_t<INT16>>( std::forward<Ts>(args)...);
    case INT32:   return f.template operator()<corresponding_type_t<INT32>>( std::forward<Ts>(args)...);
    case INT64:   return f.template operator()<corresponding_type_t<INT64>>( std::forward<Ts>(args)...);
    case FLOAT32: return f.template operator()<corresponding_type_t<FLOAT32>>( std::forward<Ts>(args)...);
    case FLOAT64: return f.template operator()<corresponding_type_t<FLOAT64>>( std::forward<Ts>(args)...);
    default: { assert(false && "Unsupported type"); }
  }
}


}  // namespace exp
}  // namespace cudf

