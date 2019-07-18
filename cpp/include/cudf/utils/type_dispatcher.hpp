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
template <> inline constexpr type_id type_to_id<int8_t>() { return INT8; };
template <> inline constexpr type_id type_to_id<int16_t>() { return INT16; };
template <> inline constexpr type_id type_to_id<int32_t>() { return INT32; };
template <> inline constexpr type_id type_to_id<int64_t>() { return INT64; };
template <> inline constexpr type_id type_to_id<float>() { return FLOAT32; };
template <> inline constexpr type_id type_to_id<double>() { return FLOAT64; };

template <cudf::type_id t> struct id_to_type_impl { using type = void; };
template <> struct id_to_type_impl<INT8> { using type = int8_t; };
template <> struct id_to_type_impl<INT16> { using type = int16_t; };
template <> struct id_to_type_impl<INT32> { using type = int32_t; };
template <> struct id_to_type_impl<INT64> { using type = int64_t; };
template <> struct id_to_type_impl<FLOAT32> { using type = float; };
template <> struct id_to_type_impl<FLOAT64> { using type = double; };
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

// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#pragma nv_exec_check_disable
template <template <cudf::type_id> typename Dispatch = id_to_type_impl,
          typename Functor, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE 
constexpr decltype(auto) type_dispatcher(
    cudf::data_type dtype, Functor f, Ts&&... args) {
  switch (dtype.id()) {
    case INT8: return f.template operator()<typename Dispatch<INT8>::type>( std::forward<Ts>(args)...);
    case INT16: return f.template operator()<typename Dispatch<INT16>::type>( std::forward<Ts>(args)...);
    case INT32: return f.template operator()<typename Dispatch<INT32>::type>( std::forward<Ts>(args)...);
    case INT64: return f.template operator()<typename Dispatch<INT64>::type>( std::forward<Ts>(args)...);
    case FLOAT32: return f.template operator()<typename Dispatch<FLOAT32>::type>( std::forward<Ts>(args)...);
    case FLOAT64: return f.template operator()<typename Dispatch<FLOAT64>::type>( std::forward<Ts>(args)...);
    default: { assert(false && "Unsupported type"); }
  }
}

}  // namespace exp
}  // namespace cudf
