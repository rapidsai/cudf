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
#ifndef TYPE_DISPATCHER_HPP
#define TYPE_DISPATCHER_HPP

#include "wrapper_types.hpp"
#include "release_assert.cuh"

#include <cudf/types.h>

// Forward decl
class NVStrings;

#include <cassert>
#include <utility>


/* --------------------------------------------------------------------------*/
/**
 * @brief  Invokes an instance of a functor template with the appropriate type
 * determined by a gdf_dtype enum value.
 *
 * This helper function accepts any object with an "operator()" template,
 * e.g., a functor. It will invoke an instance of the template by passing
 * in as the template argument an appropriate type determined by the value of
 * the gdf_dtype argument.
 *
 * The template may have 1 or more template parameters, but the first parameter
 * must be the type dispatched from the gdf_dtype enum.  The remaining template
 * parameters must be able to be automatically deduced.
 *
 * There is a 1-to-1 mapping of gdf_dtype enum values and dispatched types.
 * However, different gdf_dtype values may have the same underlying type.
 * Therefore, in order to provide the 1-to-1 mapping, a wrapper struct may be
 * dispatched for certain gdf_dtype enum values in order to emulate a "strong
 * typedef".
 *
 * A strong typedef  provides a new, concrete type unlike a normal C++ typedef
 * which is simply a type alias. These "strong typedef" structs simply wrap a
 * single member variable of a fundamental type called 'value'.
 *
 * The standard arithmetic operators are defined for the wrapper structs and
 * therefore the wrapper struct types can be used as if they were fundamental
 * types.
 *
 * See wrapper_types.hpp for more detail.
 *
 * Example usage with a functor that returns the size of the dispatched type:
 *
 * struct example_functor{
 *  template <typename T>
 *  int operator()(){
 *    return sizeof(T);
 *  }
 * };
 *
 * cudf::type_dispatcher(GDF_INT8, example_functor);  // returns 1
 * cudf::type_dispatcher(GDF_INT64, example_functor); // returns 8
 *
 * Example usage of a functor for checking if element "i" in column "lhs" is
 * equal to element "j" in column "rhs":
 *
 * struct elements_are_equal{
 *   template <typename ColumnType>
 *   bool operator()(void const * lhs, int i,
 *                   void const * rhs, int j)
 *   {
 *     // Cast the void* data buffer to the dispatched type and retrieve
 * elements
 *     // "i" and "j" from the respective columns
 *     ColumnType const i_elem = static_cast<ColumnType const*>(lhs)[i];
 *     ColumnType const j_elem = static_cast<ColumnType const*>(rhs)[j];
 *
 *     // operator== is defined for wrapper structs such that it performs the
 *     // operator== on the underlying values. Therefore, the wrapper structs
 *     // can be used as if they were fundamental arithmetic types
 *     return i_elem == j_elem;
 *   }
 * };
 *
 * The return type for all template instantiations of the functor's "operator()"
 * lambda must be the same, else there will be a compiler error as you would be
 * trying to return different types from the same function.
 * 
 * NOTE: It is undefined behavior if an unsupported or invalid `gdf_dtype` is
 * supplied.
 *
 * @param dtype The gdf_dtype enum that determines which type will be dispatched
 * @param f The functor with a templated "operator()" that will be invoked with 
 * the dispatched type
 * @param args A parameter-pack (i.e., arbitrary number of arguments) that will 
 * be perfectly-forwarded as the arguments of the functor's "operator()".
 *
 * @returns Whatever is returned by the functor's "operator()". 
 *
 */
/* ----------------------------------------------------------------------------*/
namespace cudf {

// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#pragma hd_warning_disable 
#pragma nv_exec_check_disable
template <class functor_t, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE decltype(auto) type_dispatcher(gdf_dtype dtype,
                                                         functor_t f,
                                                         Ts&&... args) {
  switch(dtype)
  {
    // The .template is known as a "template disambiguator"
    // See here for more information:
    // https://stackoverflow.com/questions/3786360/confusing-template-error
    case GDF_INT8:      { return f.template operator()< int8_t >(std::forward<Ts>(args)...); }
    case GDF_INT16:     { return f.template operator()< int16_t >(std::forward<Ts>(args)...); }
    case GDF_INT32:     { return f.template operator()< int32_t >(std::forward<Ts>(args)...); }
    case GDF_INT64:     { return f.template operator()< int64_t >(std::forward<Ts>(args)...); }
    case GDF_FLOAT32:   { return f.template operator()< float >(std::forward<Ts>(args)...); }
    case GDF_FLOAT64:   { return f.template operator()< double >(std::forward<Ts>(args)...); }
    case GDF_BOOL8:     { return f.template operator()< bool8 >(std::forward<Ts>(args)...); }
    case GDF_DATE32:    { return f.template operator()< date32 >(std::forward<Ts>(args)...); }
    case GDF_DATE64:    { return f.template operator()< date64 >(std::forward<Ts>(args)...); }
    case GDF_TIMESTAMP: { return f.template operator()< timestamp >(std::forward<Ts>(args)...); }
    case GDF_CATEGORY:  { return f.template operator()< category >(std::forward<Ts>(args)...); }
    case GDF_STRING_CATEGORY:  
                        { return f.template operator()< nvstring_category >(std::forward<Ts>(args)...); }
    default: {
#ifdef __CUDA_ARCH__
      
      // This will cause the calling kernel to crash as well as invalidate
      // the GPU context
      release_assert(false && "Invalid gdf_dtype in type_dispatcher");

      // The following code will never be reached, but the compiler generates a
      // warning if there isn't a return value.

      // Need to find out what the return type is in order to have a default
      // return value and solve the compiler warning for lack of a default
      // return
      using return_type =
          decltype(f.template operator()<int8_t>(std::forward<Ts>(args)...));
      return return_type();
#else
      // In host-code, the compiler is smart enough to know we don't need a
      // default return type since we're throwing an exception.
      throw std::runtime_error("Invalid gdf_dtype in type_dispatcher");
#endif
    }
  }
}

/**---------------------------------------------------------------------------*
 * @brief Maps a C++ type to it's corresponding gdf_dtype.
 *
 * When explicitly passed a template argument of a given type, returns the
 * appropriate `gdf_dtype` for the specified C++ type.
 *
 * For example:
 * ```
 * return gdf_dtype_of<int32_t>();        // Returns GDF_INT32
 * return gdf_dtype_of<cudf::category>(); // Returns GDF_CATEGORY
 * ```
 *
 * @tparam T The type to map to a `gdf_dtype`
 *---------------------------------------------------------------------------**/
template <typename T>
inline constexpr gdf_dtype gdf_dtype_of() {
  return GDF_invalid;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<int8_t>() {
  return GDF_INT8;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<int16_t>() {
  return GDF_INT16;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<int32_t>() {
  return GDF_INT32;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<int64_t>() {
  return GDF_INT64;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<float>() {
  return GDF_FLOAT32;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<double>() {
  return GDF_FLOAT64;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<cudf::bool8>() {
  return GDF_BOOL8;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<cudf::date32>() {
  return GDF_DATE32;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<cudf::date64>() {
  return GDF_DATE64;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<cudf::timestamp>() {
  return GDF_TIMESTAMP;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<cudf::category>() {
  return GDF_CATEGORY;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<cudf::nvstring_category>() {
  return GDF_STRING_CATEGORY;
};

template <>
inline constexpr gdf_dtype gdf_dtype_of<NVStrings>() {
  return GDF_STRING;
};

}  // namespace cudf

#endif
