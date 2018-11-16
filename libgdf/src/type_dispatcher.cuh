#ifndef GDF_TYPE_DISPATCHER_H
#define GDF_TYPE_DISPATCHER_H

#include <cassert>
#include <utility>
#include <gdf/cffi/types.h>
#include "NVStrings.h"

namespace {
/* --------------------------------------------------------------------------*/
/** 
 * @brief  Traits struct that maps a gdf_dtype to the appropriate underlying type.
 */
/* ----------------------------------------------------------------------------*/
template<gdf_dtype t> struct default_enum_map;
template <> struct default_enum_map<GDF_INT8>{ using type = int8_t; };
template <> struct default_enum_map<GDF_INT16>{ using type = int16_t; };
template <> struct default_enum_map<GDF_INT32>{ using type = int32_t; };
template <> struct default_enum_map<GDF_INT64>{ using type = int64_t; };
template <> struct default_enum_map<GDF_FLOAT32>{ using type = float; };
template <> struct default_enum_map<GDF_FLOAT64>{ using type = double; };
template <> struct default_enum_map<GDF_DATE32>{ using type = gdf_date32; };
template <> struct default_enum_map<GDF_DATE64>{ using type = gdf_date64; };
template <> struct default_enum_map<GDF_TIMESTAMP>{ using type = gdf_timestamp; };
template <> struct default_enum_map<GDF_CATEGORY>{ using type = gdf_category; };
template <> struct default_enum_map<GDF_STRING>{ using type = NVStrings; };
} // anonymous namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Invokes an instance of a functor template with a traits struct that
 * can be used within the functor's body to retrieve the appropriate underlying 
 * type corresponding to the passed in gdf_dtype enum.
 *
 * This helper function accepts any callable object with an "operator()" template,
 * e.g., a functor or a templated lambda. It will invoke an instance of the template
 * by passing in as the template argument a "traits struct" that is templated on 
 * a gdf_dtype. This trait struct has a member type called "type" that defines 
 * what underlying type should be used for the corresponding gdf_dtype.
 * 
 * The template may have 1 or more template parameters, but the first parameter must 
 * be the traits structure dispatched from the gdf_dtype enum.  The remaining template 
 * parameters must be able to be automatically deduced. 
 *
 * Example usage with standalone functor that returns the size of the dispatched type:
 *
 * struct example_functor{
 *  template <typename type_info>
 *  int operator()(){
 *    using T = typename type_info::type; // Retrieve underlying type from type_info
 *    return sizeof(T);
 *  }
 * };
 *
 * gdf_type_dispatcher(GDF_INT8, example_functor);  // returns 1
 * gdf_type_dispatcher(GDF_INT64, example_functor); // returns 8
 *
 * Example usage with a "template lambda" that returns size of the dispatched type:
 *
 * template <typename type_info>
 * auto example_lambda = []{ return sizeof(typename type_info::type); };
 *
 * gdf_type_dispatcher(GDF_INT8, example_lambda);  // returns 1
 * gdf_type_dispatcher(GDF_INT64, example_lambda); // returns 8
 *
 * NOTE: "template lambdas" can only be declared in namespace scope, i.e., outside
 * the scope of a function. Furthermore, they can only be *host* lambdas. As of 
 * CUDA 10, nvcc does not support templated, extended device lambdas.
 *
 * The return type for all template instantiations of the functor's "operator()" 
 * or the templated lambda must be the same, otherwise there will be a compiler 
 * error as you would be  trying to return different types from the same function.
 *
 * @Param dtype The gdf_dtype enum that determines which type will be dispatched
 * @Param f The functor with a templated "operator()" that will be invoked with 
 * the dispatched type
 * @Param args A parameter-pack (i.e., arbitrary number of arguments) that will 
 * be perfectly-forwarded as the arguments of the functor's "operator()".
 * @tparam enum_map A traits struct templated on a gdf_dtype value with a member 
 * type called "type" that maps a gdf_dtype to the underlying datatype.
 *
 * @Returns Whatever is returned by the functor's "operator()". 
 *
 */
/* ----------------------------------------------------------------------------*/
// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
namespace gdf{

#pragma hd_warning_disable
template < template <gdf_dtype> typename enum_map = default_enum_map, 
           class functor_t, 
           typename... Ts>
__host__ __device__ __forceinline__
decltype(auto) type_dispatcher(gdf_dtype dtype, 
                               functor_t f, 
                               Ts&&... args)
{

  switch(dtype)
  {
    // The .template is known as a "template disambiguator" 
    // See here for more information: https://stackoverflow.com/questions/3786360/confusing-template-error
    case GDF_INT8:      { return f.template operator()< enum_map<GDF_INT8> >(std::forward<Ts>(args)...); }
    case GDF_INT16:     { return f.template operator()< enum_map<GDF_INT16> >(std::forward<Ts>(args)...); }
    case GDF_INT32:     { return f.template operator()< enum_map<GDF_INT32> >(std::forward<Ts>(args)...); }
    case GDF_INT64:     { return f.template operator()< enum_map<GDF_INT64> >(std::forward<Ts>(args)...); }
    case GDF_FLOAT32:   { return f.template operator()< enum_map<GDF_FLOAT32> >(std::forward<Ts>(args)...); }
    case GDF_FLOAT64:   { return f.template operator()< enum_map<GDF_FLOAT64> >(std::forward<Ts>(args)...); }
    case GDF_DATE32:    { return f.template operator()< enum_map<GDF_DATE32> >(std::forward<Ts>(args)...); }
    case GDF_DATE64:    { return f.template operator()< enum_map<GDF_DATE64> >(std::forward<Ts>(args)...); }
    case GDF_TIMESTAMP: { return f.template operator()< enum_map<GDF_TIMESTAMP> >(std::forward<Ts>(args)...); }
    case GDF_CATEGORY:  { return f.template operator()< enum_map<GDF_CATEGORY> >(std::forward<Ts>(args)...); }
    //case GDF_STRING:    { return f.template operator()< enum_map<GDF_STRING> >(std::forward<Ts>(args)...); }
  }

  // This will only fire with a DEBUG build
  assert(0 && "type_dispatcher: invalid gdf_dtype");

  // Need to find out what the return type is in order to have a default return value
  // and solve the compiler warning for lack of a default return
  using return_type = decltype(f.template operator()<enum_map<GDF_INT8>>(std::forward<Ts>(args)...));
  return return_type();

}

} // namespace gdf

#endif
