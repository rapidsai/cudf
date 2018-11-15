#ifndef GDF_TYPE_DISPATCHER_H
#define GDF_TYPE_DISPATCHER_H

#include <utility>
#include <gdf/cffi/types.h>

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Default mapping of gdf_dtype enums to corresponding C++ datatypes
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

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Mapping of gdf_dtype enum that can be used when one only cares about
 * using a type with the appropriate byte-width of the underlying C++ datatype.
 *
 * For example, this could be useful if one is simply moving the data in a column
 * and one doesn't care what the underlying datatype really is. This can save on 
 * compile time by reducing the number of template instantiations of the templated
 * functor/lambda that is called by the dispatcher.
 * 
 */
/* ----------------------------------------------------------------------------*/
template<gdf_dtype t> struct byte_width_map;
template <> struct byte_width_map<GDF_INT8>{ using type = uint8_t; };
template <> struct byte_width_map<GDF_INT16>{ using type = uint16_t; };
template <> struct byte_width_map<GDF_INT32>{ using type = uint32_t; };
template <> struct byte_width_map<GDF_INT64>{ using type = uint64_t; };
template <> struct byte_width_map<GDF_FLOAT32>{ using type = uint32_t; };
template <> struct byte_width_map<GDF_FLOAT64>{ using type = uint64_t; };
template <> struct byte_width_map<GDF_DATE32>{ using type = uint32_t; };
template <> struct byte_width_map<GDF_DATE64>{ using type = uint64_t; };
template <> struct byte_width_map<GDF_TIMESTAMP>{ using type = uint64_t; };
template <> struct byte_width_map<GDF_CATEGORY>{ using type = uint32_t; };

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Invokes a templated functor by dispatching the appropriate type from 
 * a gdf_dtype enum value.
 *
 * This helper function accepts any functor with an "operator()" template or a 
 * templated lambda. The template may have 1 or more template parameters, but 
 * the first parameter must be of the type dispatched from the  gdf_dtype enum. 
 * The remaining template parameters must be able to be automatically deduced. 
 *
 * Example usage with standalone functor that returns the size of the dispatched type:
 *
 * struct example_functor{
 *  template <typename T>
 *  int operator()(){
 *    return sizeof(T);
 *  }
 * };
 *
 * gdf_type_dispatcher(GDF_INT8, example_functor);  // returns 1
 * gdf_type_dispatcher(GDF_INT64, example_functor); // returns 8
 *
 * Example usage with a "template lambda" that returns size of the dispatched type:
 *
 * template <typename T>
 * auto example_lambda = []{ return sizeof(T); };
 *
 * gdf_type_dispatcher(GDF_INT8, example_lambda);  // returns 1
 * gdf_type_dispatcher(GDF_INT64, example_lambda); // returns 8
 *
 * NOTE: "template lambdas" can only be declared in namespace scope, i.e., outside
 * the scope of a function.
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
 * @tparam enum_map A templated structure that maps a gdf_dtype enum to a C++ type
 *
 * @Returns Whatever is returned by the functor's "operator()". 
 *
 */
/* ----------------------------------------------------------------------------*/
// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#pragma hd_warning_disable
template < template <gdf_dtype> typename enum_map = default_enum_map, 
           class functor_t, 
           typename... Ts>
__host__ __device__ __forceinline__
decltype(auto) gdf_type_dispatcher(gdf_dtype dtype, 
                                   functor_t f, 
                                   Ts&&... args)
{
  switch(dtype)
  {
    // The .template is known as a "template disambiguator" 
    // See here for more information: https://stackoverflow.com/questions/3786360/confusing-template-error
    case GDF_INT8:      { return f.template operator()<typename enum_map<GDF_INT8>::type>(std::forward<Ts>(args)...); }
    case GDF_INT16:     { return f.template operator()<typename enum_map<GDF_INT16>::type>(std::forward<Ts>(args)...); }
    case GDF_INT32:     { return f.template operator()<typename enum_map<GDF_INT32>::type>(std::forward<Ts>(args)...); }
    case GDF_INT64:     { return f.template operator()<typename enum_map<GDF_INT64>::type>(std::forward<Ts>(args)...); }
    case GDF_FLOAT32:   { return f.template operator()<typename enum_map<GDF_FLOAT32>::type>(std::forward<Ts>(args)...); }
    case GDF_FLOAT64:   { return f.template operator()<typename enum_map<GDF_FLOAT64>::type>(std::forward<Ts>(args)...); }
    case GDF_DATE32:    { return f.template operator()<typename enum_map<GDF_DATE32>::type>(std::forward<Ts>(args)...); }
    case GDF_DATE64:    { return f.template operator()<typename enum_map<GDF_DATE64>::type>(std::forward<Ts>(args)...); }
    case GDF_TIMESTAMP: { return f.template operator()<typename enum_map<GDF_TIMESTAMP>::type>(std::forward<Ts>(args)...); }
    case GDF_CATEGORY:  { return f.template operator()<typename enum_map<GDF_CATEGORY>::type>(std::forward<Ts>(args)...); }
  }
  // Need to find out what the return type is in order to have a default return value
  // and solve the compiler warning for lack of a default return
  using return_type = decltype(f.template operator()<typename enum_map<GDF_INT8>::type>(std::forward<Ts>(args)...));
  return return_type();
}

#endif
