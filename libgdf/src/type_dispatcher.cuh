#ifndef GDF_TYPE_DISPATCHER_H
#define GDF_TYPE_DISPATCHER_H

#include <cassert>
#include <utility>
#include <gdf/cffi/types.h>
#include "types.cuh"
#include "NVStrings.h"

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Invokes an instance of a functor template with the appropriate type
 * determined by a gdf_dtype enum value.
 *
 * This helper function accepts any callable object with an "operator()" template,
 * e.g., a functor. It will invoke an instance of the template by passing 
 * in as the template argument an appropriate type determined by the value of the 
 * gdf_dtype argument.
 *
 * The template may have 1 or more template parameters, but the first parameter must 
 * be the type dispatched from the gdf_dtype enum.  The remaining template parameters 
 * must be able to be automatically deduced. 
 *
 * There is a 1-to-1 mapping of gdf_dtype enum values and dispatched types. However,
 * different gdf_dtype values may have the same underlying type. Therefore, in
 * order to provide the 1-to-1 mapping, a wrapper struct may be dispatched for certain
 * gdf_dtype enum values in order to emulate a "strong typedef". 
 *
 * A strong typedef  provides a new, concrete type, unlike a normal C++ typedef which
 * is simply a type alias. These "strong typedef" structs simply wrap a single member
 * variable of a fundamental type called 'value'. In order to access the underlying 
 * value, one must use the "unwrap" function which will provide a reference to the 
 * underlying value.
 *
 * See types.cuh for more detail.
 *
 * Example usage with a functor that returns the size of the dispatched type:
 *
 * struct example_functor{
 *  template <typename col_type>
 *  int operator()(){
 *    return sizeof(T);
 *  }
 * };
 *
 * gdf::type_dispatcher(GDF_INT8, example_functor);  // returns 1
 * gdf::type_dispatcher(GDF_INT64, example_functor); // returns 8
 *
 * Example usage of of the "unwrap" function in a functor for checking if element "i" 
 * in column "lhs" is equal to element "j" in column "rhs":
 *
 * struct elements_are_equal{
 *   template <typename col_type>
 *   bool operator()(void const * lhs, int i,
 *                   void const * rhs, int j)
 *   {
 *     // Cast the void* data buffer to the dispatched type and retrieve elements 
 *     // "i" and "j" from the respective columns
 *     col_type const i_elem = static_cast<col_type const*>(lhs)[i];
 *     col_type const j_elem = static_cast<col_type const*>(rhs)[j];
 *
 *     // "col_type" may be a wrapper struct. Therefore, use the "unwrap" function
 *     // to retrieve a reference to the underlying value. If "col_type" is a 
 *     // fundamental type, "unwrap" simply passes through the same value and
 *     // is effectively a no-op
 *     return gdf::detail::unwrap(i_elem) == gdf::detail::unwrap(j_elem);
 *   }
 * };
 *
 * The return type for all template instantiations of the functor's "operator()" 
 * lambda must be the same, otherwise there will be a compiler error as you would be
 * trying to return different types from the same function.
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
template < class functor_t, 
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
    case GDF_INT8:      { return f.template operator()< int8_t >(std::forward<Ts>(args)...); }
    case GDF_INT16:     { return f.template operator()< int16_t >(std::forward<Ts>(args)...); }
    case GDF_INT32:     { return f.template operator()< int32_t >(std::forward<Ts>(args)...); }
    case GDF_INT64:     { return f.template operator()< int64_t >(std::forward<Ts>(args)...); }
    case GDF_FLOAT32:   { return f.template operator()< float >(std::forward<Ts>(args)...); }
    case GDF_FLOAT64:   { return f.template operator()< double >(std::forward<Ts>(args)...); }
    case GDF_DATE32:    { return f.template operator()< date32 >(std::forward<Ts>(args)...); }
    case GDF_DATE64:    { return f.template operator()< date64 >(std::forward<Ts>(args)...); }
    case GDF_TIMESTAMP: { return f.template operator()< timestamp >(std::forward<Ts>(args)...); }
    case GDF_CATEGORY:  { return f.template operator()< category >(std::forward<Ts>(args)...); }
    //case GDF_STRING:    { return f.template operator()< enum_map<GDF_STRING> >(std::forward<Ts>(args)...); }
  }

  // This will only fire with a DEBUG build
  assert(0 && "type_dispatcher: invalid gdf_dtype");

  // Need to find out what the return type is in order to have a default return value
  // and solve the compiler warning for lack of a default return
  using return_type = decltype(f.template operator()<int8_t>(std::forward<Ts>(args)...));
  return return_type();
}

} // namespace gdf

#endif
