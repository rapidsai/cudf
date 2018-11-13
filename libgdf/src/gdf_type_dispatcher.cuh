#ifndef GDF_TYPE_DISPATCHER_H
#define GDF_TYPE_DISPATCHER_H

#include <utility>
#include <gdf/cffi/types.h>

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Invokes a templated functor by dispatching the appropriate type from 
 * a gdf_dtype enum value.
 *
 * This helper function accepts any *functor* (generic lambdas will also work, but
 * only in host code.) with an "operator()" template. The template may have 1 or more
 * template parameters, but the first parameter must be the type dispatched from the 
 * gdf_dtype enum. The remaining template arguments must be able to be automatically 
 * deduced. 
 * 
 * @Param dtype The gdf_dtype enum that determines which type will be dispatched
 * @Param f The functor with a templated "operator()" that will be invoked with 
 * the dispatched type
 * @Param args A parameter-pack (i.e., arbitrary number of arguments) that will 
 * be perfectly-forwarded as the arguments of the functor's "operator()".
 *
 * @Returns Whatever is returned by the functor's "operator()". 
 *
 * NOTE: The return type for all template instantiations of the functor's "operator()" 
 * must be the same, otherwise there will be a compiler error as you would be 
 * trying to return different types from the same function.
 */
/* ----------------------------------------------------------------------------*/
// This pragma disables a compiler warning that complains about the valid usage
// of calling a __host__ functor from this function which is __host__ __device__
#pragma hd_warning_disable
template <class functor_t, typename... Ts>
__host__ __device__ __forceinline__
decltype(auto) gdf_type_dispatcher(gdf_dtype dtype, functor_t f, Ts&&... args)
{
    switch(dtype)
    {
      // The .template is known as a "template disambiguator" 
      // See here for more information: https://stackoverflow.com/questions/3786360/confusing-template-error
      case GDF_INT8:      { return f.template operator()<int8_t>(std::forward<Ts>(args)...); }
      case GDF_INT16:     { return f.template operator()<int16_t>(std::forward<Ts>(args)...); }
      case GDF_INT32:     { return f.template operator()<int32_t>(std::forward<Ts>(args)...); }
      case GDF_INT64:     { return f.template operator()<int64_t>(std::forward<Ts>(args)...); }
      case GDF_FLOAT32:   { return f.template operator()<float>(std::forward<Ts>(args)...); }
      case GDF_FLOAT64:   { return f.template operator()<double>(std::forward<Ts>(args)...); }
      case GDF_DATE32:    { return f.template operator()<int32_t>(std::forward<Ts>(args)...); }
      case GDF_DATE64:    { return f.template operator()<int64_t>(std::forward<Ts>(args)...); }
      case GDF_TIMESTAMP: { return f.template operator()<int64_t>(std::forward<Ts>(args)...); }
      case GDF_CATEGORY:  { return f.template operator()<int32_t>(std::forward<Ts>(args)...); }
    }

    // Need to find out what the return type is in order to have a default return value
    // and solve the compiler warning for lack of a default return
    using return_type = decltype(f.template operator()<int8_t>(std::forward<Ts>(args)...));
    return return_type{};
}

#endif
