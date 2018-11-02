#ifndef GDF_TYPE_DISPATCHER_H
#define GDF_TYPE_DISPATCHER_H

#include <utility>
#include <gdf/cffi/types.h>

#pragma hd_warning_disable
template <class functor_t, typename... Ts>
__host__ __device__ __forceinline__
decltype(auto) gdf_type_dispatcher(gdf_dtype dtype, functor_t f, Ts&&... args)
{
    switch(dtype)
    {
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
    return;
}

#endif
