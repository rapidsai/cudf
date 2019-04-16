/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

//Quantile (percentile) functionality

#include "quantiles.hpp"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "utilities/type_dispatcher.hpp"
#include "utilities/wrapper_types.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include "cudf.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace cudf {
namespace interpolate {

QuantiledIndex find_quantile_index(gdf_size_type length, double quant)
{
    // clamp quant value.
    // Todo: use std::clamp if c++17 is supported.
    quant = std::min(std::max(quant, 0.0), 1.0);
    double val = quant*(length -1);
    QuantiledIndex qi;
    qi.lower_bound = std::floor(val);
    qi.upper_bound = static_cast<size_t>(std::ceil(val));
    qi.nearest = static_cast<size_t>(std::nearbyint(val));
    qi.fraction = val - qi.lower_bound;

    return qi;
}

}
}


namespace{ //unknown

  template<typename T, typename RetT>
  gdf_error select_quantile(T* dv,
                          gdf_size_type n,
                          double q, 
                          gdf_quantile_method interpolation,
                          RetT& result,
                          bool flag_sorted = false,
                          cudaStream_t stream = NULL)
  {
    std::vector<T> hv(2);

    if( q >= 1.0 && flag_sorted )
    {
      T* d_res = thrust::max_element(rmm::exec_policy(stream)->on(stream), dv, dv+n);
      cudaMemcpy(&hv[0], d_res, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
      result = static_cast<RetT>( hv[0] );
      return GDF_SUCCESS;
    }

    if( q <= 0.0 && flag_sorted )
    {
      T* d_res = thrust::min_element(rmm::exec_policy(stream)->on(stream), dv, dv+n);
      cudaMemcpy(&hv[0], d_res, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
      result = static_cast<RetT>( hv[0] );
      return GDF_SUCCESS;
    }

    if( n < 2 )
    {
      cudaMemcpy(&hv[0], dv, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
      result = static_cast<RetT>( hv[0] );
      return GDF_SUCCESS;
    }

    // sort if the input is not sorted.
    if( !flag_sorted ){
      thrust::sort(rmm::exec_policy(stream)->on(stream), dv, dv+n);
    }

    cudf::interpolate::QuantiledIndex qi = cudf::interpolate::find_quantile_index(n, q);

    switch( interpolation )
    {
    case GDF_QUANT_LINEAR:
      cudaMemcpy(&hv[0], dv+qi.lower_bound, sizeof(T), cudaMemcpyDeviceToHost); //TODO: async with streams
      cudaMemcpy(&hv[1], dv+qi.upper_bound, sizeof(T), cudaMemcpyDeviceToHost); //TODO: async with streams
      cudf::interpolate::linear(result, cudf::detail::unwrap(hv[0]), cudf::detail::unwrap(hv[1]), qi.fraction);
      break;
    case GDF_QUANT_MIDPOINT:
      cudaMemcpy(&hv[0], dv+qi.lower_bound, sizeof(T), cudaMemcpyDeviceToHost); //TODO: async with streams
      cudaMemcpy(&hv[1], dv+qi.upper_bound, sizeof(T), cudaMemcpyDeviceToHost); //TODO: async with streams
      cudf::interpolate::midpoint(result, cudf::detail::unwrap(hv[0]), cudf::detail::unwrap(hv[1]));
      break;
    case GDF_QUANT_LOWER:
      cudaMemcpy(&hv[0], dv+qi.lower_bound, sizeof(T), cudaMemcpyDeviceToHost); //TODO: async with streams
      result = static_cast<RetT>( hv[0] );
      break;
    case GDF_QUANT_HIGHER:
      cudaMemcpy(&hv[0], dv+qi.upper_bound, sizeof(T), cudaMemcpyDeviceToHost); //TODO: async with streams
      result = static_cast<RetT>( hv[0] );
      break;
    case GDF_QUANT_NEAREST:
      cudaMemcpy(&hv[0], dv+qi.nearest, sizeof(T), cudaMemcpyDeviceToHost); //TODO: async with streams
      result = static_cast<RetT>( hv[0] );
      break;

    default:
      return GDF_UNSUPPORTED_METHOD;
    }
    
    return GDF_SUCCESS;
 }

  template<typename ColType,
           typename RetT = double> // just in case double won't be enough to hold result, in the future
  gdf_error trampoline_exact(gdf_column*  col_in,
                             gdf_quantile_method prec,
                             double q,
                             void* t_erased_res,
                             gdf_context* ctxt)
  {
    RetT* ptr_res = static_cast<RetT*>(t_erased_res);
    size_t n = col_in->size;
    ColType* p_dv = static_cast<ColType*>(col_in->data);
    
    if( ctxt->flag_sort_inplace || ctxt->flag_sorted)
      {
        return select_quantile(p_dv,
                               n,
                               q, 
                               prec,
                               *ptr_res,
                               ctxt->flag_sorted);
      }
    else
      {
        rmm::device_vector<ColType> dv(n);
        thrust::copy_n(thrust::device, /*TODO: stream*/p_dv, n, dv.begin());
        cudaDeviceSynchronize();
        p_dv = dv.data().get();

        return select_quantile(p_dv,
                               n,
                               q, 
                               prec,
                               *ptr_res,
                               ctxt->flag_sorted);
      }
  }
    
  struct trampoline_exact_functor{
    template <typename T,
              typename std::enable_if_t<!std::is_arithmetic<T>::value, int> = 0>
    gdf_error operator()(gdf_column* col_in,
                         gdf_quantile_method prec,
                         double q,
                         void* t_erased_res,
                         gdf_context* ctxt)
    {
      return GDF_UNSUPPORTED_DTYPE;
    }

    template <typename T,
              typename std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
    gdf_error operator()(gdf_column*  col_in,              //input column;
                         gdf_quantile_method prec,         //precision: type of quantile method calculation
                         double              q,            //requested quantile in [0,1]
                         void*               t_erased_res, //result; for <exact> should probably be double*; it's void* because
                                                           //(1) for uniformity of interface with <approx>;
                                                           //(2) for possible types bigger than double, in the future;
                         gdf_context*        ctxt)         //context info
    {
      return trampoline_exact<T, double>
                 (col_in, prec, q, t_erased_res, ctxt);
    }
  };

  struct trampoline_approx_functor{
    template <typename UnderLyingT>
    gdf_error call(gdf_column*  col_in,
                    double       q,
                    void*        t_erased_res,
                    gdf_context* ctxt,
                    UnderLyingT* data)
    {
      return trampoline_exact<UnderLyingT, UnderLyingT>
                 (col_in, GDF_QUANT_LOWER, q, t_erased_res, ctxt);
    }

    template <typename T>
    gdf_error operator()(gdf_column*  col_in,       //input column;
                    double       q,            //requested quantile in [0,1]
                    void*        t_erased_res, //type-erased result of same type as column;
                    gdf_context* ctxt)         //context info)
    { 
      // ToDo: remove calling "call" function with UnderLying type and directly call "trampoline_exact"
      // without this, `select_quantile` cannnot be compiled for wrapper types
      // since double * wrapper types is not defined.
      // The error occurs at GDF_QUANT_LINEAR and GDF_QUANT_MIDPOINT cases 
      // even though this call won't get in.
      T* data = static_cast<T*>( col_in->data );
      return call(col_in, q, t_erased_res, ctxt, &cudf::detail::unwrap(data[0]));
    }


  };

}//unknown namespace

gdf_error gdf_quantile_exact( gdf_column*         col_in,       //input column;
                              gdf_quantile_method prec,         //precision: type of quantile method calculation
                              double              q,            //requested quantile in [0,1]
                              gdf_scalar*         result,       // the result
                              gdf_context*        ctxt)         //context info
{
  GDF_REQUIRE(!col_in->valid || !col_in->null_count, GDF_VALIDITY_UNSUPPORTED);
  gdf_error ret = GDF_SUCCESS;
  assert( col_in->size > 0 );

  result->dtype = GDF_FLOAT64;
  result->is_valid = false; // the scalar is not valid for error case

  ret = cudf::type_dispatcher(col_in->dtype,
                              trampoline_exact_functor{},
                              col_in, prec, q, &result->data, ctxt);

  if( ret == GDF_SUCCESS ) result->is_valid = true;
  return ret;
}

gdf_error gdf_quantile_approx(	gdf_column*  col_in,       //input column;
                                double       q,            //requested quantile in [0,1]
                                gdf_scalar*  result,       // the result
                                gdf_context* ctxt)         //context info
{
  GDF_REQUIRE(!col_in->valid || !col_in->null_count, GDF_VALIDITY_UNSUPPORTED);
  gdf_error ret = GDF_SUCCESS;
  assert( col_in->size > 0 );

  result->dtype = col_in->dtype;
  result->is_valid = false; // the scalar is not valid for error case

  ret = cudf::type_dispatcher(col_in->dtype,
                              trampoline_approx_functor{},
                              col_in, q, &result->data, ctxt);
  
  if( ret == GDF_SUCCESS ) result->is_valid = true;
  return ret;
}

