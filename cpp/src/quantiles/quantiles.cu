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

#include "quantiles.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "utilities/type_dispatcher.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include "cudf.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace{ //unknown
  template<typename VType,
           typename RetT = double>
    void f_quantile_tester(rmm::device_vector<VType>& d_in)
  {
    using FctrType = std::function<RetT(VType, VType, double)>;

    FctrType lin_interp{[](VType y0, VType y1, double x){
        return static_cast<RetT>(static_cast<double>(y0) + x*static_cast<double>(y1-y0));//(f(x) - y0) / (x - 0) = m = (y1 - y0)/(1 - 0)
      }};

    FctrType midpoint{[](VType y0, VType y1, double x){
        return static_cast<RetT>(static_cast<double>(y0 + y1)/2.0);
      }};

    FctrType nearest{[](VType y0, VType y1, double x){
        return static_cast<RetT>(x < 0.5 ? y0 : y1);
      }};

    FctrType lowest{[](VType y0, VType y1, double x){
        return static_cast<RetT>(y0);
      }};

    FctrType highest{[](VType y0, VType y1, double x){
        return static_cast<RetT>(y1);
      }};
  
  
    std::vector<std::string> methods{"lin_interp", "midpoint", "nearest", "lowest", "highest"};
    size_t n_methods = methods.size();
    std::vector<FctrType> vf{lin_interp, midpoint, nearest, lowest, highest};
  
    std::vector<double> qvals{0.0, 0.25, 0.33, 0.5, 1.0};

  
    assert( n_methods == methods.size() );
  
    for(auto q: qvals)
      {
        VType res = quantile_approx(d_in.data().get(), d_in.size(), q);
        std::cout<<"q: "<<q<<"; exact res: "<<res<<"\n";
        for(auto i = 0;i<n_methods;++i)
          {
            RetT rt = quantile_exact(d_in.data().get(), d_in.size(), q, vf[i]);
            std::cout<<"q: "<<q<<"; method: "<<methods[i]<<"; rt: "<<rt<<"\n";
          }
      }
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

  template<typename ColType>
  void trampoline_approx(gdf_column*  col_in,
                         double q,
                         void* t_erased_res,
                         gdf_context* ctxt)
  {
    ColType* ptr_res = static_cast<ColType*>(t_erased_res);
    size_t n = col_in->size;
    ColType* p_dv = static_cast<ColType*>(col_in->data);
    if( ctxt->flag_sort_inplace || ctxt->flag_sorted )
      {
        *ptr_res = quantile_approx(p_dv, n, q, NULL, ctxt->flag_sorted);
      }
    else
      {
        rmm::device_vector<ColType> dv(n);
        thrust::copy_n(thrust::device, /*TODO: stream*/p_dv, n, dv.begin());
        cudaDeviceSynchronize();
        p_dv = dv.data().get();

        *ptr_res = quantile_approx(p_dv, n, q, NULL, ctxt->flag_sorted);
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
      return trampoline_exact<T>(col_in, prec, q, t_erased_res, ctxt);
    }
  };

  struct trampoline_approx_functor{
    template <typename T>
    void operator()(gdf_column*  col_in,       //input column;
                    double       q,            //requested quantile in [0,1]
                    void*        t_erased_res, //type-erased result of same type as column;
                    gdf_context* ctxt)         //context info)
    {
      trampoline_approx<T>(col_in, q, t_erased_res, ctxt);
    }
  };

}//unknown namespace

gdf_error gdf_quantile_exact(	gdf_column*         col_in,       //input column;
                              gdf_quantile_method prec,         //precision: type of quantile method calculation
                              double              q,            //requested quantile in [0,1]
                              void*               t_erased_res, //result; for <exact> should probably be double*; it's void* because
                                                                //(1) for uniformity of interface with <approx>;
                                                                //(2) for possible types bigger than double, in the future;
                              gdf_context*        ctxt)         //context info
{
  GDF_REQUIRE(!col_in->valid || !col_in->null_count, GDF_VALIDITY_UNSUPPORTED);
  gdf_error ret = GDF_SUCCESS;
  assert( col_in->size > 0 );

  ret = cudf::type_dispatcher(col_in->dtype,
                              trampoline_exact_functor{},
                              col_in, prec, q, t_erased_res, ctxt);

  return ret;
}

gdf_error gdf_quantile_approx(	gdf_column*  col_in,       //input column;
                                double       q,            //requested quantile in [0,1]
                                void*        t_erased_res, //type-erased result of same type as column;
                                gdf_context* ctxt)         //context info
{
  GDF_REQUIRE(!col_in->valid || !col_in->null_count, GDF_VALIDITY_UNSUPPORTED);
  gdf_error ret = GDF_SUCCESS;
  assert( col_in->size > 0 );

  cudf::type_dispatcher(col_in->dtype,
                        trampoline_approx_functor{},
                        col_in, q, t_erased_res, ctxt);
  
  return ret;
}

