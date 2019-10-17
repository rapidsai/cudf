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

#include <cudf/cudf.h>
#include <quantiles/quantiles_util.hpp>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>


namespace{ // anonymous

  // compute quantile value as `result` from `quant` value by `interpolation` method
  template<typename T, typename RetT>
  gdf_error select_quantile(T* devarr,
                          gdf_size_type n,
                          double quant, 
                          cudf::interpolation interpolation,
                          RetT& result,
                          bool flag_sorted,
                          cudaStream_t stream)
  {
    T hvalue;

    if( quant >= 1.0 && !flag_sorted )
    {
      T* d_res = thrust::max_element(rmm::exec_policy(stream)->on(stream), devarr, devarr+n);
      hvalue = cudf::detail::get_array_value(d_res, 0);
      result = static_cast<RetT>( hvalue );
      return GDF_SUCCESS;
    }

    if( quant <= 0.0 && !flag_sorted )
    {
      T* d_res = thrust::min_element(rmm::exec_policy(stream)->on(stream), devarr, devarr+n);
      hvalue = cudf::detail::get_array_value(d_res, 0);
      result = static_cast<RetT>( hvalue );
      return GDF_SUCCESS;
    }

    // sort if the input is not sorted.
    if( !flag_sorted ){
      thrust::sort(rmm::exec_policy(stream)->on(stream), devarr, devarr+n);
    }

    result = cudf::detail::select_quantile(devarr, n, quant, interpolation);
    
    return GDF_SUCCESS;
  }

  template<typename ColType,
           typename RetT = double>
  gdf_error trampoline_exact(gdf_column*  col_in,
                             cudf::interpolation interpolation,
                             double quant,
                             void* t_erased_res,
                             gdf_context* ctxt,
                             cudaStream_t stream)
  {
    RetT* ptr_res = static_cast<RetT*>(t_erased_res);
    size_t n = col_in->size;
    ColType* col_data = static_cast<ColType*>(col_in->data);
    
    if( ctxt->flag_sort_inplace  && ctxt->flag_sorted )
    {
      return select_quantile(col_data,
                             n,
                             quant, 
                             interpolation,
                             *ptr_res,
                             ctxt->flag_sorted,
                             stream);
    }else{
      // create a clone of col_data if sort is required but sort_inplace is not allowed.
      rmm::device_vector<ColType> dv(n);
      thrust::copy_n(rmm::exec_policy(stream)->on(stream), col_data, n, dv.begin());
      ColType* clone_data = dv.data().get();

      return select_quantile(clone_data,
                             n,
                             quant, 
                             interpolation,
                             *ptr_res,
                             ctxt->flag_sorted,
                             stream);
    }
  }
    
  struct trampoline_exact_functor{
    template <typename T,
              typename std::enable_if_t<!std::is_arithmetic<T>::value, int> = 0>
    gdf_error operator()(gdf_column* col_in,
                         cudf::interpolation interpolation,
                         double              quant,
                         void*               t_erased_res,
                         gdf_context*        ctxt,
                         cudaStream_t        stream = NULL)
    {
      return GDF_UNSUPPORTED_DTYPE;
    }

    template <typename T,
              typename std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
    gdf_error operator()(gdf_column*  col_in,
                         cudf::interpolation interpolation,
                         double              quant,
                         void*               t_erased_res,
                         gdf_context*        ctxt,
                         cudaStream_t        stream = NULL)
    {
      // just in case double won't be enough to hold result
      // it can be changed in future
      return trampoline_exact<T, double>
                 (col_in, interpolation, quant, t_erased_res, ctxt, stream);
    }
  };

  struct trampoline_approx_functor{
    template <typename T,
              typename std::enable_if_t<!std::is_arithmetic<T>::value, int> = 0>
    gdf_error operator()(gdf_column* col_in,
                         double              quant,
                         void*               t_erased_res,
                         gdf_context*        ctxt,
                         cudaStream_t        stream = NULL)
    {
      // TODO: support non-arithemetic types
      return GDF_UNSUPPORTED_DTYPE;
    }

    template <typename T,
              typename std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
    gdf_error operator()(gdf_column*  col_in, 
                    double       quant,
                    void*        t_erased_res,
                    gdf_context* ctxt,
                    cudaStream_t stream = NULL)
    {
      return trampoline_exact<T, T>(col_in, cudf::interpolation::LOWER, quant, t_erased_res, ctxt, stream);
    }
  };

} // end of anonymous

namespace cudf {

gdf_error quantile_exact( gdf_column*         col_in,       // input column
                          interpolation       prec,         // interpolation method
                          double              q,            // requested quantile in [0,1]
                          gdf_scalar*         result,       // the result
                          gdf_context*        ctxt)         // context info
{
  GDF_REQUIRE(nullptr != col_in, GDF_DATASET_EMPTY);

  if (col_in->size == 0) {
     result->is_valid = false;
     return GDF_SUCCESS;
  }

  GDF_REQUIRE(nullptr != col_in->data, GDF_DATASET_EMPTY);
  GDF_REQUIRE(0 < col_in->size, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr == col_in->valid || 0 == col_in->null_count, GDF_VALIDITY_UNSUPPORTED);

  gdf_error ret = GDF_SUCCESS;
  result->dtype = GDF_FLOAT64;
  result->is_valid = false; // the scalar is not valid for error case

  ret = cudf::type_dispatcher(col_in->dtype,
                              trampoline_exact_functor{},
                              col_in, prec, q, &result->data, ctxt);

  if( ret == GDF_SUCCESS ) result->is_valid = true;
  return ret;
}

gdf_error quantile_approx(	gdf_column*  col_in,       // input column
                            double       q,            // requested quantile in [0,1]
                            gdf_scalar*  result,       // the result
                            gdf_context* ctxt)         // context info
{
  GDF_REQUIRE(nullptr != col_in, GDF_DATASET_EMPTY);
  
  if (col_in->size == 0) {
     result->is_valid = false;
     return GDF_SUCCESS;
  }

  GDF_REQUIRE(nullptr != col_in->data, GDF_DATASET_EMPTY);
  GDF_REQUIRE(0 < col_in->size, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr == col_in->valid || 0 == col_in->null_count, GDF_VALIDITY_UNSUPPORTED);

  gdf_error ret = GDF_SUCCESS;
  result->dtype = col_in->dtype;
  result->is_valid = false; // the scalar is not valid for error case

  ret = cudf::type_dispatcher(col_in->dtype,
                              trampoline_approx_functor{},
                              col_in, q, &result->data, ctxt);
  
  if( ret == GDF_SUCCESS ) result->is_valid = true;
  return ret;
}

} // namespace cudf
