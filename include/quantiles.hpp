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

#pragma once

//Quantile (percentile) functionality

#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>

#include <vector>
#include <cassert>
#include <algorithm>
#include <functional>


template<typename T>
size_t quantile_index(T* dv, size_t n, double q, double& fract_pos, cudaStream_t stream, bool flag_sorted)
{
  if( !flag_sorted )
    thrust::sort(thrust::cuda::par.on(stream), dv, dv+n);

  double pos = q*static_cast<double>(n);//(n-1);
  size_t k = static_cast<size_t>(pos);
  
  fract_pos = pos - static_cast<double>(k);
  if( k > 0 )//using n and (k-1) gives more intuitive result than using (n-1) and k
    --k;

  return k;
}

template<typename T>
T quantile_approx(T* dv, size_t n, double q, cudaStream_t stream = NULL, bool flag_sorted = false)
{
  std::vector<T> hv(2);
  if( q >= 1.0 )
    {
      T* d_res = thrust::max_element(thrust::device, dv, dv+n);
      cudaMemcpy(&hv[0], d_res, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
      return hv[0];
    }

  if( n < 2 )
    {
      cudaMemcpy(&hv[0], dv, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
      return hv[0];
    }

   double fract_pos = 0;
   size_t k = quantile_index(dv, n, q, fract_pos, stream, flag_sorted);

   cudaMemcpy(&hv[0], dv+k, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
   return hv[0];
}

template<typename T,
         typename RetT>
RetT quantile_exact(T* dv, size_t n, double q, std::function<RetT(T, T, double)>& e, cudaStream_t stream = NULL, bool flag_sorted = false)
{
  std::vector<T> hv(2);
  if( q >= 1.0 )
    {
      T* d_res = thrust::max_element(thrust::device, dv, dv+n);
      cudaMemcpy(&hv[0], d_res, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
      return hv[0];
    }

  if( n < 2 )
    {
      cudaMemcpy(&hv[0], dv, sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
      return hv[0];
    }
    
  double fract_pos = 0;
  
  size_t k = quantile_index(dv, n, q, fract_pos, stream, flag_sorted);
  assert( fract_pos >= 0 && fract_pos < 1);

  cudaMemcpy(&hv[0], dv+k, 2*sizeof(T), cudaMemcpyDeviceToHost);//TODO: async with streams?
  RetT val = e(hv[0], hv[1], fract_pos);
  return val;
}

template<typename VType,
         typename RetT = double> // just in case double won't be enough to hold result, in the future
gdf_error select_quantile(VType* dv,
                          size_t n,
                          double q, 
                          gdf_quantile_method prec,
                          RetT& result,
                          bool flag_sorted = false,
                          cudaStream_t stream = NULL)
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
  FctrType fctr;
  switch( prec )
    {
    case GDF_QUANT_LINEAR:
      fctr = lin_interp;
      break;
        
    case GDF_QUANT_LOWER:
      fctr = lowest;
      break;
        
    case GDF_QUANT_HIGHER:
      fctr = highest;
      break;
        
    case GDF_QUANT_MIDPOINT:
      fctr = midpoint;
      break;
        
    case GDF_QUANT_NEAREST:
      fctr = nearest;
      break;

    default:
      return GDF_UNSUPPORTED_METHOD;
    }
  result = quantile_exact(dv, n, q, fctr, stream, flag_sorted);
  return GDF_SUCCESS;
}
