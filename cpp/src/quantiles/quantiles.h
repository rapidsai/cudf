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

#include "cudf.h"

#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>

#include <vector>
#include <cassert>
#include <algorithm>
#include <functional>
#include <rmm/thrust_rmm_allocator.h>

struct QuantiledIndex {
  size_t lower_bound;
  size_t upper_bound;
  size_t nearest;
  double fraction;
};

QuantiledIndex find_quantile_index(size_t length, double quant);


template<typename T>
size_t quantile_index(T* dv, size_t n, double q, double& fract_pos, cudaStream_t stream, bool flag_sorted)
{
  if( !flag_sorted )
    thrust::sort(rmm::exec_policy(stream)->on(stream), dv, dv+n);

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
      T* d_res = thrust::max_element(rmm::exec_policy(stream)->on(stream), dv, dv+n);
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


