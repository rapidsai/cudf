/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>

#include "cudf.h"
#include "utilities/error_utils.h"
#include "dataframe/type_dispatcher.hpp"

namespace{ //anonymous

  constexpr int BLOCK_SIZE = 256;
  
  template <class T>
  struct elements_are_equal{
    __device__
    elements_are_equal(T el) : element{el}
    {
    }

    __device__
    bool operator()(T other)
    { 
      return (cudf::detail::unwrap(other) == cudf::detail::unwrap(element));
    }

  private:
    T element;
  };

  template <class T>
  __global__
  void replace_kernel(T*                          d_col_data,
                      size_t                      nrows,
                      thrust::device_ptr<const T> old_values_begin,
                      thrust::device_ptr<const T> old_values_end,
                      const T*                    d_new_values)
  {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < nrows)
    {
      auto found_ptr = thrust::find_if(thrust::seq, old_values_begin, old_values_end, elements_are_equal<T>(d_col_data[i]));

      if (found_ptr != old_values_end) {
          auto d = thrust::distance(old_values_begin, found_ptr);
          d_col_data[i] = d_new_values[d];
      }

      i += blockDim.x * gridDim.x;
    }
  }

  struct replace_kernel_forwarder {
    template <typename col_type>
    void operator()(void*       d_col_data,
                    size_t      nrows,
                    const void* d_old_values,
                    const void* d_new_values,
                    size_t      nvalues)
    {
      thrust::device_ptr<const col_type> old_values_begin(static_cast<const col_type*>(d_old_values));
      thrust::device_ptr<const col_type> old_values_end(static_cast<const col_type*>(d_old_values) + nvalues);

      const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);
      replace_kernel<<<grid_size, BLOCK_SIZE>>>(static_cast<col_type*>(d_col_data),
                                             nrows,
                                             old_values_begin,
                                             old_values_end,
                                             static_cast<const col_type*>(d_new_values));
    }
  };

  gdf_error find_and_replace_all(gdf_column*       col,
                                 const gdf_column* old_values,
                                 const gdf_column* new_values)
  {
    GDF_REQUIRE(col != nullptr && old_values != nullptr && new_values != nullptr, GDF_DATASET_EMPTY);
    GDF_REQUIRE(old_values->size == new_values->size, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(col->dtype == old_values->dtype && col->dtype == new_values->dtype, GDF_DTYPE_MISMATCH);
    GDF_REQUIRE(col->valid == nullptr && col->null_count == 0, GDF_VALIDITY_UNSUPPORTED);
    
    cudf::type_dispatcher(col->dtype, replace_kernel_forwarder{},
                          col->data,
                          col->size,
                          old_values->data,
                          new_values->data,
                          new_values->size); 

    return GDF_SUCCESS;
  }

} //end anonymous namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief Replace elements from col according to the mapping old_values to new_values
 * 
 * @Param[in] col gdf_column with the data to be modified
 * @Param[in] old_values gdf_column with the old values to be replaced
 * @Param[in] new_values gdf_column with the new values
 * 
 * @Returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_find_and_replace_all(gdf_column*       col,
                                   const gdf_column* old_values,
                                   const gdf_column* new_values)
{
  return find_and_replace_all(col, old_values, new_values);
}
