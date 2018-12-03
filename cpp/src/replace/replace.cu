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
#include <thrust/execution_policy.h>

#include "cudf.h"
#include "utilities/error_utils.h"
#include "dataframe/type_dispatcher.hpp"

namespace{ //anonymous

  constexpr int BLOCK_SIZE = 256;

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

      // TODO: calc blocks and threads
      const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);
      replace_kernel<<<grid_size, BLOCK_SIZE>>>(static_cast<col_type*>(d_col_data),
                                             nrows,
                                             old_values_begin,
                                             old_values_end,
                                             static_cast<const col_type*>(d_new_values));
    }
 };

  /// /brief Replace kernel
  /// \param[in/out] data with elements to be replaced
  /// \param[in] values contains the replacement values
  /// \param[in] to_replace_begin begin pointer of `to_replace` array
  /// \param[in] to_replace_begin end pointer of `to_replace` array
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
      // TODO: find by map kernel
      auto found_ptr = thrust::find(thrust::device, old_values_begin, old_values_end, d_col_data[i]);

      if (found_ptr != to_replace_end) {
          auto d = thrust::distance(old_values_begin, found_ptr);
          d_col_data[i] = d_new_values[d];
      }

      i += blockDim.x * gridDim.x
    }
  }

  /// /brief Call replace kernel according to primitive type T
  /// \param[in/out] data with elements to be replaced
  /// \param[in] data_size number of elements in data
  /// \param[in] to_replace contains values that will be replaced
  /// \param[in] values contains the replacement values
  /// \param[in] replacement_ptrdiff to get the end pointer of `to_replace` array
  gdf_error find_and_replace_all(gdf_column*       d_col,
                                 const gdf_column* d_old_values,
                                 const gdf_column* d_new_values)
  {
    GDF_REQUIRE(d_col != nullptr && d_old_values != nullptr && d_new_values != nullptr, GDF_DATASET_EMPTY);
    GDF_REQUIRE(d_old_values->size == d_new_values->size, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(d_col->dtype == d_old_values->dtype && d_col->dtype == d_new_values->dtype, GDF_DTYPE_MISMATCH);
    
    cudf::type_dispatcher(d_col->dtype, replace_kernel_forwarder{},
                          d_col->data,
                          d_col->size,
                          d_old_values->data,
                          d_new_values->data,
                          d_new_values->size); 

    return GDF_SUCCESS;
  }

} //end anonymous namespace

/// \brief For each value in `to_replace`, find all instances of that value
///        in `column` and replace it with the corresponding value in `values`.
/// \param[in/out] column data
/// \param[in] to_replace contains values of column that will be replaced
/// \param[in] values contains the replacement values
///
/// Note that `to_replace` and `values` are related by the index
gdf_error gdf_find_and_replace_all(gdf_column*       d_col,
                                   const gdf_column* d_old_values,
                                   const gdf_column* d_new_values)
{
  return find_and_replace_all(d_col, d_old_values, d_new_values);
}
