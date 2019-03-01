/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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
#include "utilities//type_dispatcher.hpp"
#include "utilities/cudf_utils.h"

namespace{ //anonymous

  constexpr int BLOCK_SIZE = 256;

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Kernel that replaces elements from `d_col_data` given the following
   *        rule: replace all `old_values[i]` in [old_values_begin`, `old_values_end`)
   *        present in `d_col_data` with `d_new_values[i]`.
   * 
   * @param[in,out] d_col_data Device array with the data to be modified
   * @param[in] nrows # rows in `d_col_data`
   * @param[in] old_values_begin Device pointer to the beginning of the sequence 
   * of old values to be replaced
   * @param[in] old_values_end  Device pointer to the end of the sequence 
   * of old values to be replaced
   * @param[in] d_new_values Device array with the new values
   * 
   * @returns
   */
  /* ----------------------------------------------------------------------------*/
  template <class T>
  __global__
  void replace_kernel(T*                          d_col_data,
                      gdf_size_type                      nrows,
                      thrust::device_ptr<const T> old_values_begin,
                      thrust::device_ptr<const T> old_values_end,
                      const T*                    d_new_values)
  {
    gdf_size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < nrows)
    {
      auto found_ptr = thrust::find(thrust::seq, old_values_begin, old_values_end, d_col_data[i]);

      if (found_ptr != old_values_end) {
          auto d = thrust::distance(old_values_begin, found_ptr);
          d_col_data[i] = d_new_values[d];
      }

      i += blockDim.x * gridDim.x;
    }
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
   *        `replace_kernel` with the apropiate data types.
   */
  /* ----------------------------------------------------------------------------*/
  struct replace_kernel_forwarder {
    template <typename col_type>
    void operator()(void*       d_col_data,
                    gdf_size_type      nrows,
                    const void* d_old_values,
                    const void* d_new_values,
                    gdf_size_type      nvalues)
    {
      thrust::device_ptr<const col_type> old_values_begin = thrust::device_pointer_cast(static_cast<const col_type*>(d_old_values));

      const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);
      replace_kernel<<<grid_size, BLOCK_SIZE>>>(static_cast<col_type*>(d_col_data),
                                             nrows,
                                             old_values_begin,
                                             old_values_begin + nvalues,
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
    GDF_REQUIRE(col->valid == nullptr || col->null_count == 0, GDF_VALIDITY_UNSUPPORTED);
    GDF_REQUIRE(old_values->valid == nullptr || old_values->null_count == 0, GDF_VALIDITY_UNSUPPORTED);
    GDF_REQUIRE(new_values->valid == nullptr || new_values->null_count == 0, GDF_VALIDITY_UNSUPPORTED);

    
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
 * @brief Replace elements from `col` according to the mapping `old_values` to
 *        `new_values`, that is, replace all `old_values[i]` present in `col` 
 *        with `new_values[i]`.
 * 
 * @param[in,out] col gdf_column with the data to be modified
 * @param[in] old_values gdf_column with the old values to be replaced
 * @param[in] new_values gdf_column with the new values
 * 
 * @returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_find_and_replace_all(gdf_column*       col,
                                   const gdf_column* old_values,
                                   const gdf_column* new_values)
{
  return find_and_replace_all(col, old_values, new_values);
}

namespace{ //anonymous

template <typename Type>
__global__
void replace_nulls_with_scalar(gdf_size_type size, Type* out_data, gdf_valid_type * out_valid, const Type *in_data_scalar) 
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    out_data[i] = gdf_is_valid(out_valid, i)? out_data[i] : *in_data_scalar;
  }
}


template <typename Type>
__global__
void replace_nulls_with_column(gdf_size_type size, Type* out_data, gdf_valid_type* out_valid, const Type *in_data) 
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    out_data[i] = gdf_is_valid(out_valid, i)? out_data[i] : in_data[i];
  }
}


/* --------------------------------------------------------------------------*/
/** 
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the apropiate data types.
 */
/* ----------------------------------------------------------------------------*/
struct replace_nulls_kernel_forwarder {
  template <typename col_type>
  void operator()(gdf_size_type           nrows,
                  gdf_size_type           new_values_length,
                  void*            d_col_data,
                  gdf_valid_type*  d_col_valid,
                  const void*      d_new_value)
  {
    const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);
    if (new_values_length == 1) {
      replace_nulls_with_scalar<<<grid_size, BLOCK_SIZE>>>(nrows,
                                            static_cast<col_type*>(d_col_data),
                                            (d_col_valid),
                                            static_cast<const col_type*>(d_new_value)
                                            );
    } else if(new_values_length == nrows) {
      replace_nulls_with_column<<<grid_size, BLOCK_SIZE>>>(nrows,
                                            static_cast<col_type*>(d_col_data),
                                            (d_col_valid),
                                            static_cast<const col_type*>(d_new_value)
                                            );
      
    }
  }
};

} //end anonymous namespace

gdf_error gdf_replace_nulls(gdf_column* col_out, const gdf_column* col_in)
{
  GDF_REQUIRE(col_out->dtype == col_in->dtype, GDF_DTYPE_MISMATCH);
  GDF_REQUIRE(col_in->size == 1 || col_in->size == col_out->size, GDF_COLUMN_SIZE_MISMATCH);
   
  GDF_REQUIRE(nullptr != col_in->data, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != col_out->data, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr == col_in->valid || 0 == col_in->null_count, GDF_VALIDITY_UNSUPPORTED);

  cudf::type_dispatcher(col_out->dtype, replace_nulls_kernel_forwarder{},
                        col_out->size,
                        col_in->size,
                        col_out->data,
                        col_out->valid,
                        col_in->data);
  return GDF_SUCCESS;
}



