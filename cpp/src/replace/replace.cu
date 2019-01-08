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
#include "utilities//type_dispatcher.hpp"

namespace{ //anonymous

  constexpr int BLOCK_SIZE = 256;

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Kernel that replaces elements from `d_col_data` given the following
   *        rule: replace all `old_values[i]` in [old_values_begin`, `old_values_end`)
   *        present in `d_col_data` with `d_new_values[i]`.
   * 
   * @Param[in,out] d_col_data Device array with the data to be modified
   * @Param[in] nrows # rows in `d_col_data`
   * @Param[in] old_values_begin Device pointer to the beginning of the sequence 
   * of old values to be replaced
   * @Param[in] old_values_end  Device pointer to the end of the sequence 
   * of old values to be replaced
   * @Param[in] d_new_values Device array with the new values
   * 
   * @Returns
   */
  /* ----------------------------------------------------------------------------*/
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
                    size_t      nrows,
                    const void* d_old_values,
                    const void* d_new_values,
                    size_t      nvalues)
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
 * @Param[in,out] col gdf_column with the data to be modified
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

template <typename Type>
__global__
void replace_nulls_kernel(int size, Type* vax_data, uint32_t* vax_valid, const Type* vay_data) 
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    int index = i / warpSize;
    uint32_t position = i % warpSize;
    uint32_t is_vax_valid = vax_valid[index];

    uint32_t sel_vax = (is_vax_valid >> position) & 1;
    vax_data[i] = sel_vax? vax_data[i] : *vay_data;
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
  void operator()(size_t           nrows,
                  void*            d_col_data,
                  gdf_valid_type*  d_col_valid,
                  const void*      d_new_value)
  {
    const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);
    replace_nulls_kernel<<<grid_size, BLOCK_SIZE>>>(nrows,
                                          static_cast<col_type*>(d_col_data),
                                          (uint32_t*)(d_col_valid),
                                          static_cast<const col_type*>(d_new_value)
                                          );
  }
};

/* --------------------------------------------------------------------------*
 * @brief This function is a binary function. It will take in two gdf_columns.
 * The first one is expected to be a regular gdf_column, the second one
 * has to be a column of the same type as the first, and it has to be of
 * size one or of the same size as the other column.
 * 
 * case 1: If the second column contains only one value, then this funciton will
 * replace all nulls in the first column with the value in the second
 * column.
 *  
 * case 2: If the second column is of the same size as the first, then the function will
 * replace all nulls of the first column with the corresponding elemetns of the
 * second column
 * 
 * @Param[out] first gdf_column
 * @Param[in] second gdf_column, reference column
 * 
 * @Returns GDF_SUCCESS upon successful completion
 *
 * --------------------------------------------------------------------------*/
gdf_error gdf_replace_nulls(gdf_column* col_out, const gdf_column* reference)
{
  GDF_REQUIRE(col_out->dtype == reference->dtype, GDF_DTYPE_MISMATCH);
  GDF_REQUIRE(reference->size == 1 || reference->size == col_out->size, GDF_COLUMN_SIZE_MISMATCH);

  if (reference->size == 1) {
    cudf::type_dispatcher(col_out->dtype, replace_nulls_kernel_forwarder{},
                          col_out->size,
                          col_out->data,
                          col_out->valid,
                          reference->data);
  } else if(reference->size == col_out->size) {
    
  }
  return GDF_SUCCESS;
}


//for_each(auto &[data, valid] in zip(col_out->data, col_out->valid) ) {
//    if !valid 
//      data = reference->data[0];   
//}

//for_each(auto &[data, valid, data_ref] in zip(col_out->data, col_out->valid, reference->data) ) {
//   if !valid 
//      data = data_ref;   
//}