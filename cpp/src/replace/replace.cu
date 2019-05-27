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
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>

#include "cudf.h"
#include "rmm/rmm.h"
#include <copying.hpp>
#include "utilities/error_utils.hpp"
#include "utilities//type_dispatcher.hpp"
#include "utilities/cudf_utils.h"
#include "bitmask/legacy_bitmask.hpp"

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
  void replace_nulls_with_scalar(gdf_size_type size,
                                 const Type* in_data,
                                 gdf_valid_type* in_valid,
                                 const Type* replacement,
                                 Type* out_data) 
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    out_data[i] = gdf_is_valid(in_valid, i)? in_data[i] : *replacement;
  }
}


template <typename Type>
__global__
void replace_nulls_with_column(gdf_size_type size,
                               const Type* in_data,
                               gdf_valid_type* in_valid,
                               const Type* replacement,
                               Type* out_data) 
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    out_data[i] = gdf_is_valid(in_valid, i)? in_data[i] : replacement[i];
  }
}


/* --------------------------------------------------------------------------*/
/** 
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the apropiate data types.
 */
/* ----------------------------------------------------------------------------*/
struct replace_nulls_column_kernel_forwarder {
  template <typename col_type>
  void operator()(gdf_size_type    nrows,
                  void*            d_in_data,
                  gdf_valid_type*  d_in_valid,
                  const void*      d_replacement,
                  void*            d_out_data)
  {
    const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);

    replace_nulls_with_column<<<grid_size, BLOCK_SIZE>>>(nrows,
                                          static_cast<const col_type*>(d_in_data),
                                          (d_in_valid),
                                          static_cast<const col_type*>(d_replacement),
                                          static_cast<col_type*>(d_out_data)
                                          );

  }
};


/* --------------------------------------------------------------------------*/
/** 
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the apropiate data types.
 */
/* ----------------------------------------------------------------------------*/
struct replace_nulls_scalar_kernel_forwarder {
  template <typename col_type>
  void operator()(gdf_size_type    nrows,
                  void*            d_in_data,
                  gdf_valid_type*  d_in_valid,
                  const void*      replacement,
                  void*            d_out_data)
  {
    const size_t grid_size = nrows / BLOCK_SIZE + (nrows % BLOCK_SIZE != 0);

    auto t_replacement = static_cast<const col_type*>(replacement);
    void *d_replacement = NULL;
    RMM_TRY(RMM_ALLOC(&d_replacement, sizeof(col_type), 0));
    CUDA_TRY(cudaMemcpy(d_replacement, t_replacement, sizeof(col_type),
                        cudaMemcpyHostToDevice));

    replace_nulls_with_scalar<<<grid_size, BLOCK_SIZE>>>(nrows,
                                          static_cast<const col_type*>(d_in_data),
                                          (d_in_valid),
                                          static_cast<const col_type*>(d_replacement),
                                          static_cast<col_type*>(d_out_data)
                                          );
    RMM_TRY(RMM_FREE(d_replacement, 0));
  }
};



} //end anonymous namespace


namespace cudf {

gdf_column replace_nulls(const gdf_column& input,
                         const gdf_column& replacement)
{
  CUDF_EXPECTS(input.dtype == replacement.dtype, "Data type mismatch");
  CUDF_EXPECTS(replacement.size == 1 || replacement.size == input.size, "Column size mismatch");
  CUDF_EXPECTS(nullptr != replacement.data, "Null replacement data");
  CUDF_EXPECTS(nullptr != input.data, "Null input data");
  CUDF_EXPECTS(nullptr == replacement.valid || 0 == replacement.null_count,
               "Invalid replacement data");

  gdf_column output = cudf::allocate_like(input);

  cudf::type_dispatcher(input.dtype, replace_nulls_column_kernel_forwarder{},
                        input.size,
                        input.data,
                        input.valid,
                        replacement.data,
                        output.data);

  RMM_TRY(RMM_FREE(output.valid, 0));
  output.valid = nullptr;
  return output;
}


gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement)
{
  if (input.size == 0) {
    return cudf::empty_like(input);
  }


  CUDF_EXPECTS(input.dtype == replacement.dtype, "Data type mismatch");
  CUDF_EXPECTS(nullptr != input.data, "Null input data");
  CUDF_EXPECTS(true == replacement.is_valid, "Invalid replacement data");

  gdf_column output = cudf::allocate_like(input);

  cudf::type_dispatcher(input.dtype, replace_nulls_scalar_kernel_forwarder{},
                        input.size,
                        input.data,
                        input.valid,
                        &(replacement.data),
                        output.data);
  output.valid = nullptr;
  return output;
}

}  // namespace cudf

