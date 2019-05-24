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
#include <iostream>
#include <stdio.h>

#include "copying.hpp"
#include "replace.hpp"
#include "cudf.h"
#include "utilities/error_utils.hpp"
#include "utilities//type_dispatcher.hpp"
#include "utilities/cudf_utils.h"
#include "utilities/cuda_utils.hpp"
#include "bitmask/legacy_bitmask.hpp"
#include "bitmask/bit_mask.cuh"

using bit_mask::bit_mask_t;

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
  template <class T,
            bool input_is_nullable, bool new_is_nullable>
  __global__
  void replace_kernel(T const * __restrict__    input_data,
                      bit_mask_t const * __restrict__ input_valid,
                      gdf_size_type             nrows,
                      T * __restrict__          output_data,
                      bit_mask_t * __restrict__ output_valid,
                      gdf_size_type*            output_null_count,
                      thrust::device_ptr<const T> old_values_begin,
                      thrust::device_ptr<const T> old_values_end,
                      const T*                    d_new_values,
                      bit_mask_t const * __restrict__ new_valid)
  {

    gdf_size_type i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < nrows)
    {
      if ( !input_is_nullable || bit_mask::is_valid(input_valid, i)){
          auto found_ptr = thrust::find(thrust::seq, old_values_begin, old_values_end, input_data[i]);

          if (found_ptr != old_values_end) {
              auto d = thrust::distance(old_values_begin, found_ptr);
              if (!new_is_nullable || bit_mask::is_valid(new_valid, d)){
                  output_data[i] = d_new_values[d];
                  if (input_is_nullable||new_is_nullable) bit_mask::set_bit_safe(output_valid, i);
              }
              else atomicAdd(output_null_count, 1);
          }
          else {
              output_data[i] = input_data[i];
              if (input_is_nullable||new_is_nullable) bit_mask::set_bit_safe(output_valid, i);
          }
      }
      else atomicAdd(output_null_count, 1);
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
    void operator()(const gdf_column  &input_col,
                    const gdf_column &old_values,
                    const gdf_column &new_values,
                    gdf_column       &output)
    {
      const bool input_is_nullable = (input_col.valid != nullptr && input_col.null_count > 0);
      const bool new_is_nullable = (new_values.valid != nullptr && new_values.null_count > 0);

      const bit_mask_t* __restrict__ typed_input_valid = reinterpret_cast<bit_mask_t*>(input_col.valid);
      const bit_mask_t* __restrict__ typed_new_valid = reinterpret_cast<bit_mask_t*>(new_values.valid);
      bit_mask_t* __restrict__ typed_out_valid = reinterpret_cast<bit_mask_t*>(output.valid);

      gdf_size_type *null_count = nullptr;
      if (input_is_nullable || new_is_nullable) {
          RMM_ALLOC(&null_count, sizeof(gdf_size_type), 0);
          CUDA_TRY(cudaMemset(null_count, 0, sizeof(gdf_size_type)));
       }

      thrust::device_ptr<const col_type> old_values_begin = thrust::device_pointer_cast(static_cast<const col_type*>(old_values.data));

      cudf::util::cuda::grid_config_1d grid{input_col.size, BLOCK_SIZE, 1};

      auto replace = replace_kernel<col_type, true, true>;

      if(true == input_is_nullable && false == new_is_nullable){
        replace = replace_kernel<col_type, true, false>;
      }
      else if (false == input_is_nullable && true == new_is_nullable){
        replace = replace_kernel<col_type, false, true>;
      }
      else if (false == input_is_nullable && false == new_is_nullable){
        replace = replace_kernel<col_type, false, false>;
      }

      replace<<<grid.num_blocks, BLOCK_SIZE>>>(static_cast<const col_type*>(input_col.data),
                                             typed_input_valid,
                                             input_col.size,
                                             static_cast<col_type*>(output.data),
                                             typed_out_valid,
                                             null_count,
                                             old_values_begin,
                                             old_values_begin + new_values.size,
                                             static_cast<const col_type*>(new_values.data),
                                             typed_new_valid);
      if (input_is_nullable || new_is_nullable) {
        CUDA_TRY(cudaMemcpy(&output.null_count, null_count,
                               sizeof(gdf_size_type), cudaMemcpyDefault));
        RMM_FREE(null_count, 0);
      }
    }
  };

  gdf_column find_and_replace_all(const gdf_column  &input_col,
                                  const gdf_column &old_values,
                                  const gdf_column &new_values)
  {
    CUDF_EXPECTS(input_col.data != nullptr, "Null input data.");
    CUDF_EXPECTS(old_values.data != nullptr && new_values.data != nullptr, "Null replace data.");
    CUDF_EXPECTS(old_values.size == new_values.size, "old_values and new_values size mismatch.");
    CUDF_EXPECTS(input_col.dtype == old_values.dtype && input_col.dtype == new_values.dtype, "Columns type mismatch.");
    CUDF_EXPECTS(old_values.valid == nullptr || old_values.null_count == 0, "Nulls can not be replaced.");

    gdf_column output = cudf::empty_like(input_col);

    gdf_size_type column_byte_width{gdf_dtype_size(input_col.dtype)};

    void *data = nullptr;
    gdf_valid_type *valid = nullptr;
    RMM_ALLOC(&data, input_col.size * column_byte_width, 0);

    if (input_col.valid != nullptr || new_values.valid != nullptr) {
      gdf_size_type bytes = gdf_valid_allocation_size(input_col.size);
      RMM_ALLOC(&valid, bytes, 0);
      CUDA_TRY(cudaMemset(valid, 0, bytes));
    }
    CUDF_EXPECTS(GDF_SUCCESS == gdf_column_view(&output, data, valid,
                                                input_col.size, input_col.dtype),
                "cudf::replace failed to create output column view");


    cudf::type_dispatcher(input_col.dtype, replace_kernel_forwarder{},
                          input_col,
                          old_values,
                          new_values,
                          output);
    return output;
  }

} //end anonymous namespace

namespace cudf{
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
gdf_column gdf_find_and_replace_all(const gdf_column  &input_col,
                                    const gdf_column &old_values,
                                    const gdf_column &new_values)
{
  return find_and_replace_all(input_col, old_values, new_values);
}

} //end cudf namespace

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



