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
#include <cub/cub.cuh>

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

  static constexpr int warp_size = 32;
  static constexpr int BLOCK_SIZE = 256;


// return the new_value for output column at index `idx`
template<class T, bool replacement_has_nulls>
__device__ T get_new_value(gdf_size_type         idx,
                           const T* __restrict__ input_data,
                           thrust::device_ptr<const T> values_to_replace_begin,
                           thrust::device_ptr<const T> values_to_replace_end,
                           const T* __restrict__       d_replacement_values,
                           bit_mask_t const * __restrict__ replacement_valid,
                           bool                        &output_is_valid)
   {
     auto found_ptr = thrust::find(thrust::seq, values_to_replace_begin,
                                      values_to_replace_end, input_data[idx]);
     T new_value{0};
     if (found_ptr != values_to_replace_end) {
       auto d = thrust::distance(values_to_replace_begin, found_ptr);
       new_value = d_replacement_values[d];
       if (replacement_has_nulls) {
         output_is_valid = bit_mask::is_valid(replacement_valid, d);
       }
     } else {
       new_value = input_data[idx];
     }
     return new_value;
   }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief Kernel that replaces elements from `output_data` given the following
   *        rule: replace all `values_to_replace[i]` in [values_to_replace_begin`,
   *        `values_to_replace_end`) present in `output_data` with `d_replacement_values[i]`.
   *
   * @tparam input_has_nulls `true` if output column has valid mask, `false` otherwise
   * @tparam replacement_has_nulls `true` if replacement_values column has valid mask, `false` otherwise
   * The input_has_nulls and replacement_has_nulls template parameters allows us to specialize
   * this kernel for the different scenario for performance without writing different kernel.
   *
   * @param[in] input_data Device array with the data to be modified
   * @param[in] input_valid Valid mask associated with input_data
   * @param[out] output_data Device array to store the data from input_data
   * @param[out] output_valid Valid mask associated with output_data
   * @param[out] output_null_count #nulls in output column
   * @param[in] nrows # rows in `output_data`
   * @param[in] values_to_replace_begin Device pointer to the beginning of the sequence
   * of old values to be replaced
   * @param[in] values_to_replace_end  Device pointer to the end of the sequence
   * of old values to be replaced
   * @param[in] d_replacement_values Device array with the new values
   * @param[in] replacement_valid Valid mask associated with d_replacement_values
   *
   * @returns
   */
  /* ----------------------------------------------------------------------------*/
  template <class T,
            bool input_has_nulls, bool replacement_has_nulls>
  __global__
  void replace_kernel(const T* __restrict__           input_data,
                      bit_mask_t const * __restrict__ input_valid,
                      T * __restrict__          output_data,
                      bit_mask_t * __restrict__ output_valid,
                      gdf_size_type *           output_null_count,
                      gdf_size_type             nrows,
                      thrust::device_ptr<const T> values_to_replace_begin,
                      thrust::device_ptr<const T> values_to_replace_end,
                      const T* __restrict__           d_replacement_values,
                      bit_mask_t const * __restrict__ replacement_valid)
  {
  gdf_size_type i = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t active_mask = 0xffffffff;
  active_mask = __ballot_sync(active_mask, i < nrows);
  __shared__ T    temp_data[BLOCK_SIZE];

  while (i < nrows) {
    bool output_is_valid = true;
    uint32_t bitmask = 0xffffffff;

    if (input_has_nulls) {
      bool const input_is_valid{bit_mask::is_valid(input_valid, i)};
      output_is_valid = input_is_valid;

      bitmask = __ballot_sync(active_mask, input_is_valid);

      if (input_is_valid) {
        temp_data[threadIdx.x] = get_new_value<T, replacement_has_nulls>(i, input_data,
                                      values_to_replace_begin,
                                      values_to_replace_end,
                                      d_replacement_values,
                                      replacement_valid,
                                      output_is_valid);
      }

    } else {
       temp_data[threadIdx.x] = get_new_value<T, replacement_has_nulls>(i, input_data,
                                      values_to_replace_begin,
                                      values_to_replace_end,
                                      d_replacement_values,
                                      replacement_valid,
                                      output_is_valid);
    }

    __syncthreads(); // waiting for temp_data to be ready

    output_data[i] = temp_data[threadIdx.x];

    /* output null counts calculations*/
    if (input_has_nulls or replacement_has_nulls){

      bitmask &= __ballot_sync(active_mask, output_is_valid);

      // allocating the shared memory for calculating the block_sum and
      // block_valid_counts
      __shared__ uint32_t warp_smem[warp_size];

      if(threadIdx.x < warp_size) warp_smem[threadIdx.x] = 0;
      __syncthreads();

      const int wid = threadIdx.x / warp_size;
      const int lane = threadIdx.x % warp_size;

      if(lane == 0)warp_smem[wid] = __popc(active_mask);
      __syncthreads(); // waiting for the sum of each warp to be ready

      uint32_t block_sum = 0;
      if (threadIdx.x < warp_size) {
        uint32_t my_warp_sum = warp_smem[threadIdx.x];

        __shared__ typename cub::WarpReduce<uint32_t>::TempStorage temp_storage;

        block_sum = cub::WarpReduce<uint32_t>(temp_storage).Sum(my_warp_sum);
      }

      // reusing the same shared memory for block valid counts
      if(threadIdx.x < warp_size) warp_smem[threadIdx.x] = 0;
      __syncthreads();
      if(lane == 0 && bitmask != 0){
        int valid_index = wid + (blockIdx.x * (BLOCK_SIZE/warp_size));
        output_valid[valid_index] = bitmask;
        warp_smem[wid] = __popc(bitmask);
      }
      __syncthreads(); // waiting for the valid counts of each warp to be ready

      // Compute total null_count for this block and add it to global count
      if (threadIdx.x < warp_size) {
        uint32_t my_valid_count = warp_smem[threadIdx.x];

        __shared__ typename cub::WarpReduce<uint32_t>::TempStorage temp_storage;

        uint32_t block_valid_count =
          cub::WarpReduce<uint32_t>(temp_storage).Sum(my_valid_count);

        if (lane == 0) { // one thread computes and adds to null count
          atomicAdd(output_null_count, block_sum - block_valid_count);
        }
      }
    }

    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < nrows);
  }
}

  /* --------------------------------------------------------------------------*/
  /**
   * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
   *        `replace_kernel` with the appropriate data types.
   */
  /* ----------------------------------------------------------------------------*/
  struct replace_kernel_forwarder {
    template <typename col_type>
    void operator()(const gdf_column &input_col,
                    const gdf_column &values_to_replace,
                    const gdf_column &replacement_values,
                    gdf_column       &output,
                    cudaStream_t     stream = 0)
    {
      const bool input_has_nulls = (input_col.valid != nullptr && input_col.null_count > 0);
      const bool replacement_has_nulls =
                    (replacement_values.valid != nullptr && replacement_values.null_count > 0);

      const bit_mask_t* __restrict__ typed_input_valid =
                                        reinterpret_cast<bit_mask_t*>(input_col.valid);
      const bit_mask_t* __restrict__ typed_replacement_valid =
                                        reinterpret_cast<bit_mask_t*>(replacement_values.valid);
      bit_mask_t* __restrict__ typed_out_valid =
                                        reinterpret_cast<bit_mask_t*>(output.valid);
      gdf_size_type *null_count = nullptr;
      if (typed_out_valid != nullptr) {
        RMM_ALLOC(&null_count, sizeof(gdf_size_type), stream);
        CUDA_TRY(cudaMemsetAsync(null_count, 0, sizeof(gdf_size_type), stream));
      }

      thrust::device_ptr<const col_type> values_to_replace_begin =
                thrust::device_pointer_cast(static_cast<const col_type*>(values_to_replace.data));

      cudf::util::cuda::grid_config_1d grid{output.size, BLOCK_SIZE, 1};

      auto replace = replace_kernel<col_type, true, true>;

      if (input_has_nulls){
        if (replacement_has_nulls){
          replace = replace_kernel<col_type, true, true>;
        }else{
          replace = replace_kernel<col_type, true, false>;
        }
      }else{
        if (replacement_has_nulls){
          replace = replace_kernel<col_type, false, true>;
        }else{
          replace = replace_kernel<col_type, false, false>;
        }
      }
      replace<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(
                                             static_cast<const col_type*>(input_col.data),
                                             typed_input_valid,
                                             static_cast<col_type*>(output.data),
                                             typed_out_valid,
                                             null_count,
                                             output.size,
                                             values_to_replace_begin,
                                             values_to_replace_begin + replacement_values.size,
                                             static_cast<const col_type*>(replacement_values.data),
                                             typed_replacement_valid);

      if(typed_out_valid != nullptr){
        CUDA_TRY(cudaMemcpyAsync(&output.null_count, null_count,
                               sizeof(gdf_size_type), cudaMemcpyDefault, stream));
        RMM_FREE(null_count, stream);
      }
    }
  };
 } //end anonymous namespace

namespace cudf{
namespace detail {

  gdf_column find_and_replace_all(const gdf_column &input_col,
                                  const gdf_column &values_to_replace,
                                  const gdf_column &replacement_values,
                                  cudaStream_t stream = 0) {

    if (0 == input_col.size )
      return cudf::empty_like(input_col);

    if (0 == values_to_replace.size || 0 == replacement_values.size)
      return cudf::copy(input_col, stream);

    CUDF_EXPECTS(values_to_replace.size == replacement_values.size,
                 "values_to_replace and replacement_values size mismatch.");
    CUDF_EXPECTS(input_col.dtype == values_to_replace.dtype &&
                 input_col.dtype == replacement_values.dtype,
                 "Columns type mismatch.");
    CUDF_EXPECTS(input_col.data != nullptr, "Null input data.");
    CUDF_EXPECTS(values_to_replace.data != nullptr && replacement_values.data != nullptr,
                 "Null replace data.");
    CUDF_EXPECTS(values_to_replace.valid == nullptr || values_to_replace.null_count == 0,
                 "Nulls are in values_to_replace column.");

    gdf_column output = cudf::allocate_like(input_col, stream);

    if (nullptr == input_col.valid && replacement_values.valid != nullptr) {
      gdf_valid_type *valid = nullptr;
      gdf_size_type bytes = gdf_valid_allocation_size(input_col.size);
      RMM_ALLOC(&valid, bytes, stream);
      CUDA_TRY(cudaMemsetAsync(valid, 0, bytes, stream));
      CUDF_EXPECTS(GDF_SUCCESS == gdf_column_view(&output, output.data, valid,
                                                input_col.size, input_col.dtype),
                "cudf::replace failed to add valid mask to output col.");
    }

    cudf::type_dispatcher(input_col.dtype, replace_kernel_forwarder{},
                          input_col,
                          values_to_replace,
                          replacement_values,
                          output,
                          stream);

    CHECK_STREAM(stream);
    return output;
  }

} //end details

/* --------------------------------------------------------------------------*/
/**
 * @brief Replace elements from `input_col` according to the mapping `values_to_replace` to
 *        `replacement_values`, that is, replace all `values_to_replace[i]` present in `input_col`
 *        with `replacement_values[i]`.
 *
 * @param[in] col gdf_column with the data to be modified
 * @param[in] values_to_replace gdf_column with the old values to be replaced
 * @param[in] replacement_values gdf_column with the new values
 *
 * @returns output gdf_column with the modified data
 */
/* ----------------------------------------------------------------------------*/
gdf_column find_and_replace_all(const gdf_column  &input_col,
                                const gdf_column &values_to_replace,
                                const gdf_column &replacement_values){
    return detail::find_and_replace_all(input_col, values_to_replace, replacement_values);
  }

} //end cudf

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



