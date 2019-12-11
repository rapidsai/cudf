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

#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/replace.hpp>
#include <cudf/cudf.h>
#include <rmm/rmm.h>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/legacy/cudf_utils.h>
#include <utilities/legacy/cuda_utils.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <bitmask/legacy/bit_mask.cuh>

using bit_mask::bit_mask_t;

namespace{ //anonymous

  static constexpr int warp_size = 32;
  static constexpr int BLOCK_SIZE = 256;


// returns the block_sum using the given shared array of warp sums.
template <typename T>
__device__ T sum_warps(T* warp_smem)
{
  T block_sum = 0;

   if (threadIdx.x < warp_size) {
    T my_warp_sum = warp_smem[threadIdx.x];
    __shared__ typename cub::WarpReduce<T>::TempStorage temp_storage;
    block_sum = cub::WarpReduce<T>(temp_storage).Sum(my_warp_sum);
  }
  return block_sum;
}


// return the new_value for output column at index `idx`
template<class T, bool replacement_has_nulls>
__device__ auto get_new_value(cudf::size_type         idx,
                           const T* __restrict__ input_data,
                           const T* __restrict__ values_to_replace_begin,
                           const T* __restrict__ values_to_replace_end,
                           const T* __restrict__       d_replacement_values,
                           bit_mask_t const * __restrict__ replacement_valid)
   {
     auto found_ptr = thrust::find(thrust::seq, values_to_replace_begin,
                                      values_to_replace_end, input_data[idx]);
     T new_value{0};
     bool output_is_valid{true};

     if (found_ptr != values_to_replace_end) {
       auto d = thrust::distance(values_to_replace_begin, found_ptr);
       new_value = d_replacement_values[d];
       if (replacement_has_nulls) {
         output_is_valid = bit_mask::is_valid(replacement_valid, d);
       }
     } else {
       new_value = input_data[idx];
     }
     return thrust::make_pair(new_value, output_is_valid);
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
   * @param[out] output_valid_count #valid in output column
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
                      cudf::size_type * __restrict__    output_valid_count,
                      cudf::size_type                   nrows,
                      const T* __restrict__ values_to_replace_begin,
                      const T* __restrict__ values_to_replace_end,
                      const T* __restrict__           d_replacement_values,
                      bit_mask_t const * __restrict__ replacement_valid)
  {
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t active_mask = 0xffffffff;
  active_mask = __ballot_sync(active_mask, i < nrows);
  __shared__ uint32_t valid_sum[warp_size];

  // init shared memory for block valid counts
  if (input_has_nulls or replacement_has_nulls){
    if(threadIdx.x < warp_size) valid_sum[threadIdx.x] = 0;
    __syncthreads();
  }

  while (i < nrows) {
    bool output_is_valid = true;
    uint32_t bitmask = 0xffffffff;

    if (input_has_nulls) {
      bool const input_is_valid{bit_mask::is_valid(input_valid, i)};
      output_is_valid = input_is_valid;

      bitmask = __ballot_sync(active_mask, input_is_valid);

      if (input_is_valid) {
        thrust::tie(output_data[i], output_is_valid)  =
            get_new_value<T, replacement_has_nulls>(i, input_data,
                                      values_to_replace_begin,
                                      values_to_replace_end,
                                      d_replacement_values,
                                      replacement_valid);
      }

    } else {
       thrust::tie(output_data[i], output_is_valid) =
            get_new_value<T, replacement_has_nulls>(i, input_data,
                                      values_to_replace_begin,
                                      values_to_replace_end,
                                      d_replacement_values,
                                      replacement_valid);
    }

    /* output valid counts calculations*/
    if (input_has_nulls or replacement_has_nulls){

      bitmask &= __ballot_sync(active_mask, output_is_valid);

      if(0 == (threadIdx.x % warp_size)){
        output_valid[(int)(i/warp_size)] = bitmask;
        valid_sum[(int)(threadIdx.x / warp_size)] += __popc(bitmask);
      }
    }

    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < nrows);
  }
  if(input_has_nulls or replacement_has_nulls){
    __syncthreads(); // waiting for the valid counts of each warp to be ready

    // Compute total valid count for this block and add it to global count
    uint32_t block_valid_count = sum_warps<uint32_t>(valid_sum);

    // one thread computes and adds to output_valid_count
    if (threadIdx.x < warp_size && 0 == (threadIdx.x % warp_size)) {
      atomicAdd(output_valid_count, block_valid_count);
    }
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
      const bool input_has_nulls = cudf::has_nulls(input_col);
      const bool replacement_has_nulls = cudf::has_nulls(replacement_values);

      const bit_mask_t* __restrict__ typed_input_valid =
                                        reinterpret_cast<bit_mask_t*>(input_col.valid);
      const bit_mask_t* __restrict__ typed_replacement_valid =
                                        reinterpret_cast<bit_mask_t*>(replacement_values.valid);
      bit_mask_t* __restrict__ typed_out_valid =
                                        reinterpret_cast<bit_mask_t*>(output.valid);
      cudf::size_type *valid_count = nullptr;
      if (typed_out_valid != nullptr) {
        RMM_ALLOC(&valid_count, sizeof(cudf::size_type), stream);
        CUDA_TRY(cudaMemsetAsync(valid_count, 0, sizeof(cudf::size_type), stream));
      }

      col_type const * values_to_replace_ptr{ cudf::get_data<col_type const>(values_to_replace) };

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
                                             valid_count,
                                             output.size,
                                             values_to_replace_ptr,
                                             values_to_replace_ptr + replacement_values.size,
                                             static_cast<const col_type*>(replacement_values.data),
                                             typed_replacement_valid);

      if(typed_out_valid != nullptr){
        cudf::size_type valids {0};
        CUDA_TRY(cudaMemcpyAsync(&valids, valid_count,
                               sizeof(cudf::size_type), cudaMemcpyDefault, stream));
        output.null_count = output.size - valids;
        RMM_FREE(valid_count, stream);
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
    {
      return cudf::empty_like(input_col);
    }

    if (0 == values_to_replace.size || 0 == replacement_values.size)
    {
      return cudf::copy(input_col, stream);
    }

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

    gdf_column output = cudf::allocate_like(input_col, RETAIN, stream);

    if (nullptr == input_col.valid && replacement_values.valid != nullptr) {
      cudf::valid_type *valid = nullptr;
      cudf::size_type bytes = gdf_valid_allocation_size(input_col.size);
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

    CHECK_CUDA(stream);
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

using bit_mask::bit_mask_t;

template <typename Type>
__global__
  void replace_nulls_with_scalar(cudf::size_type size,
                                 const Type* __restrict__ in_data,
                                 const bit_mask_t* __restrict__ in_valid,
                                 const Type* __restrict__ replacement,
                                 Type* __restrict__ out_data)
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    out_data[i] = bit_mask::is_valid(in_valid, i)? in_data[i] : *replacement;
  }
}


template <typename Type>
__global__
void replace_nulls_with_column(cudf::size_type size,
                               const Type* __restrict__ in_data,
                               const bit_mask_t* __restrict__ in_valid,
                               const Type* __restrict__ replacement,
                               Type* __restrict__ out_data)
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {
    out_data[i] = bit_mask::is_valid(in_valid, i)? in_data[i] : replacement[i];
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
  void operator()(cudf::size_type    nrows,
                  void*            d_in_data,
                  cudf::valid_type*  d_in_valid,
                  const void*      d_replacement,
                  void*            d_out_data,
                  cudaStream_t     stream = 0)
  {
    cudf::util::cuda::grid_config_1d grid{nrows, BLOCK_SIZE};

    replace_nulls_with_column<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(nrows,
                                          static_cast<const col_type*>(d_in_data),
                                          reinterpret_cast<bit_mask_t*>(d_in_valid),
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
  void operator()(cudf::size_type    nrows,
                  void*            d_in_data,
                  cudf::valid_type*  d_in_valid,
                  const void*      replacement,
                  void*            d_out_data,
                  cudaStream_t     stream = 0)
  {
    cudf::util::cuda::grid_config_1d grid{nrows, BLOCK_SIZE};

    auto t_replacement = static_cast<const col_type*>(replacement);
    col_type* d_replacement = nullptr;
    RMM_TRY(RMM_ALLOC(&d_replacement, sizeof(col_type), stream));
    CUDA_TRY(cudaMemcpyAsync(d_replacement, t_replacement, sizeof(col_type),
                             cudaMemcpyHostToDevice, stream));

    replace_nulls_with_scalar<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(nrows,
                                          static_cast<const col_type*>(d_in_data),
                                          reinterpret_cast<bit_mask_t*>(d_in_valid),
                                          static_cast<const col_type*>(d_replacement),
                                          static_cast<col_type*>(d_out_data)
                                          );
    RMM_TRY(RMM_FREE(d_replacement, stream));
  }
};



} //end anonymous namespace


namespace cudf {
namespace detail {

gdf_column replace_nulls(const gdf_column& input,
                         const gdf_column& replacement,
                         cudaStream_t stream)
{
  if (input.size == 0) {
    return cudf::empty_like(input);
  }

  CUDF_EXPECTS(nullptr != input.data, "Null input data");

  if (input.null_count == 0 || input.valid == nullptr) {
    return cudf::copy(input);
  }

  CUDF_EXPECTS(input.dtype == replacement.dtype, "Data type mismatch");
  CUDF_EXPECTS(replacement.size == 1 || replacement.size == input.size, "Column size mismatch");
  CUDF_EXPECTS(nullptr != replacement.data, "Null replacement data");
  CUDF_EXPECTS(nullptr == replacement.valid || 0 == replacement.null_count,
               "Invalid replacement data");

  gdf_column output = cudf::allocate_like(input, NEVER, stream);

  cudf::type_dispatcher(input.dtype, replace_nulls_column_kernel_forwarder{},
                        input.size,
                        input.data,
                        input.valid,
                        replacement.data,
                        output.data,
                        stream);
  return output;
}


gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement,
                         cudaStream_t stream)
{
  if (input.size == 0) {
    return cudf::empty_like(input);
  }

  CUDF_EXPECTS(nullptr != input.data, "Null input data");

  if (input.null_count == 0 || input.valid == nullptr) {
    return cudf::copy(input);
  }

  CUDF_EXPECTS(input.dtype == replacement.dtype, "Data type mismatch");
  CUDF_EXPECTS(true == replacement.is_valid, "Invalid replacement data");

  gdf_column output = cudf::allocate_like(input, NEVER, stream);

  cudf::type_dispatcher(input.dtype, replace_nulls_scalar_kernel_forwarder{},
                        input.size,
                        input.data,
                        input.valid,
                        &(replacement.data),
                        output.data);
  return output;
}

}  // namespace detail

gdf_column replace_nulls(const gdf_column& input,
                         const gdf_column& replacement)
{
  return detail::replace_nulls(input, replacement, 0);
}


gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement)
{
  return detail::replace_nulls(input, replacement, 0);
}

}  // namespace cudf

