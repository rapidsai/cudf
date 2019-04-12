/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
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

#include <cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <stream_compaction.hpp>
#include <bitmask/legacy_bitmask.hpp>
#include <utilities/device_atomics.cuh>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>
#include <utilities/wrapper_types.hpp>

namespace {
struct nonnull_and_true {
  nonnull_and_true(gdf_column const boolean_mask)
      : data{static_cast<cudf::bool8*>(boolean_mask.data)},
        bitmask{boolean_mask.valid} {
    CUDF_EXPECTS(boolean_mask.dtype == GDF_BOOL, "Expected boolean column");
    CUDF_EXPECTS(boolean_mask.data != nullptr, "Null boolean_mask data");
    CUDF_EXPECTS(boolean_mask.valid != nullptr, "Null boolean_mask bitmask");
  }

  __device__ int operator()(gdf_index_type i) {
    return ((cudf::true_v == data[i]) && gdf_is_valid(bitmask, i));
  }

 private:
  cudf::bool8 const * const data;
  gdf_valid_type const * const bitmask;
};
}  // namespace

namespace cudf {

/**
 * @brief Filters a column using a column of boolean values as a mask.
 *
 */
/*gdf_column apply_boolean_mask_old(gdf_column const *input,
                                  gdf_column const *boolean_mask) {
  CUDF_EXPECTS(nullptr != input, "Null input");
  CUDF_EXPECTS(nullptr != boolean_mask, "Null boolean_mask");
  CUDF_EXPECTS(input->size == boolean_mask->size, "Column size mismatch");
  CUDF_EXPECTS(boolean_mask->dtype == GDF_BOOL, "Mask must be Boolean type");

  // High Level Algorithm:
  // First, compute a `gather_map` from the boolean_mask that will gather
  // input[i] if boolean_mask[i] is non-null and "true".
  // Second, use the `gather_map` to gather elements from the `input` column
  // into the `output` column

  // We don't know the exact size of the gather_map a priori, but we know it's
  // upper bounded by the size of the boolean_mask
  rmm::device_vector<gdf_index_type> gather_map(boolean_mask->size);

  // Returns an iterator to the end of the gather_map
  auto end = thrust::copy_if(
      rmm::exec_policy()->on(0), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(boolean_mask->size),
      thrust::make_counting_iterator(0), gather_map.begin(),
      nonnull_and_true{*boolean_mask});

  // Use the returned iterator to determine the size of the gather_map
  gdf_size_type output_size{
      static_cast<gdf_size_type>(end - gather_map.begin())};
  gdf_column output;
  gdf_column_view(&output, 0, 0, 0, input->dtype);
  output.dtype_info = input->dtype_info;

  if (output_size > 0) {
    // have to do this because cudf::gather operates on cudf::tables and
    // there seems to be no way to create a cudf::table from a const gdf_column!
    gdf_column* input_view[1] = {new gdf_column};
    CUDF_EXPECTS(GDF_SUCCESS == gdf_column_view(input_view[0], input->data,
                                                input->valid, input->size,
                                                input->dtype),
                "cudf::apply_boolean_mask failed to create input column view");

     // Allocate/initialize output column
    gdf_size_type column_byte_width{gdf_dtype_size(input->dtype)};

    void *data = nullptr;
    gdf_valid_type *valid = nullptr;
    RMM_ALLOC(&data, output_size * column_byte_width, 0);
    if (input->valid != nullptr)
      RMM_ALLOC(&valid, gdf_valid_allocation_size(output_size*column_byte_width), 0);

    gdf_column* outputs[1] = {&output};
    CUDF_EXPECTS(GDF_SUCCESS == gdf_column_view(outputs[0], data, valid,
                                                output_size, input->dtype),
                "cudf::apply_boolean_mask failed to create output column view");

    cudf::table input_table{input_view, 1};
    cudf::table output_table{outputs, 1};

    cudf::gather(&input_table, thrust::raw_pointer_cast(gather_map.data()),
                &output_table);

    delete input_view[0];
  }
  return output;
}*/

struct scatter_functor 
{
  template <typename T>
  __host__ __device__
  void operator()(gdf_column *out, int out_index, 
                  gdf_column const *in, int in_index) {
    static_cast<T*>(out->data)[out_index] = 
      static_cast<T const*>(in->data)[in_index];
  }
};

__device__ void atomicSetValidBit(gdf_valid_type *valid_mask, gdf_index_type i)
{
  const gdf_index_type index = gdf_valid_mask_index(i);
  const gdf_index_type bit   = gdf_valid_bit_index(i);
  atomicOr( &valid_mask[index], gdf_valid_type(1 << bit) );
}

__device__ void atomicClearValidBit(gdf_valid_type *valid_mask, gdf_index_type i)
{
  const gdf_index_type index = gdf_valid_mask_index(i);
  const gdf_index_type bit   = gdf_valid_bit_index(i);
  atomicAnd( &valid_mask[index], gdf_valid_type(~(1 << bit)) );
}

template <int block_size, int per_thread, typename MaskFunc>
__global__ void scatter_foo(gdf_column *output_column,
                            gdf_column const * input_column,
                            gdf_index_type const *scatter_map,
                            gdf_size_type scatter_size,
                            gdf_size_type num_columns,
                            bool has_valid,
                            MaskFunc mask)
{
  int tid = threadIdx.x + per_thread * block_size * blockIdx.x;

  //if (tid + block_size * (per_thread - 1) < scatter_size) {
    for (int i = 0; i < per_thread; i++) {
      if (tid < scatter_size) {
        if (mask(tid)) {

        const gdf_index_type in_index = tid;
        const gdf_index_type out_index = scatter_map[tid];

        //for (int c = 0; c < num_columns; c++) {
        //static_cast<T*>(out->data)[out_index] = 
        //  static_cast<T const*>(in->data)[in_index];
        cudf::type_dispatcher(output_column->dtype, scatter_functor{},
                              output_column, out_index,
                              input_column, in_index);

        if (has_valid) {
          // Scatter the valid bit
          if (gdf_is_valid(input_column->valid, in_index)) {
            //printf("Setting bit for index %d\n", out_index);
            atomicSetValidBit(output_column->valid, out_index);
          }
          else {
            atomicAdd(&(output_column->null_count), 1);
            //printf("Clearing bit for index %d\n", out_index);
            atomicClearValidBit(output_column->valid, out_index);
          }
        }
      }
    }
    tid += block_size;
  }
  //}
}

template <typename MaskFunc>
__global__ void get_output_size(gdf_size_type  *output_size, 
                                gdf_index_type *scatter_map,
                                gdf_size_type   mask_size,
                                MaskFunc        mask)
{
  *output_size = scatter_map[mask_size-1] + gdf_index_type{mask(mask_size-1)};
}

gdf_column apply_boolean_mask(gdf_column const *input,
                              gdf_column const *boolean_mask) {
  CUDF_EXPECTS(nullptr != input, "Null input");
  CUDF_EXPECTS(nullptr != boolean_mask, "Null boolean_mask");
  CUDF_EXPECTS(input->size == boolean_mask->size, "Column size mismatch");
  CUDF_EXPECTS(boolean_mask->dtype == GDF_BOOL, "Mask must be Boolean type");

  // High Level Algorithm:
  // First, compute a `scatter_map` from the boolean_mask that will scatter
  // input[i] if boolean_mask[i] is non-null and "true". This is simply an exclusive
  // scan of nonnull_and_true
  // Second, use the `scatter_map` to scatter elements from the `input` column
  // into the `output` column

  rmm::device_vector<gdf_index_type> scatter_map(boolean_mask->size);

  auto xform = thrust::make_transform_iterator(thrust::make_counting_iterator(0), 
                                               nonnull_and_true{*boolean_mask});
  thrust::exclusive_scan(rmm::exec_policy(0)->on(0),
                         xform, 
                         xform+boolean_mask->size, 
                         scatter_map.begin(), 
                         gdf_index_type{0});

  // Last element of scan contains size if the last element of the mask is 0,
  // or size-1 if it is 1.
  gdf_size_type *output_size = nullptr;
  cudaMallocHost(&output_size, sizeof(gdf_size_type));
  get_output_size<<<1, 1>>>(output_size, 
                            thrust::raw_pointer_cast(scatter_map.data()), 
                            boolean_mask->size, 
                            nonnull_and_true{*boolean_mask});
  
  gdf_column output;
  gdf_column_view(&output, 0, 0, 0, input->dtype);
  output.dtype_info = input->dtype_info;

  cudaDeviceSynchronize();

  if (*output_size > 0) {
    // have to do this because cudf::scatter operates on cudf::tables and
    // there seems to be no way to create a cudf::table from a const gdf_column!
    //gdf_column const * inputs[1] = {input};
    
    // Allocate/initialize output column
    gdf_size_type column_byte_width{gdf_dtype_size(input->dtype)};

    void *data = nullptr;
    gdf_valid_type *valid = nullptr;
    RMM_ALLOC(&data, *output_size * column_byte_width, 0);
    if (input->valid != nullptr) {
      gdf_size_type bytes = gdf_valid_allocation_size(*output_size);
      RMM_ALLOC(&valid, bytes, 0);
    }
    
    CUDF_EXPECTS(GDF_SUCCESS == gdf_column_view(&output, data, valid,
                                                *output_size, input->dtype),
                "cudf::apply_boolean_mask failed to create output column view");
    gdf_column *d_input = nullptr, *d_output = nullptr;
    RMM_ALLOC(&d_input, sizeof(gdf_column), 0);
    RMM_ALLOC(&d_output, sizeof(gdf_column), 0);
    cudaMemcpy(d_input, input, sizeof(gdf_column), cudaMemcpyDefault);
    cudaMemcpy(d_output, &output, sizeof(gdf_column), cudaMemcpyDefault);
    
    //gdf_column* outputs[1] = {&output};

    constexpr int block_size = 256;
    constexpr int per_thread = 32;
    constexpr int per_block = block_size * per_thread;
    const int num_blocks = (boolean_mask->size + per_block - 1) / per_block;
    scatter_foo<block_size, per_thread>
      <<<num_blocks, block_size>>>(d_output, d_input, 
                                   thrust::raw_pointer_cast(scatter_map.data()),
                                   boolean_mask->size, gdf_size_type{1},
                                   input->valid != nullptr, 
                                   nonnull_and_true{*boolean_mask});

    CHECK_STREAM(0);

    cudaMemcpy(&output, d_output, sizeof(gdf_column), cudaMemcpyDefault);
    RMM_FREE(d_input, 0);
    RMM_FREE(d_output, 0);
  }
  return output;
}


}  // namespace cudf