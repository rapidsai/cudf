/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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

#include "types.hpp"
#include "copying/utilities/copying_utils.cuh"
#include "utilities/type_dispatcher.hpp"
#include "utilities/error_utils.hpp"
#include "utilities/cuda_utils.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include <nvstrings/NVCategory.h>

namespace cudf {

namespace {

using cudf::utilities::bitmask_partition_params;
using cudf::utilities::data_partition_params;
using cudf::utilities::block_type;
using cudf::utilities::double_block_type;
using cudf::utilities::BLOCK_MASK_VALUE;
using cudf::utilities::BITS_PER_BLOCK;

__device__ __forceinline__
void calculate_data_params(data_partition_params* params,
                           gdf_index_type const*  indices,
                           gdf_index_type const   indices_position) {
  gdf_index_type position = indices_position * 2;
  params->input_offset = indices[position];
  params->row_size = indices[position + 1] - params->input_offset;
}

__device__ __forceinline__
void calculate_bitmask_params(bitmask_partition_params* params,
                              gdf_valid_type*           output_bitmask,
                              gdf_valid_type const*     input_bitmask,
                              gdf_size_type const       input_size,
                              gdf_index_type const*     indices,
                              gdf_size_type const       indices_size,
                              gdf_index_type const      indices_position) {
  params->block_output = reinterpret_cast<block_type*>(output_bitmask);
  params->block_input = reinterpret_cast<block_type const*>(input_bitmask);
  
  gdf_index_type position = indices_position * 2;
  gdf_index_type input_index_begin = indices[position];
  gdf_index_type input_index_end = indices[position + 1];

  params->input_offset = input_index_begin / BITS_PER_BLOCK;
  params->rotate_input = input_index_begin % BITS_PER_BLOCK;
  params->mask_last = (double_block_type{1} << ((input_index_end - input_index_begin) % BITS_PER_BLOCK)) - double_block_type{1};

  params->input_block_length = (input_size + (BITS_PER_BLOCK - 1)) / BITS_PER_BLOCK;
  params->partition_block_length = ((input_index_end - input_index_begin) + (BITS_PER_BLOCK - 1)) / BITS_PER_BLOCK;
}

template <typename ColumnType>
__global__
void slice_data_kernel(ColumnType*           output_data,
                       ColumnType const*     input_data,
                       gdf_index_type const* indices,
                       gdf_index_type const  indices_position) {
  // Obtain the indices for copying
  cudf::utilities::data_partition_params data_params;
  calculate_data_params(&data_params, indices, indices_position);

  // Perform the copy operation
  cudf::utilities::copy_data<ColumnType>(&data_params, output_data, input_data);
}

__global__
void slice_bitmask_kernel(gdf_valid_type*       output_bitmask,
                          gdf_size_type*        output_null_count,
                          gdf_valid_type const* input_bitmask,
                          gdf_size_type const   input_size,
                          gdf_index_type const* indices,
                          gdf_size_type const   indices_size,
                          gdf_index_type const  indices_position) {
  // Obtain the indices for copying
  cudf::utilities::bitmask_partition_params bitmask_params;
  calculate_bitmask_params(&bitmask_params,
                           output_bitmask,
                           input_bitmask,
                           input_size,
                           indices,
                           indices_size,
                           indices_position);

  // Calculate kernel parameters
  gdf_size_type row_index = threadIdx.x + blockIdx.x * blockDim.x;
  gdf_size_type row_step = blockDim.x * gridDim.x;

  // Perform the copying operation
  while (row_index < bitmask_params.partition_block_length) {
    cudf::utilities::copy_bitmask(&bitmask_params, row_index);
    cudf::utilities::perform_bitmask_null_count(&bitmask_params,
                                                output_null_count,
                                                row_index);
    row_index += row_step;
  }
}

class Slice {
public:
  Slice(gdf_column const*                 input_column,
        gdf_index_type const*             indices,
        gdf_size_type                     num_indices,
        std::vector<gdf_column*> const &  output_columns,
        std::vector<cudaStream_t> &      streams)
  : input_column_(input_column), indices_(indices), num_indices_(num_indices),
    output_columns_(output_columns), streams_(streams)  { }


public:
  template <typename ColumnType>
  void operator()() {

    gdf_size_type columns_quantity = num_indices_/2;
    
    // Perform operation
    for (gdf_index_type index = 0; index < columns_quantity; ++index) {
      
      // Empty output column
      if (output_columns_[index]->size == 0) {
        continue;
      }

      // Create a new cuda variable for null count in the bitmask
      rmm::device_vector<gdf_size_type> bit_set_counter(1, 0);

      // Gather stream
      cudaStream_t stream = get_stream(index);

      // Allocate Column
      gdf_column* output_column = output_columns_[index];
      int col_width;
      get_column_byte_width(output_column, &col_width);
      RMM_TRY( RMM_ALLOC(&(output_column->data), col_width * output_column->size, stream) );
      if(input_column_->valid != nullptr){
        RMM_TRY( RMM_ALLOC(&(output_column->valid), sizeof(gdf_valid_type)*gdf_valid_allocation_size(output_column->size), stream) );
      } 

      
      // Calculate kernel occupancy for data
      auto data_grid_config = cudf::util::cuda::grid_config_1d(output_column->size, 256);

      // Make a copy of the data in the gdf_column
      slice_data_kernel<ColumnType>
      <<<
        data_grid_config.num_blocks,
        data_grid_config.num_threads_per_block,
        cudf::utilities::NO_DYNAMIC_MEMORY,
        stream
      >>>(
        static_cast<ColumnType*>(output_column->data),
        static_cast<ColumnType const*>(input_column_->data),
        indices_,
        index
      );

      // Calculate kernel occupancy for bitmask
      auto valid_grid_config = cudf::util::cuda::grid_config_1d(gdf_num_bitmask_elements(output_column->size), 256);
      
      // Make a copy of the bitmask in the gdf_column
      slice_bitmask_kernel
      <<<
        valid_grid_config.num_blocks,
        valid_grid_config.num_threads_per_block,
        cudf::utilities::NO_DYNAMIC_MEMORY,
        stream
      >>>(
        output_column->valid,
        bit_set_counter.data().get(),
        input_column_->valid,
        input_column_->size,
        indices_,
        num_indices_,
        index
      );

      CHECK_STREAM(stream);

      // Update the other fields in the output column
      gdf_size_type num_nulls;
      CUDA_TRY(cudaMemcpyAsync(&num_nulls, bit_set_counter.data().get(), sizeof(gdf_size_type), 
                  cudaMemcpyDeviceToHost, stream));
      output_column->null_count = output_column->size - num_nulls;

      if (output_column->dtype == GDF_STRING_CATEGORY){
        NVCategory* new_category = static_cast<NVCategory*>(input_column_->dtype_info.category)->gather_and_remap(
                      static_cast<int*>(output_column->data), (unsigned int)output_column->size);
        output_column->dtype_info.category = new_category;
      }
    }
  }

  private:

    cudaStream_t get_stream(gdf_index_type index) {
      if (streams_.size() == 0) {
        return cudaStream_t{nullptr};
      }
      return streams_[index % streams_.size()];
    }


    gdf_column const*               input_column_;
    gdf_index_type const*           indices_;
    gdf_size_type                   num_indices_;
    std::vector<gdf_column*> const  output_columns_;
    std::vector<cudaStream_t>      streams_; 
};
} // namespace

namespace detail {

std::vector<gdf_column*> slice(gdf_column const*          input_column,
                               gdf_index_type const*      indices,
                               gdf_size_type              num_indices,
                               std::vector<cudaStream_t> & streams) {
  
  std::vector<gdf_column*> output_columns;

  CUDF_EXPECTS(indices != nullptr, "indices array is null");
  CUDF_EXPECTS(input_column != nullptr, "input column is null");
  if (num_indices == 0) {
    return output_columns;
  }
  if (input_column->size == 0) {
    return output_columns;
  }
  CUDF_EXPECTS(input_column->data != nullptr, "input column data is null");
  CUDF_EXPECTS((num_indices % 2) == 0, "indices size must be even");
  
  // Get indexes on host side
  std::vector<gdf_size_type> host_indices(num_indices);
  CUDA_TRY( cudaMemcpy(host_indices.data(), indices, num_indices * sizeof(gdf_size_type), cudaMemcpyDeviceToHost) );

  // Initialize output_columns
  output_columns.resize(num_indices/2);
  for (gdf_size_type i = 0; i < num_indices/2; i++){
    output_columns[i] = new gdf_column;
    output_columns[i]->size = host_indices[2*i + 1] - host_indices[2*i];
    output_columns[i]->dtype = input_column->dtype;
    output_columns[i]->dtype_info.time_unit = input_column->dtype_info.time_unit;
    output_columns[i]->null_count = 0;
    output_columns[i]->data = nullptr;
    output_columns[i]->valid = nullptr;
  }

  // Create slice helper class
  Slice slice(input_column, indices, num_indices, output_columns, streams);

  // Perform cudf operation
  cudf::type_dispatcher(input_column->dtype, slice);

  return output_columns;
}

} // namespace detail


std::vector<gdf_column*> slice(gdf_column const*          input_column,
                               gdf_index_type const*      indices,
                               gdf_size_type              num_indices) {

  std::vector<cudaStream_t> streams;
  return cudf::detail::slice(input_column, indices, num_indices, streams);
}

} // namespace cudf
