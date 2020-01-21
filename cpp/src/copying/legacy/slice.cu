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

#include <utilities/legacy/column_utils.hpp>
#include <cudf/legacy/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/error.hpp>
#include <utilities/legacy/cuda_utils.hpp>
#include <utilities/legacy/bit_util.cuh>
#include <rmm/thrust_rmm_allocator.h>
#include <bitmask/legacy/bit_mask.cuh> 
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <copying/legacy/slice.hpp>

namespace cudf {

namespace {

using bit_mask_t = bit_mask::bit_mask_t; 


/**
 * @brief Improve the readability of the source code.
 * Parameter for the CUDA kernel.
 */
constexpr std::size_t NO_DYNAMIC_MEMORY = 0;


template <typename ColumnType>
__global__
void slice_data_kernel(ColumnType*           output_data,
                       ColumnType const*     input_data,
                       cudf::size_type const* indices,
                       cudf::size_type const  indices_position) {
  
  cudf::size_type input_offset = indices[indices_position*2];    /**< The start index position of the input data. */
  cudf::size_type row_size = indices[indices_position*2 + 1] - input_offset; 

  // Calculate kernel parameters
  cudf::size_type row_index = threadIdx.x + blockIdx.x * blockDim.x;
  cudf::size_type row_step = blockDim.x * gridDim.x;

  // Perform the copying operation
  while (row_index < row_size) {
    output_data[row_index] = input_data[input_offset + row_index];
    row_index += row_step;
  }
}

/** @brief This function copies a slice of a bitmask.
 * 
 * If the slice is from element 10 to element 40, element 10 corresponds to bit 3 of the second byte, 
 * that bit needs to become bit 0. So we are reading two adjacent blocks and bitshifting them together,
 * to then write one block. We also take care that if the last bits of a bit_mask_t block don't 
 * correspond to this slice, then we to apply a mask to clear those bits.
*/
__global__
void slice_bitmask_kernel(bit_mask_t*           output_bitmask,
                          cudf::size_type*        output_null_count,
                          bit_mask_t const*     input_bitmask,
                          cudf::size_type const   input_size,
                          cudf::size_type const* indices,
                          cudf::size_type const   indices_size,
                          cudf::size_type const  indices_position) {
  // Obtain the indices for copying
  cudf::size_type input_index_begin = indices[indices_position * 2];
  cudf::size_type input_index_end = indices[indices_position * 2 + 1];

  cudf::size_type input_offset = cudf::util::detail::bit_container_index<bit_mask_t, cudf::size_type>(input_index_begin);
  cudf::size_type rotate_input = cudf::util::detail::intra_container_index<bit_mask_t, cudf::size_type>(input_index_begin);
  bit_mask_t mask_last = (bit_mask_t{1} << ((input_index_end - input_index_begin) % bit_mask::bits_per_element)) - bit_mask_t{1};

  cudf::size_type input_block_length = bit_mask::num_elements(input_size);
  cudf::size_type partition_block_length = bit_mask::num_elements(input_index_end - input_index_begin);

  // Calculate kernel parameters
  cudf::size_type row_index = threadIdx.x + blockIdx.x * blockDim.x;
  cudf::size_type row_step = blockDim.x * gridDim.x;

  // Perform the copying operation
  while (row_index < partition_block_length) {
    // load data into one or two adjacent bitmask blocks
    if (rotate_input == 0){
      output_bitmask[row_index] = input_bitmask[input_offset + row_index];
    } else {
      bit_mask_t lower_value = input_bitmask[input_offset + row_index];
      bit_mask_t upper_value = bit_mask_t{0};
      if (row_index < (input_block_length - 1)) {
        upper_value = input_bitmask[input_offset + row_index + 1];      
      }
      
      // Perform rotation 
      output_bitmask[row_index] = __funnelshift_rc(lower_value, upper_value, rotate_input);
    }

    // Apply mask for the last value in the bitmask
    if ((row_index == (partition_block_length - 1)) && mask_last) {
      output_bitmask[row_index] &= mask_last;
    }


    // Perform null bitmask null count
    std::uint32_t null_count_value = __popc(output_bitmask[row_index]); // Count the number of bits that are set to 1 in a 32 bit integer.
    atomicAdd(output_null_count, null_count_value);
    
    row_index += row_step;
  }
}

class Slice {
public:
  Slice(gdf_column const &                input_column,
        cudf::size_type const*             indices,
        cudf::size_type                     num_indices,
        std::vector<gdf_column*> const &  output_columns,
        std::vector<cudaStream_t> const & streams)
  : input_column_(input_column), indices_(indices), num_indices_(num_indices),
    output_columns_(output_columns), streams_(streams)  { }


public:
  template <typename ColumnType>
  void operator()() {

    cudf::size_type columns_quantity = output_columns_.size();
    
    // Perform operation
    for (cudf::size_type index = 0; index < columns_quantity; ++index) {
      
      // Empty output column
      if (output_columns_[index]->size == 0) {
        continue;
      }

      // Create a new cuda variable for null count in the bitmask
      rmm::device_vector<cudf::size_type> bit_set_counter(1, 0);

      // Gather stream
      cudaStream_t stream = get_stream(index);

      // Allocate Column
      gdf_column* output_column = output_columns_[index];
      auto col_width { cudf::byte_width(*output_column) };
      RMM_TRY( RMM_ALLOC(&(output_column->data), col_width * output_column->size, stream) );
      if(input_column_.valid != nullptr){
        RMM_TRY( RMM_ALLOC(&(output_column->valid), sizeof(cudf::valid_type)*gdf_valid_allocation_size(output_column->size), stream) );
      } else {
        output_column->valid = nullptr;
      }

      
      // Configure grid for data kernel launch
      auto data_grid_config = cudf::util::cuda::grid_config_1d(output_column->size, 256);

      // Make a copy of the data in the gdf_column
      slice_data_kernel<ColumnType>
      <<<
        data_grid_config.num_blocks,
        data_grid_config.num_threads_per_block,
        NO_DYNAMIC_MEMORY,
        stream
      >>>(
        static_cast<ColumnType*>(output_column->data),
        static_cast<ColumnType const*>(input_column_.data),
        indices_,
        index
      );

      if(input_column_.valid != nullptr){
        // Configure grid for bit mask kernel launch
        auto valid_grid_config = cudf::util::cuda::grid_config_1d(gdf_num_bitmask_elements(output_column->size), 256);
        
        // Make a copy of the bitmask in the gdf_column
        slice_bitmask_kernel
        <<<
          valid_grid_config.num_blocks,
          valid_grid_config.num_threads_per_block,
          NO_DYNAMIC_MEMORY,
          stream
        >>>(
          reinterpret_cast<bit_mask_t*>(output_column->valid),
          bit_set_counter.data().get(),
          reinterpret_cast<bit_mask_t const*>(input_column_.valid),
          input_column_.size,
          indices_,
          num_indices_,
          index
        );

        CHECK_CUDA(stream);

        // Update the other fields in the output column
        cudf::size_type num_nulls;
        CUDA_TRY(cudaMemcpyAsync(&num_nulls, bit_set_counter.data().get(), sizeof(cudf::size_type), 
                    cudaMemcpyDeviceToHost, stream));
        output_column->null_count = output_column->size - num_nulls;
      } else {
        output_column->null_count = 0;
      }

      if (output_column->dtype == GDF_STRING_CATEGORY){
        CUDF_TRY(nvcategory_gather(output_column, static_cast<NVCategory*>(input_column_.dtype_info.category)));
      }
    }
  }

  private:

    cudaStream_t get_stream(cudf::size_type index) {
      if (streams_.size() == 0) {
        return cudaStream_t{nullptr};
      }
      return streams_[index % streams_.size()];
    }


    gdf_column const                input_column_;
    cudf::size_type const*           indices_;
    cudf::size_type                   num_indices_;
    std::vector<gdf_column*> const  output_columns_;
    std::vector<cudaStream_t>      streams_; 
};
} // namespace

namespace detail {

std::vector<gdf_column*> slice(gdf_column const &         input_column,
                               cudf::size_type const*      indices,
                               cudf::size_type              num_indices,
                               std::vector<cudaStream_t> const & streams) {
  
  std::vector<gdf_column*> output_columns;

  if (num_indices == 0 || indices == nullptr) {
    return output_columns;
  }
  if (input_column.size == 0) {
    return output_columns;
  }
  CUDF_EXPECTS(input_column.data != nullptr, "input column data is null");
  CUDF_EXPECTS((num_indices % 2) == 0, "indices size must be even");
  
  // Get indexes on host side
  std::vector<cudf::size_type> host_indices(num_indices);
  CUDA_TRY( cudaMemcpy(host_indices.data(), indices, num_indices * sizeof(cudf::size_type), cudaMemcpyDeviceToHost) );

  // Initialize output_columns
  output_columns.resize(num_indices/2);
  //TODO: optimize to launch all slices in parallel
  for (cudf::size_type i = 0; i < num_indices/2; i++){
    output_columns[i] = new gdf_column{};
    gdf_column_view_augmented(output_columns[i],
                              nullptr,
                              nullptr,
                              host_indices[2*i + 1] - host_indices[2*i],
                              input_column.dtype,
                              0,
                              {input_column.dtype_info.time_unit, nullptr});
  }

  // Create slice helper class
  Slice slice(input_column, indices, num_indices, output_columns, streams);

  // Perform cudf operation
  cudf::type_dispatcher(input_column.dtype, slice);

  return output_columns;
}

} // namespace detail


std::vector<gdf_column*> slice(gdf_column const &         input_column,
                               cudf::size_type const*      indices,
                               cudf::size_type              num_indices) {

  return cudf::detail::slice(input_column, indices, num_indices);
}

} // namespace cudf
