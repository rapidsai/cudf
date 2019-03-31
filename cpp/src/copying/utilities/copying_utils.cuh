/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
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

#ifndef COPYING_UTILITIES_COPYING_UTILS_CUH
#define COPYING_UTILITIES_COPYING_UTILS_CUH

#include <limits>
#include <cstdint>
#include "cudf/types.h"
#include "copying/utilities/copying_utils.hpp"

namespace cudf {
namespace utilities {

/**
 * @brief Improve the readability of the source code.
 * Parameter for the CUDA kernel.
 */
constexpr std::size_t NO_DYNAMIC_MEMORY = 0;

/**
 * @brief Using limits in order to obtain the max value for the 'block_type' type.
 */
__constant__
block_type BLOCK_MASK_VALUE = std::numeric_limits<block_type>::max();

/**
 * @brief Using limits in order to obtain the number of bits for the 'block_type' type.
 */
__constant__
block_type BITS_PER_BLOCK = std::numeric_limits<block_type>::digits;

/**
 * @brief A helper struct to store information related to the copy of the data.
 */
struct data_partition_params {
  gdf_index_type input_offset;    /**< The start index position of the input data. */
  gdf_size_type row_size;         /**< The total size of data that will be copied. */

  //gdf_index_type output_offset; /**< The start index of the output data. */
};

/**
 * @brief A helper struct to store information related to the copy of the bitmask.
 */
struct bitmask_partition_params {
  block_type* block_output;             /**< An alias of the output bitmask pointer in 'block_type'. */
  block_type const* block_input;        /**< An alias of the input bitmask pointer in 'block_type'. */

  gdf_index_type input_offset;          /**< The start index of the input bitmask (pointer in 'block_type'). */
  block_type rotate_input;              /**< The number of bits that the input bitmask will be rotated or discarded. */
  double_block_type mask_last;          /**< The mask that will be applied to the last calculated bitmask. */

  gdf_size_type input_block_length;     /**< The size of the input data (gdf_column::size) in 'block_type' units. */
  gdf_size_type partition_block_length; /**< The size of the interval to be copied in 'block_type' units. */

  //gdf_index_type output_offset;       /**< The start index of the output bitmask (pointer in 'block_type')*/
  //block_type rotate_output;           /**< The number of bits that the output bitmask will be rotated or discarded. */
  //double_block_type mask_first;       /**< The mask that will be applied to the first calculated bitmask. */
};

/**
 * @brief Bit hack used to calculate the number of ones in 32 bits.
 * The algorithm is optimized only for 32 bits.
 */
__device__ __forceinline__
std::uint32_t count_bits_set_32bits(std::uint32_t bitmask) {
  bitmask = bitmask - ((bitmask >> 1) & 0x55555555);
  bitmask = (bitmask & 0x33333333) + ((bitmask >> 2) & 0x33333333);
  return (((bitmask + (bitmask >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}

/**
 * @brief A CUDA function used to perform the count of the null values.
 * It counts the bits that are set in the bitmask.
 */
__device__ __forceinline__
void perform_bitmask_null_count(bitmask_partition_params const* params,
                                gdf_size_type*                  counter,
                                gdf_index_type const            index) {
  std::uint32_t bitmask = (std::uint32_t)params->block_output[index];
  std::uint32_t null_count_value = count_bits_set_32bits(bitmask);
  atomicAdd(counter, null_count_value);
}

/**
 * @brief A CUDA function used to perform the copy of the data.
 */
template <typename ColumnType>
__device__ __forceinline__
void copy_data(data_partition_params* data_params,
               ColumnType*            output_data,
               ColumnType const*      input_data) {
  // Calculate kernel parameters
  gdf_size_type row_index = threadIdx.x + blockIdx.x * blockDim.x;
  gdf_size_type row_step = blockDim.x * gridDim.x;

  // Perform the copying operation
  while (row_index < data_params->row_size) {
    output_data[row_index] = input_data[data_params->input_offset + row_index];
    row_index += row_step;
  }
}

/**
 * @brief A CUDA function used to perform the copy of the bitmask.
 * The function reads a block of size 'double_block_type' from the input
 * bitmask and process (apply rotation and logical operators) in order to
 * store only a block of size 'block_type' into the output bitmask.
 */
__device__ __forceinline__
void copy_bitmask(bitmask_partition_params const* params,
                  gdf_index_type const            index) {
  // Load bitmask from input in a 'double_block_type'
  double_block_type bitmask_value{0};
  {
    double_block_type lower_value = params->block_input[params->input_offset + index];
    double_block_type upper_value = double_block_type{0};
    if (index < (params->input_block_length - 1)) {
      upper_value = params->block_input[params->input_offset + index + 1];
      upper_value <<= BITS_PER_BLOCK;
    }
    bitmask_value = upper_value + lower_value;
  }

  // Perform rotation in the 'bitmask_value'
  bitmask_value >>= params->rotate_input;

  // Apply mask for the last value in the bitmask
  if ((index == (params->partition_block_length - 1)) && params->mask_last) {
    bitmask_value &= params->mask_last;
  }

  // Store the 'block_type' value into the output bitmask
  params->block_output[index] = bitmask_value & BLOCK_MASK_VALUE;
}

}  // namespace utilities
}  // namespace cudf

#endif

