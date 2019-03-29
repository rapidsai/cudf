#ifndef COPYING_UTILITIES_COPYING_UTILS_CUH
#define COPYING_UTILITIES_COPYING_UTILS_CUH

#include <limits>
#include <cstdint>
#include "cudf/types.h"
#include "copying/utilities/copying_utils.hpp"

namespace cudf {
namespace utilities {

constexpr std::size_t NO_DYNAMIC_MEMORY = 0;

__constant__
block_type BLOCK_MASK_VALUE = std::numeric_limits<block_type>::max();

__constant__
block_type BITS_PER_BLOCK = std::numeric_limits<block_type>::digits;

/**
 * @brief A helper struct to to store information related to copy data.
 */
struct data_partition_params {
  gdf_index_type input_offset; /**< The start index position of the input data*/
  gdf_size_type row_size;      /**< The total size of data to be copied*/

  // Not used in the implementation. It provide the start index position
  // of the output data.
  //gdf_index_type output_offset;
};

struct bitmask_partition_params {
  block_type* block_output;
  block_type const* block_input;

  gdf_index_type input_offset;
  block_type rotate_input;
  double_block_type mask_last;

  gdf_size_type input_block_length;
  gdf_size_type partition_block_length;

  //gdf_index_type output_offset;
  //block_type rotate_output;
  //double_block_type mask_first;
};

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

  // Perform rotations in the 'bitmask_value'
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

