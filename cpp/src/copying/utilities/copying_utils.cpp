#include "copying/utilities/copying_utils.hpp"
#include "cudf.h"

namespace cudf {
namespace utilities {

BaseCopying::BaseCopying(gdf_column const*   input_column,
                         gdf_column const*   indexes,
                         cudf::column_array* output_columns,
                         cudaStream_t*       streams,
                         gdf_size_type       streams_size)
: input_column_{input_column},
  indexes_{indexes},
  output_columns_{output_columns},
  streams_{streams},
  streams_size_{streams_size}
{ }

BaseCopying::KernelOccupancy BaseCopying::calculate_kernel_data_occupancy(gdf_size_type size) {
  int block_size{256};
  int grid_size = round_up_size(size, block_size);
  return BaseCopying::KernelOccupancy{ .grid_size = grid_size, .block_size = block_size };
}

BaseCopying::KernelOccupancy BaseCopying::calculate_kernel_bitmask_occupancy(gdf_size_type size) {
  gdf_size_type bitmask_size = gdf_num_bitmask_elements(size);
  bitmask_size = round_up_size(size, sizeof(block_type));

  int block_size{256};
  int grid_size = round_up_size(bitmask_size, block_size);
  return BaseCopying::KernelOccupancy{ .grid_size = grid_size, .block_size = block_size };
}

cudaStream_t BaseCopying::get_stream(gdf_index_type index) {
  if (streams_ == nullptr) {
    return cudaStream_t{nullptr};
  }
  return streams_[index % streams_size_];
}

gdf_size_type BaseCopying::round_up_size(gdf_size_type size, gdf_size_type base) {
    return (size + base - 1) / base;
}

} // namespace utilitites 
} // namespace cudf
