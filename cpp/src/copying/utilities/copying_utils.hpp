#ifndef COPYING_UTILITIES_COPYING_UTILS_HPP
#define COPYING_UTILITIES_COPYING_UTILS_HPP

#include <cstdint>
#include <cuda_runtime.h>
#include "cudf/types.h"

// Forward declaration
namespace cudf {
struct column_array;
} // namespace cudf

namespace cudf {
namespace utilities {

using block_type = std::uint32_t;
using double_block_type = std::uint64_t;

class BaseCopying {
protected:
  BaseCopying(gdf_column const*   input_column,
              gdf_column const*   indexes,
              cudf::column_array* output_columns,
              cudaStream_t*       streams,
              gdf_size_type       streams_size);

protected:
  struct KernelOccupancy {
    int grid_size{0};
    int block_size{0}; 
  };

  KernelOccupancy calculate_kernel_data_occupancy(gdf_size_type size);

  KernelOccupancy calculate_kernel_bitmask_occupancy(gdf_size_type size);

protected:
  cudaStream_t get_stream(gdf_index_type index);

protected:
  gdf_size_type round_up_size(gdf_size_type size, gdf_size_type base);

protected:
  bool validate_inputs();

protected:
  gdf_column const*   input_column_;
  gdf_column const*   indexes_;
  cudf::column_array* output_columns_;
  cudaStream_t*       streams_;
  gdf_size_type       streams_size_;
};

}  // namespace utilities
}  // namespace cudf

#endif
