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

class CudaVariableScope {
public:
  CudaVariableScope(cudaStream_t stream);
  ~CudaVariableScope();

protected:
  CudaVariableScope(CudaVariableScope&&) = delete;
  CudaVariableScope(const CudaVariableScope&&) = delete;
  CudaVariableScope& operator=(CudaVariableScope&&) = delete;
  CudaVariableScope& operator=(const CudaVariableScope&) = delete;

public:
  gdf_size_type* get_pointer();
  void load_value(gdf_size_type& value);

private:
  gdf_size_type* counter_;
  cudaStream_t stream_;
};

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

  void update_column(gdf_column*        output,
                     gdf_column const*  input_column,
                     CudaVariableScope& variable);

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
