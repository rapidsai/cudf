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

#include "copying/utilities/copying_utils.hpp"
#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.hpp"

namespace cudf {
namespace utilities {

CudaVariableScope::CudaVariableScope(cudaStream_t stream)
: counter_{nullptr}, stream_{stream} {
  RMM_ALLOC((void**)&counter_, sizeof(gdf_size_type), stream_);
  CUDA_TRY(cudaMemsetAsync(counter_, 0, sizeof(gdf_size_type), stream_));
}

CudaVariableScope::~CudaVariableScope() {
  RMM_FREE(counter_, stream_);
}

gdf_size_type* CudaVariableScope::get_pointer() {
  return counter_;
}

void CudaVariableScope::load_value(gdf_size_type& value) {
  CUDA_TRY(cudaMemcpyAsync(&value, counter_, sizeof(gdf_size_type), cudaMemcpyDeviceToHost, stream_));
}


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
  bitmask_size = round_up_size(bitmask_size, sizeof(block_type));

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

bool BaseCopying::validate_inputs() {
  CUDF_EXPECTS(indexes_ != nullptr, "indexes array is null");
  CUDF_EXPECTS(input_column_ != nullptr, "input column is null");
  CUDF_EXPECTS(output_columns_ != nullptr, "output columns array is null");

  if (indexes_->size == 0) {
    return false;
  }
  if (input_column_->size == 0) {
    return false;
  }

  CUDF_EXPECTS(indexes_->data != nullptr, "indexes data array is null");
  CUDF_EXPECTS(input_column_->data != nullptr, "input column data is null");
  CUDF_EXPECTS(input_column_->valid != nullptr, "input column bitmask is null");

  return true;
}

void BaseCopying::update_column(gdf_column*        output_column,
                                gdf_column const*  input_column,
                                CudaVariableScope& variable) {
  gdf_size_type value;
  variable.load_value(value);
  output_column->null_count = output_column->size - value;
}

} // namespace utilitites 
} // namespace cudf
