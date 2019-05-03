/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include <bitmask/legacy_bitmask.hpp>
#include <cassert>
#include <table.hpp>
#include <utilities/error_utils.hpp>

#include <algorithm>

namespace cudf {

table::table(gdf_column* cols[], gdf_size_type num_cols)
    : _columns(cols, cols + num_cols) {
  CUDF_EXPECTS(nullptr != cols[0], "Null input column");
  this->_num_rows = cols[0]->size;

  std::for_each(_columns.begin(), _columns.end(), [this](gdf_column* col) {
    CUDF_EXPECTS(nullptr != col, "Null input column");
    CUDF_EXPECTS(_num_rows == col->size, "Column size mismatch");
  });
}

table::table(gdf_size_type num_rows, std::vector<gdf_dtype> const& dtypes,
             bool allocate_bitmasks, bool all_valid, cudaStream_t stream)
    : _columns(dtypes.size()), _num_rows{num_rows} {
  std::transform(
      _columns.begin(), _columns.end(), dtypes.begin(), _columns.begin(),
      [num_rows, allocate_bitmasks, all_valid, stream](gdf_column*& col,
                                                       gdf_dtype dtype) {
        CUDF_EXPECTS(dtype != GDF_invalid, "Invalid gdf_dtype.");
        CUDF_EXPECTS(dtype != GDF_TIMESTAMP, "Timestamp unsupported.");
        col = new gdf_column;
        col->size = num_rows;
        col->dtype = dtype;
        col->null_count = 0;
        col->valid = nullptr;

        // Timestamp currently unsupported as it would require passing in
        // additional resolution information
        gdf_dtype_extra_info extra_info;
        extra_info.time_unit = TIME_UNIT_NONE;
        col->dtype_info = extra_info;

        RMM_ALLOC(&col->data, gdf_dtype_size(dtype) * num_rows, stream);
        if (allocate_bitmasks) {
          int fill_value = (all_valid) ? 0xff : 0;

          RMM_ALLOC(
              &col->valid,
              gdf_valid_allocation_size(num_rows) * sizeof(gdf_valid_type),
              stream);

          CUDA_TRY(cudaMemsetAsync(
              col->valid, fill_value,
              gdf_valid_allocation_size(num_rows) * sizeof(gdf_valid_type),
              stream));
        }
        return col;
      });
}

}  // namespace cudf
