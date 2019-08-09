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

#include <cudf/cudf.h>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cassert>
#include <cudf/copying.hpp>
#include <cudf/legacy/table.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/error_utils.hpp>

#include <algorithm>

namespace cudf {

table::table(std::vector<gdf_column*> const& cols) : _columns{cols} {
  CUDF_EXPECTS(nullptr != cols[0], "Null input column");
  gdf_size_type num_rows = cols[0]->size;

  std::for_each(_columns.begin(), _columns.end(),
                [this, num_rows](gdf_column* col) {
                  CUDF_EXPECTS(nullptr != col, "Null input column");
                  CUDF_EXPECTS(num_rows == col->size, "Column size mismatch");
                });
}

table::table(std::initializer_list<gdf_column*> list)
    : table{std::vector<gdf_column*>(list)} {}

table::table(gdf_column* cols[], gdf_size_type num_cols)
    : table{std::vector<gdf_column*>(cols, cols + num_cols)} {}

table::table(gdf_size_type num_rows,
             std::vector<gdf_dtype> const& dtypes,
             std::vector<gdf_dtype_extra_info> const& dtype_infos,
             bool allocate_bitmasks, bool all_valid, cudaStream_t stream)
    : _columns(dtypes.size()) {
  std::transform(
      _columns.begin(), _columns.end(), dtypes.begin(), _columns.begin(),
      [num_rows, allocate_bitmasks, all_valid, stream](gdf_column*& col,
                                                       gdf_dtype dtype) {
        CUDF_EXPECTS(dtype != GDF_invalid, "Invalid gdf_dtype.");
        col = new gdf_column{};

        gdf_dtype_extra_info extra_info{TIME_UNIT_NONE};
        extra_info.category = nullptr;
        CUDF_EXPECTS(GDF_SUCCESS ==
                      gdf_column_view_augmented(col, nullptr, nullptr, num_rows,
                                                dtype, 0, extra_info),
                     "Invalid column parameters");

        RMM_ALLOC(&col->data, cudf::size_of(dtype) * num_rows, stream);
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

    std::transform(
        _columns.begin(), _columns.end(), dtype_infos.begin(), _columns.begin(),
        [](gdf_column*& col, gdf_dtype_extra_info dtype_info) {
          col->dtype_info.time_unit = dtype_info.time_unit;
          return col;
      });
}

void table::destroy(void) {
  for (auto& col : _columns) {
    gdf_column_free(col);
    delete col;
  }
}

std::vector<gdf_dtype> column_dtypes(cudf::table const& table) {
  std::vector<gdf_dtype> dtypes(table.num_columns());

  std::transform(table.begin(), table.end(), dtypes.begin(),
                 [](gdf_column const* col) { return col->dtype; });
  return dtypes;
}

std::vector<gdf_dtype_extra_info> column_dtype_infos(cudf::table const& table) {
  std::vector<gdf_dtype_extra_info> dtype_infos(table.num_columns());

  std::transform(table.begin(), table.end(), dtype_infos.begin(),
                 [](gdf_column const* col) { return col->dtype_info; });
  return dtype_infos;
}

bool has_nulls(cudf::table const& table) {
  return std::any_of(table.begin(), table.end(), [](gdf_column const* col) {
    return (nullptr != col->valid) and (col->null_count > 0);
  });
}

}  // namespace cudf
