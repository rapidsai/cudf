/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/legacy/column.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/error_utils.hpp>
#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <nvstrings/NVCategory.h>

#include <cuda_runtime.h>
#include <algorithm>

namespace cudf
{

/*
 * Initializes and returns gdf_column of the same type as the input.
 */
gdf_column empty_like(gdf_column const& input)
{
  CUDF_EXPECTS(input.size == 0 || input.data != 0, "Null input data");
  gdf_column output{};

  gdf_dtype_extra_info info = input.dtype_info;
  info.category = nullptr;

  CUDF_EXPECTS(GDF_SUCCESS == 
               gdf_column_view_augmented(&output, nullptr, nullptr, 0,
                                         input.dtype, 0, info, input.col_name),
               "Invalid column parameters");

  return output;
}

/*
 * Allocates a new column of the same size and type as the input.
 * Does not copy data.
 */
gdf_column allocate_like(gdf_column const& input, bool allocate_mask_if_exists, cudaStream_t stream)
{
  gdf_column output = empty_like(input);

  output.size = input.size;
  bool allocate_mask = allocate_mask_if_exists && (input.valid != nullptr);

  detail::allocate_column_fields(output, allocate_mask, stream);
  
  return output;
}

/*
 * Allocates a new column of specified size of the same type as the input.
 * Does not copy data.
 */
gdf_column allocate_like(gdf_column const& input, gdf_size_type size,
                         bool allocate_mask_if_exists, cudaStream_t stream)
{
  gdf_column output = empty_like(input);
  
  output.size = size;
  const auto byte_width = (input.dtype == GDF_STRING)
                        ? sizeof(std::pair<const char *, size_t>)
                        : cudf::size_of(input.dtype);
  RMM_TRY(RMM_ALLOC(&output.data, size * byte_width, stream));
  if ((input.valid != nullptr) && allocate_mask_if_exists) {
    size_t valid_size = gdf_valid_allocation_size(size);
    RMM_TRY(RMM_ALLOC(&output.valid, valid_size, stream));
  }
  
  return output;
}

/*
 * Creates a new column that is a copy of input
 */
gdf_column copy(gdf_column const& input, cudaStream_t stream)
{
  CUDF_EXPECTS(input.size == 0 || input.data != 0, "Null input data");

  gdf_column output = allocate_like(input, true, stream);
  output.null_count = input.null_count;
  if (input.size > 0) {
    const auto byte_width = (input.dtype == GDF_STRING)
                          ? sizeof(std::pair<const char *, size_t>)
                          : cudf::size_of(input.dtype);
    CUDA_TRY(cudaMemcpyAsync(output.data, input.data, input.size * byte_width,
                             cudaMemcpyDefault, stream));
    if (input.valid != nullptr) {
      size_t valid_size = gdf_valid_allocation_size(input.size);
      CUDA_TRY(cudaMemcpyAsync(output.valid, input.valid, valid_size,
                               cudaMemcpyDefault, stream));
    }

    output.null_count = input.null_count;
  }

  if (input.dtype == GDF_STRING_CATEGORY) {
    if (input.dtype_info.category != nullptr) {
      NVCategory *cat = static_cast<NVCategory*>(input.dtype_info.category);
      output.dtype_info.category = cat->copy();
    }
  }
  return output;
}

table empty_like(table const& t) {
  std::vector<gdf_column*> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [](gdf_column* out_col, gdf_column const* in_col) {
      out_col = new gdf_column{};
      *out_col = empty_like(*in_col);
      return out_col;
    });

  return table{columns.data(), static_cast<gdf_size_type>(columns.size())};
}

table allocate_like(table const& t, bool allocate_mask_if_exists, cudaStream_t stream) {
  std::vector<gdf_column*> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [allocate_mask_if_exists,stream](gdf_column* out_col, gdf_column const* in_col) {
      out_col = new gdf_column{};
      *out_col = allocate_like(*in_col,allocate_mask_if_exists,stream);
      return out_col;
    });

  return table{columns.data(), static_cast<gdf_size_type>(columns.size())};
}

table allocate_like(table const& t, gdf_size_type size, 
                    bool allocate_mask_if_exists, cudaStream_t stream) {
  std::vector<gdf_column*> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [size,allocate_mask_if_exists,stream](gdf_column* out_col, gdf_column const* in_col) {
      out_col = new gdf_column{};
      *out_col = allocate_like(*in_col,size,allocate_mask_if_exists,stream);
      return out_col;
    });

  return table{columns.data(), static_cast<gdf_size_type>(columns.size())};
}

table copy(table const& t, cudaStream_t stream) {
  std::vector<gdf_column*> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [stream](gdf_column* out_col, gdf_column const* in_col) {
      out_col = new gdf_column{};
      *out_col = copy(*in_col, stream);
      return out_col;
    });

  return table{columns.data(), static_cast<gdf_size_type>(columns.size())};
}

} // namespace cudf
