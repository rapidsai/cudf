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
#include "cudf.h"
#include "rmm/rmm.h"
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "utilities/type_dispatcher.hpp"
#include <thrust/system/cuda/execution_policy.h>
#include <rmm/thrust_rmm_allocator.h>


namespace { // unnamed namespace
  template<typename ColumnType>
  void binary_search_bound(
    bool upper_bound,
    void *bins,
    const size_t& num_bins,
    void *col_data,
    const size_t& num_rows,
    uint32_t *output)
  {
    ColumnType const * const p_bins{static_cast<ColumnType*>(bins)};
    ColumnType const * const p_values{static_cast<ColumnType const*>(col_data)};
    if (upper_bound)
      thrust::upper_bound(rmm::exec_policy()->on(0), p_bins, p_bins + num_bins, p_values, p_values + num_rows, output, thrust::less_equal<ColumnType>());
    else
      thrust::lower_bound(rmm::exec_policy()->on(0), p_bins, p_bins + num_bins, p_values, p_values + num_rows, output, thrust::less_equal<ColumnType>());
  }
} // end unnamed namespace

gdf_error gdf_digitize(gdf_column* col,
                       void* bins,   // same type as col
                       size_t num_bins,
                       bool right,
                       gdf_column* out_col_indices)
{

  using size_type = int64_t;
  const size_type num_rows = col->size;
  thrust::device_vector<uint32_t> output(size_type);
  uint32_t * p_output = static_cast<uint32_t*>(out_col_indices->data);
  const gdf_dtype gdf_input_type = col->dtype;

  switch (gdf_input_type) {
    case GDF_INT8:    { binary_search_bound<int8_t>( right, bins, num_bins, col->data,num_rows, p_output); break; }
    case GDF_INT16:   { binary_search_bound<int16_t>( right, bins, num_bins, col->data,num_rows, p_output); break; }
    case GDF_INT32:   { binary_search_bound<int32_t>( right, bins, num_bins, col->data,num_rows, p_output); break; }
    case GDF_INT64:   { binary_search_bound<int64_t>( right, bins, num_bins, col->data,num_rows, p_output); break; }
    case GDF_FLOAT32: { binary_search_bound<float>( right, bins, num_bins, col->data,num_rows, p_output); break; }
    case GDF_FLOAT64: { binary_search_bound<double>( right, bins, num_bins, col->data,num_rows, p_output);  break; }
    default: return GDF_UNSUPPORTED_DTYPE;
  }

  CUDA_CHECK_LAST();

  return GDF_SUCCESS;
}
