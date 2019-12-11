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
#include <cudf/cudf.h>
#include <rmm/rmm.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/legacy/error_utils.hpp>
#include <thrust/system/cuda/execution_policy.h>
#include <rmm/thrust_rmm_allocator.h>


namespace { // unnamed namespace
  struct binary_search_bound{

    template<typename ColumnType>
    void operator()(
      bool upper_bound,
      void *bins,
      const size_t& num_bins,
      void *col_data,
      const size_t& num_rows,
      cudf::size_type *output)
    {
      ColumnType const * const p_bins{static_cast<ColumnType*>(bins)};
      ColumnType const * const p_values{static_cast<ColumnType const*>(col_data)};
      if (upper_bound)
        thrust::upper_bound(rmm::exec_policy()->on(0), p_bins, p_bins + num_bins, p_values, p_values + num_rows, output, thrust::less_equal<ColumnType>());
      else
        thrust::lower_bound(rmm::exec_policy()->on(0), p_bins, p_bins + num_bins, p_values, p_values + num_rows, output, thrust::less_equal<ColumnType>());
    }
  };
} // end unnamed namespace

gdf_error gdf_digitize(gdf_column* col,
                       gdf_column* bins,   // same type as col
                       bool right,
                       cudf::size_type out_indices[])
{
  GDF_REQUIRE(nullptr != col, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != bins, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != out_indices, GDF_DATASET_EMPTY);

  GDF_REQUIRE(col->dtype == bins->dtype, GDF_DTYPE_MISMATCH);

  // TODO: Handle when col or bins have null values
  GDF_REQUIRE(!col->null_count, GDF_VALIDITY_UNSUPPORTED);
  GDF_REQUIRE(!bins->null_count, GDF_VALIDITY_UNSUPPORTED);

  cudf::type_dispatcher(col->dtype,
                        binary_search_bound{},
                        right, bins->data, bins->size, col->data, col->size, out_indices);

  CHECK_CUDA(0);

  return GDF_SUCCESS;
}
