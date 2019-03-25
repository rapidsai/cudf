/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#include <thrust/copy.h>
#include <cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <stream_compaction.hpp>
#include <bitmask/legacy_bitmask.hpp>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <utilities/wrapper_types.hpp>

namespace cudf {

namespace {
struct nonnull_and_true {
  nonnull_and_true(gdf_column const boolean_mask)
      : data{static_cast<cudf::bool8*>(boolean_mask.data)},
        bitmask{boolean_mask.valid} {
    CUDF_EXPECTS(boolean_mask.dtype == GDF_BOOL, "Expected boolean column");
    CUDF_EXPECTS(boolean_mask.data != nullptr, "Null boolean_mask data");
    CUDF_EXPECTS(boolean_mask.valid != nullptr, "Null boolean_mask bitmask");
  }

  __device__ bool operator()(gdf_index_type i) {
    return (cudf::true_v == data[i]) && gdf_is_valid(bitmask, i);
  }

 private:
  cudf::bool8 const * const data;
  gdf_valid_type const * const bitmask;
};
}  // namespace

/**
 * @brief Filters a column using a column of boolean values as a mask.
 *
 */
gdf_column apply_boolean_mask(gdf_column const *input,
                              gdf_column const *boolean_mask) {
  CUDF_EXPECTS(nullptr != input, "Null input");
  CUDF_EXPECTS(nullptr != boolean_mask, "Null boolean_mask");
  CUDF_EXPECTS(input->size == boolean_mask->size, "Column size mismatch");
  CUDF_EXPECTS(boolean_mask->dtype == GDF_BOOL, "Mask must be Boolean type");

  // High Level Algorithm:
  // First, compute a `gather_map` from the boolean_mask that will gather
  // input[i] if boolean_mask[i] is non-null and "true".
  // Second, use the `gather_map` to gather elements from the `input` column
  // into the `output` column

  // We don't know the exact size of the gather_map a priori, but we know it's
  // upper bounded by the size of the boolean_mask
  rmm::device_vector<gdf_index_type> gather_map(boolean_mask->size);

  // Returns an iterator to the end of the gather_map
  auto end = thrust::copy_if(
      rmm::exec_policy()->on(0), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(boolean_mask->size),
      thrust::make_counting_iterator(0), gather_map.begin(),
      nonnull_and_true{*boolean_mask});

  // Use the returned iterator to determine the size of the gather_map
  gdf_size_type output_size{
      static_cast<gdf_size_type>(end - gather_map.begin())};

  // Allocate/initialize output column
  gdf_size_type column_byte_width{gdf_dtype_size(input->dtype)};

  gdf_column output{};

  return output;
}

}  // namespace cudf