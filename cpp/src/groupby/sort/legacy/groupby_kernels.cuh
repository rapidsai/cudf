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
#ifndef GROUPBY_KERNELS_CUH
#define GROUPBY_KERNELS_CUH

#include <cudf/groupby.hpp>
#include "../common/type_info.hpp"
#include "../common/kernel_utils.hpp"
#include "utilities/device_atomics.cuh"

namespace cudf {
namespace groupby {
namespace sort {


/**---------------------------------------------------------------------------*
 @brief  Compute the aggregation(s) of corresponding rows in the `values` input 
 * table using the key sorted order and the group labels.
 * 
 * The aggregations(s) is computed  by  performing elementwise aggregation
 * operations between a target (the corresponding output value row for group_id)
 * and source   (the corresponding input value row using the current keys sorted
 * order). This aggregation(s) are done for every element `j` in the row by
 * applying aggregation operation `j` between the new and existing element.
 *
 * @tparam skip_rows_with_nulls Indicates if rows in `input_keys` containing
 * null values should be skipped. It `true`, it is assumed `row_bitmask` is a
 * bitmask where bit `i` indicates the presence of a null value in row `i`.
 * @tparam values_have_nulls Indicates if rows in `input_values` contain null
 * values
 * @param input_values The table whose rows will be aggregated in the 
 * output values table 
 * @param output_values Table that stores the results of aggregating rows of
 * `input_values`.
 * @param key_sorted_order The sorted order of the `keys` in sort-based groupby
 * @param group_labels The group labels corresponding to the sorted order of `keys`
 * @param skip_null_keys User input option to whether to include or not null
 *  keys in groupby
 * @param ops The set of aggregation operations to perform accross the columns
 * of the `input_values` rows
 * @param row_bitmask Bitmask where bit `i` indicates the presence of a null
 * value in row `i` of `input_keys`. Only used if `skip_rows_with_nulls` is
 * `true` and skip_null_keys option is true.
 *---------------------------------------------------------------------------**/
template <bool skip_rows_with_nulls, bool values_have_nulls>
__global__ void aggregate_all_rows(
    device_table input_values,
    device_table output_values,
    gdf_size_type const* key_sorted_order, 
    gdf_size_type const* group_labels, 
    bool skip_null_keys,
    operators* ops,
    bit_mask::bit_mask_t const* const __restrict__ row_bitmask) {

  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while (i < input_values.num_rows()) {
    if (skip_null_keys and skip_rows_with_nulls and not bit_mask::is_valid(row_bitmask, key_sorted_order[i])) {
      i += blockDim.x * gridDim.x;
      continue;
    }
    auto group_index = group_labels[i];
    aggregate_row<values_have_nulls>(output_values, group_index,
                                      input_values, key_sorted_order[i], ops);
    i += blockDim.x * gridDim.x;
  }
}

}  // namespace sort
}  // namespace groupby
}  // namespace cudf

#endif
