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

template <bool skip_rows_with_nulls, bool values_have_nulls>
__global__ void aggregate_all_rows(
    gdf_size_type num_rows, 
    device_table input_values,
    device_table output_values,
    gdf_size_type* key_sorted_order, 
    gdf_size_type* group_labels, 
    operators* ops,
    bit_mask::bit_mask_t const* const __restrict__ row_bitmask,
    bool skip_null_keys) {
  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while (i < num_rows) {
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
