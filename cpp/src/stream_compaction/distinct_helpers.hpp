/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "stream_compaction_common.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf::detail {

/**
 * @brief Return the reduction identity used to initialize results of `hash_reduce_by_row`.
 *
 * @param keep A value of `duplicate_keep_option` type, must not be `KEEP_ANY`.
 * @return The initial reduction value.
 */
auto constexpr reduction_init_value(duplicate_keep_option keep)
{
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST: return std::numeric_limits<size_type>::max();
    case duplicate_keep_option::KEEP_LAST: return std::numeric_limits<size_type>::min();
    case duplicate_keep_option::KEEP_NONE: return size_type{0};
    default: CUDF_UNREACHABLE("This function should not be called with KEEP_ANY");
  }
}

/**
 * @brief Perform a reduction on groups of rows that are compared equal.
 *
 * This is essentially a reduce-by-key operation with keys are non-contiguous rows and are compared
 * equal. A hash table is used to find groups of equal rows.
 *
 * Depending on the `keep` parameter, the reduction operation for each row group is:
 * - If `keep == KEEP_FIRST`: min of row indices in the group.
 * - If `keep == KEEP_LAST`: max of row indices in the group.
 * - If `keep == KEEP_NONE`: count of equivalent rows (group size).
 *
 * Note that this function is not needed when `keep == KEEP_NONE`.
 *
 * At the beginning of the operation, the entire output array is filled with a value given by
 * the `reduction_init_value()` function. Then, the reduction result for each row group is written
 * into the output array at the index of an unspecified row in the group.
 *
 * @param map The auxiliary map to perform reduction
 * @param preprocessed_input The preprocessed of the input rows for computing row hashing and row
 *        comparisons
 * @param num_rows The number of all input rows
 * @param has_nulls Indicate whether the input rows has any nulls at any nested levels
 * @param has_nested_columns Indicates whether the input table has any nested columns
 * @param keep The parameter to determine what type of reduction to perform
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether NaN values in floating point column should be
 *        considered equal.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device_uvector containing the reduction results
 */
rmm::device_uvector<size_type> reduce_by_row(
  hash_map_type const& map,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const preprocessed_input,
  size_type num_rows,
  cudf::nullate::DYNAMIC has_nulls,
  bool has_nested_columns,
  duplicate_keep_option keep,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
