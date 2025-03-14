/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "join_common_utils.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

namespace cudf {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_inner_join(table_view const& left,
           table_view const& right,
           null_equality compare_nulls,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
 CUDF_FUNC_RANGE();

  // Sanity checks
  CUDF_EXPECTS(left.num_columns() == right.num_columns(), "Number of columns must match for a join");
  CUDF_EXPECTS(left.num_rows() > 0 && right.num_rows() > 0, "Input tables must not be empty");

  CUDF_EXPECTS(!cudf::has_nested_columns(left) && !cudf::has_nested_columns(right), "Don't have sorting logic for nested columns yet");
  /*
    left_segment_mask = [1, 0, 0, ...., 0] // length = left.num_rows
    right_segment_mask = [1, 0, 0, ...., 0] // length = right.num_rows
    for colidx in numcols {
      left_sorting_idx, left_segment_submask = segmented_sort(left.column(colidx), left_segment_mask);
      right_sorting_idx, right_segment_submask = segmented_sort(right.column(colidx), right_segment_mask);
      // merge submasks segmented by masks
      merge(left_sorting_idx, right_sorting_idx, left_segment_mask, right_segment_mask, left_segment_submask, right_segment_submask);
    }
  */


  // Step 1: Sort the input tables by keys
  auto sorted_left = table_view(); // Placeholder for sorted left table
  auto sorted_right = table_view(); // Placeholder for sorted right table

  // Allocate device vectors to store row indices
  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(right.num_rows(), stream, mr);

  // Step 2: Sort the indices based on keys using Thrust
  thrust::sequence(rmm::exec_policy(stream), left_indices->begin(), left_indices->end());
  thrust::sequence(rmm::exec_policy(stream), right_indices->begin(), right_indices->end());
  // TODO: Add sorting based on table keys.

  // Step 3: Merge the sorted tables to simulate the inner join
  auto output_left_indices = std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
  auto output_right_indices = std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
  // TODO: Perform merge operation.

  return {std::move(output_left_indices), std::move(output_right_indices)};
}

} //namespace cudf
