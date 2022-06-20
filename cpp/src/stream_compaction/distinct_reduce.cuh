/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "stream_compaction_common.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace cudf::detail {

/**
 * @brief Perform a reduction on each group of rows compared equal.
 *
 * For a given map with keys are row indices and were already inserted, perform a reduction on each
 * group of rows where each that are compared equal. This is essentially a reduce-by-key with keys
 * are rows compared equal.
 *
 * Depending on the `keep` parameter, the reduction operation is:
 * - If `keep == KEEP_FIRST`: min of row index.
 * - If `keep == KEEP_LAST`: max of row index.
 * - If `keep == KEEP_NONE`: sum number of row appearances.
 *
 * @return A device_uvector containing indices of distinct rows with desired behavior specified by
 *         the `keep` option.
 */
rmm::device_uvector<size_type> reduce_by_row(
  hash_map_type const& map,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const& preprocessed_input,
  size_type input_size,
  cudf::nullate::DYNAMIC has_nulls,
  duplicate_keep_option keep,
  null_equality nulls_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr);

}  // namespace cudf::detail
