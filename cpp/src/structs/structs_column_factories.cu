/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <memory>
namespace cudf {

/// Column factory that adopts child columns.
std::unique_ptr<cudf::column> make_structs_column(
  size_type num_rows,
  std::vector<std::unique_ptr<column>>&& child_columns,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(null_count <= 0 || !null_mask.is_empty(),
               "Struct column with nulls must be nullable.");

  CUDF_EXPECTS(std::all_of(child_columns.begin(),
                           child_columns.end(),
                           [&](auto const& child_col) { return num_rows == child_col->size(); }),
               "Child columns must have the same number of rows as the Struct column.");

  if (!null_mask.is_empty()) {
    for (auto& child : child_columns) {
      child = structs::detail::superimpose_nulls(static_cast<bitmask_type const*>(null_mask.data()),
                                                 null_count,
                                                 std::move(child),
                                                 stream,
                                                 mr);
    }
  }

  return std::make_unique<column>(cudf::data_type{type_id::STRUCT},
                                  num_rows,
                                  rmm::device_buffer{},  // Empty data buffer. Structs hold no data.
                                  std::move(null_mask),
                                  null_count,
                                  std::move(child_columns));
}

}  // namespace cudf
