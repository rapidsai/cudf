/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/structs/structs_column_view.hpp>

#include <structs/utilities.hpp>

#include <memory>

namespace cudf {
namespace structs {
namespace detail {

/**
 * @copydoc cudf::structs::detail::concatenate
 *
 */
std::unique_ptr<column> concatenate(
  std::vector<column_view> const& columns,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  // get ordered children
  auto ordered_children = extract_ordered_struct_children(columns);

  // concatenate them
  std::vector<std::unique_ptr<column>> children;
  children.reserve(columns[0].num_children());
  std::transform(ordered_children.begin(),
                 ordered_children.end(),
                 std::back_inserter(children),
                 [mr, stream](std::vector<column_view> const& cols) {
                   return cudf::detail::concatenate(cols, mr, stream);
                 });

  size_type const total_length = children[0]->size();

  // if any of the input columns have nulls, construct the output mask
  bool const has_nulls =
    std::any_of(columns.cbegin(), columns.cend(), [](auto const& col) { return col.has_nulls(); });
  rmm::device_buffer null_mask =
    create_null_mask(total_length, has_nulls ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED);
  if (has_nulls) {
    cudf::detail::concatenate_masks(columns, static_cast<bitmask_type*>(null_mask.data()), stream);
  }

  // assemble into outgoing list column
  return make_structs_column(total_length,
                             std::move(children),
                             has_nulls ? UNKNOWN_NULL_COUNT : 0,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
