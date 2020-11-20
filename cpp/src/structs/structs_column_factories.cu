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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <memory>

namespace cudf {
namespace {
// Helper function to superimpose validity of parent struct
// over the specified member (child) column.
void superimpose_parent_nullmask(bitmask_type const* parent_null_mask,
                                 std::size_t parent_null_mask_size,
                                 size_type parent_null_count,
                                 column& child,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  if (!child.nullable()) {
    // Child currently has no null mask. Copy parent's null mask.
    child.set_null_mask(rmm::device_buffer{parent_null_mask, parent_null_mask_size, stream, mr});
    child.set_null_count(parent_null_count);
  } else {
    // Child should have a null mask.
    // `AND` the child's null mask with the parent's.

    auto data_type{child.type()};
    auto num_rows{child.size()};

    auto current_child_mask = child.mutable_view().null_mask();

    cudf::detail::inplace_bitmask_and(current_child_mask,
                                      {reinterpret_cast<bitmask_type const*>(parent_null_mask),
                                       reinterpret_cast<bitmask_type const*>(current_child_mask)},
                                      {0, 0},
                                      child.size(),
                                      stream,
                                      mr);
    child.set_null_count(UNKNOWN_NULL_COUNT);
  }

  // If the child is also a struct, repeat for all grandchildren.
  if (child.type().id() == cudf::type_id::STRUCT) {
    const auto current_child_mask = child.mutable_view().null_mask();
    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(child.num_children()),
                  [&current_child_mask, &child, parent_null_mask_size, stream, mr](auto i) {
                    superimpose_parent_nullmask(current_child_mask,
                                                parent_null_mask_size,
                                                UNKNOWN_NULL_COUNT,
                                                child.child(i),
                                                stream,
                                                mr);
                  });
  }
}
}  // namespace

/// Column factory that adopts child columns.
std::unique_ptr<cudf::column> make_structs_column(
  size_type num_rows,
  std::vector<std::unique_ptr<column>>&& child_columns,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(null_count <= 0 || !null_mask.is_empty(),
               "Struct column with nulls must be nullable.");

  CUDF_EXPECTS(std::all_of(child_columns.begin(),
                           child_columns.end(),
                           [&](auto const& child_col) { return num_rows == child_col->size(); }),
               "Child columns must have the same number of rows as the Struct column.");

  if (!null_mask.is_empty()) {
    for (auto& child : child_columns) {
      superimpose_parent_nullmask(static_cast<bitmask_type const*>(null_mask.data()),
                                  null_mask.size(),
                                  null_count,
                                  *child,
                                  stream,
                                  mr);
    }
  }

  return std::make_unique<column>(
    cudf::data_type{type_id::STRUCT},
    num_rows,
    rmm::device_buffer{0, stream, mr},  // Empty data buffer. Structs hold no data.
    null_mask,
    null_count,
    std::move(child_columns));
}

}  // namespace cudf
