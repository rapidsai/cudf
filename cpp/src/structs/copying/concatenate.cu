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
  // concatenate children.
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(columns.size());
  auto child_index = thrust::make_counting_iterator(0);
  std::transform(child_index,
                 child_index + columns[0].num_children(),
                 std::back_inserter(children),
                 [&columns, mr, stream](int child_index) {
                   std::vector<column_view> children;

                   auto col_index = thrust::make_counting_iterator(0);
                   std::transform(
                     col_index,
                     col_index + columns.size(),
                     std::back_inserter(children),
                     [&columns, child_index](int col_index) {
                       structs_column_view scv(columns[col_index]);

                       CUDF_EXPECTS(columns[0].num_children() == scv.num_children(),
                                    "Mismatch in number of children during struct concatenate");
                       CUDF_EXPECTS(
                         columns[0].child(child_index).type() == scv.child(child_index).type(),
                         "Mismatch in number of children during struct concatenate");
                       return scv.get_sliced_child(child_index);
                     });

                   return cudf::detail::concatenate(children, mr, stream);
                 });
  size_type total_length = children[0]->size();

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
