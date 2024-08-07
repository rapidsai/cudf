/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/detail/gather.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/gather.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_max_by(column_view const& structs_column,
                                     cudf::device_span<size_type const> group_labels,
                                     size_type num_groups,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const values = structs_column.child(0);
  auto const orders = structs_column.child(1);

  // Nulls in orders column should be excluded, so we need to create a new bitmask
  // that is the combination of the nulls in both values and orders columns.
  auto const new_mask_buffer_cnt = bitmask_and(table_view{{values, orders}});

  std::vector<column_view> struct_children(values.num_children());
  for (size_type i = 0; i < values.num_children(); i++) {
    struct_children[i] = values.child(i);
  }
  
  column_view const values_null_excluded(
      values.type(),
      values.size(),
      values.head(),
      static_cast<bitmask_type const*>(new_mask_buffer_cnt.first.data()),
      new_mask_buffer_cnt.second,
      values.offset(),
      struct_children);

  column_view const structs_column_null_excluded(
      structs_column.type(),
      structs_column.size(),
      structs_column.head(),
      nullptr,
      0,
      structs_column.offset(),
      {values_null_excluded, orders});

  auto const indices = type_dispatcher(orders.type(),
                                 group_reduction_dispatcher<aggregation::ARGMAX>{},
                                 orders,
                                 num_groups,
                                 group_labels,
                                 stream,
                                 mr);
  
  column_view const null_removed_map(
      data_type(type_to_id<size_type>()),
      indices->size(),
      static_cast<void const*>(indices->view().template data<size_type>()),
      nullptr,
      0);

  auto res = cudf::detail::gather(table_view{{structs_column_null_excluded}},
                                  null_removed_map,
                                  indices->nullable() ? cudf::out_of_bounds_policy::NULLIFY
                                                  : cudf::out_of_bounds_policy::DONT_CHECK,
                                  cudf::detail::negative_index_policy::NOT_ALLOWED,
                                  stream,
                                  mr);

  return std::move(res->release()[0]);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
