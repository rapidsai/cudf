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
std::unique_ptr<column> group_min_by(column_view const& structs_column,
                                     cudf::device_span<size_type const> group_labels,
                                     size_type num_groups,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const orders = structs_column.child(1);

  auto indices = type_dispatcher(orders.type(),
                                 group_reduction_dispatcher<aggregation::ARGMIN>{},
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

  auto res = cudf::detail::gather(table_view{{structs_column}},
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
