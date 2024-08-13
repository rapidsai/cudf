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

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/detail/gather.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/gather.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_argmin(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto indices = type_dispatcher(values.type(),
                                 group_reduction_dispatcher<aggregation::ARGMIN>{},
                                 values,
                                 num_groups,
                                 group_labels,
                                 stream,
                                 mr);

  auto indices_view = indices->mutable_view();

  thrust::gather_if(rmm::exec_policy(stream),
                    indices_view.begin<size_type>(),    // map first
                    indices_view.end<size_type>(),      // map last
                    indices_view.begin<size_type>(),    // stencil
                    key_sort_order.begin<size_type>(),  // input
                    indices_view.begin<size_type>(),    // result
                    [] __device__(auto i) {return (i != cudf::detail::ARGMIN_SENTINEL); });
  return indices;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
