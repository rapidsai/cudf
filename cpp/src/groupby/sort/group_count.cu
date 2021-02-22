/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_count_valid(column_view const& values,
                                          rmm::device_vector<size_type> const& group_labels,
                                          size_type num_groups,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  auto result = make_numeric_column(
    data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

  if (num_groups == 0) { return result; }

  if (values.nullable()) {
    auto values_view = column_device_view::create(values);

    // make_validity_iterator returns a boolean iterator that sums to 1 (1+1=1)
    // so we need to transform it to cast it to an integer type
    auto bitmask_iterator =
      thrust::make_transform_iterator(cudf::detail::make_validity_iterator(*values_view),
                                      [] __device__(auto b) { return static_cast<size_type>(b); });

    thrust::reduce_by_key(rmm::exec_policy(stream),
                          group_labels.begin(),
                          group_labels.end(),
                          bitmask_iterator,
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<size_type>());
  } else {
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          group_labels.begin(),
                          group_labels.end(),
                          thrust::make_constant_iterator(1),
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<size_type>());
  }

  return result;
}

std::unique_ptr<column> group_count_all(rmm::device_vector<size_type> const& group_offsets,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");

  auto result = make_numeric_column(
    data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

  if (num_groups == 0) { return result; }

  thrust::adjacent_difference(rmm::exec_policy(stream),
                              group_offsets.begin() + 1,
                              group_offsets.end(),
                              result->mutable_view().begin<size_type>());
  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
