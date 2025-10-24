/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_count_valid(column_view const& values,
                                          cudf::device_span<size_type const> group_labels,
                                          size_type num_groups,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  auto result = make_numeric_column(
    data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

  if (num_groups == 0) { return result; }

  if (values.nullable()) {
    auto values_view = column_device_view::create(values, stream);

    // make_validity_iterator returns a boolean iterator that sums to 1 (1+1=1)
    // so we need to transform it to cast it to an integer type
    auto bitmask_iterator =
      thrust::make_transform_iterator(cudf::detail::make_validity_iterator(*values_view),
                                      cuda::proclaim_return_type<size_type>([] __device__(auto b) {
                                        return static_cast<size_type>(b);
                                      }));

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

std::unique_ptr<column> group_count_all(cudf::device_span<size_type const> group_offsets,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
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
