/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/groupby/group_replace_nulls.hpp>
#include <cudf/detail/replace/nulls.cuh>
#include <cudf/replace.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>

#include <utility>

namespace cudf {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_replace_nulls(cudf::column_view const& grouped_value,
                                            device_span<size_type const> group_labels,
                                            cudf::replace_policy replace_policy,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  cudf::size_type size = grouped_value.size();

  auto device_in = cudf::column_device_view::create(grouped_value);
  auto index     = thrust::make_counting_iterator<cudf::size_type>(0);
  auto valid_it  = cudf::detail::make_validity_iterator(*device_in);
  auto in_begin  = thrust::make_zip_iterator(thrust::make_tuple(index, valid_it));

  rmm::device_uvector<cudf::size_type> gather_map(size, stream);
  auto gm_begin = thrust::make_zip_iterator(
    thrust::make_tuple(gather_map.begin(), thrust::make_discard_iterator()));

  auto func = cudf::detail::replace_policy_functor();
  thrust::equal_to<cudf::size_type> eq;
  if (replace_policy == cudf::replace_policy::PRECEDING) {
    thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                  group_labels.begin(),
                                  group_labels.begin() + size,
                                  in_begin,
                                  gm_begin,
                                  eq,
                                  func);
  } else {
    auto gl_rbegin = thrust::make_reverse_iterator(group_labels.begin() + size);
    auto in_rbegin = thrust::make_reverse_iterator(in_begin + size);
    auto gm_rbegin = thrust::make_reverse_iterator(gm_begin + size);
    thrust::inclusive_scan_by_key(
      rmm::exec_policy(stream), gl_rbegin, gl_rbegin + size, in_rbegin, gm_rbegin, eq, func);
  }

  auto output = cudf::detail::gather(cudf::table_view({grouped_value}),
                                     gather_map.begin(),
                                     gather_map.end(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);

  return std::move(output->release()[0]);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
