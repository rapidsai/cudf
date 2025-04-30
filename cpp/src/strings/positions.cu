/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "positions.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::strings::detail {

std::unique_ptr<column> create_offsets_from_positions(strings_column_view const& input,
                                                      device_span<int64_t const> const& positions,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // first, create a vector of string indices for each position
  auto indices = rmm::device_uvector<size_type>(positions.size(), stream);
  thrust::upper_bound(rmm::exec_policy_nosync(stream),
                      d_offsets,
                      d_offsets + input.size(),
                      positions.begin(),
                      positions.end(),
                      indices.begin());

  // compute position offsets per string
  auto counts = rmm::device_uvector<size_type>(input.size(), stream);
  // memset to zero-out the counts for any null-entries or strings with no positions
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), counts.begin(), counts.end(), 0);

  // next, count the number of positions per string
  auto d_counts  = counts.data();
  auto d_indices = indices.data();
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<int64_t>(0),
    positions.size(),
    [d_indices, d_counts] __device__(int64_t idx) {
      auto const str_idx = d_indices[idx] - 1;
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*(d_counts + str_idx)};
      ref.fetch_add(1L, cuda::std::memory_order_relaxed);
    });

  // finally, convert the counts into offsets
  return std::get<0>(
    cudf::strings::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr));
}

}  // namespace cudf::strings::detail
