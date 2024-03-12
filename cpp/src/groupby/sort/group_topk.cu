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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuda/functional>
#include <thrust/transform_scan.h>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {

__global__ void compute_topk_indices(cudf::device_span<size_type const> group_offsets,
                                     cudf::device_span<size_type const> index_offsets,
                                     cudf::device_span<size_type> indices,
                                     size_type num_groups,
                                     size_type k)
{
  // Each group's output indices are processed by a single warp
  for (thread_index_type idx = threadIdx.x + blockDim.x * blockIdx.x;
       // TODO: overflow if num_groups > 2**26 - 1
       idx < cudf::detail::warp_size * num_groups;
       idx += gridDim.x * blockDim.x) {
    auto const group{idx / cudf::detail::warp_size};
    auto const lane{idx % cudf::detail::warp_size};
    if (group >= num_groups) break;
    auto const off1       = group_offsets[group + 1];
    auto const off0       = group_offsets[group];
    auto const k_loc      = min(off1 - off0, k);
    auto const out_offset = index_offsets[group];
    // TODO: these writes are uncoalesced
    for (size_type off = lane; off < k_loc; off += cudf::detail::warp_size) {
      indices[out_offset + off] = off0 + off;
    }
  }
}

std::unique_ptr<column> group_topk(column_view const& values,
                                   cudf::device_span<size_type const> group_sizes,
                                   cudf::device_span<size_type const> group_offsets,
                                   size_type num_groups,
                                   size_type k,
                                   order order,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto group_size_fn = cuda::proclaim_return_type<size_type>(
    [group_sizes, num_groups, k] __device__(size_type i) -> size_type {
      return i < num_groups ? min(k, group_sizes[i]) : size_type{0};
    });
  auto size_per_group = cudf::detail::make_counting_transform_iterator(0, group_size_fn);
  auto [offsets_column, output_size] = cudf::detail::make_offsets_child_column(
    size_per_group, size_per_group + num_groups, stream, mr);
  auto indices = rmm::device_uvector<size_type>(output_size, stream);
  // TODO: this will overflow if there are more than (2**26 - 1) groups
  cudf::detail::grid_1d const config{num_groups * cudf::detail::warp_size, 128};
  compute_topk_indices<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    group_offsets, offsets_column->view(), indices, num_groups, k);

  auto ordered_values = cudf::detail::segmented_sort_by_key(table_view{{values}},
                                                            table_view{{values}},
                                                            group_offsets,
                                                            {order},
                                                            {null_order::BEFORE},
                                                            stream,
                                                            mr);
  auto output_table   = cudf::detail::gather(ordered_values->view(),
                                           indices,
                                           out_of_bounds_policy::DONT_CHECK,
                                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr);
  return cudf::make_lists_column(num_groups,
                                 std::move(offsets_column),
                                 std::move(output_table->release()[0]),
                                 0,
                                 {},
                                 stream,
                                 mr);
}
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
