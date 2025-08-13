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

#include "sort_column_impl.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

namespace cudf {
namespace detail {
namespace {

CUDF_KERNEL void resolve_segment_indices(device_span<size_type const> d_offsets,
                                         size_type k,
                                         device_span<size_type> d_indices,
                                         device_span<size_type> d_segment_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= d_indices.size()) { return; }

  auto const sitr = thrust::upper_bound(thrust::seq, d_offsets.begin(), d_offsets.end(), tid);
  auto const segment_start = *(sitr - 1);
  auto const segment_end   = *sitr;
  auto const index         = tid - segment_start;
  if (index >= k) { d_indices[tid] = -1; }  // mark values outside of top k

  if (index == 0) {
    auto const segment_size  = segment_end - segment_start;
    auto const segment_index = thrust::distance(d_offsets.begin(), sitr) - 1;
    // segment is k or less elements
    d_segment_sizes[segment_index] = cuda::std::min(k, segment_size);
  }
}
}  // namespace

std::unique_ptr<column> top_k_segmented_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order sort_order,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k > 0, "k must be greater than 0", std::invalid_argument);
  CUDF_EXPECTS(segment_offsets.size() > 0,
               "segment_offsets must have at least one element",
               std::invalid_argument);
  auto const size_data_type = data_type{type_to_id<size_type>()};
  CUDF_EXPECTS(segment_offsets.type() == size_data_type,
               "segment_offsets must be of type INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(segment_offsets.null_count() == 0,
               "segment_offsets must not have nulls",
               std::invalid_argument);

  auto const nulls   = sort_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const indices = cudf::detail::segmented_sorted_order(
    cudf::table_view({col}), segment_offsets, {sort_order}, {nulls}, stream, temp_mr);
  if (indices->view().is_empty()) {
    return cudf::make_empty_lists_column(size_data_type, stream, mr);
  }

  auto const d_indices = indices->mutable_view().begin<size_type>();
  auto segment_sizes   = rmm::device_uvector<size_type>(segment_offsets.size() - 1, stream);
  auto span_indices = device_span<size_type>{d_indices, static_cast<std::size_t>(indices->size())};
  auto const grid   = cudf::detail::grid_1d(indices->size(), 256);
  resolve_segment_indices<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
    segment_offsets, k, span_indices, segment_sizes);
  auto [offsets, total] =
    cudf::detail::make_offsets_child_column(segment_sizes.begin(), segment_sizes.end(), stream, mr);
  auto result =
    cudf::make_fixed_width_column(size_data_type, total, mask_state::UNALLOCATED, stream, mr);
  auto d_result = result->mutable_view().begin<size_type>();
  thrust::remove_copy(
    rmm::exec_policy(stream), d_indices, d_indices + indices->size(), d_result, -1);
  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(
    num_rows, std::move(offsets), std::move(result), 0, rmm::device_buffer{}, stream, mr);
}

std::unique_ptr<column> top_k_segmented(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order sort_order,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (col.is_empty()) { return cudf::make_empty_column(col.type()); }

  auto ordered =
    cudf::detail::top_k_segmented_order(col, segment_offsets, k, sort_order, stream, mr);
  auto lv = cudf::lists_column_view(ordered->view());
  if (lv.is_empty()) { return cudf::make_empty_lists_column(col.type(), stream, mr); }

  auto result         = cudf::detail::gather(cudf::table_view({col}),
                                     lv.child(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  auto offsets        = std::move(ordered->release().children.front());
  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(num_rows,
                           std::move(offsets),
                           std::move(result->release().front()),
                           0,
                           rmm::device_buffer{},
                           stream,
                           mr);
}

}  // namespace detail

std::unique_ptr<column> top_k_segmented(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order sort_order,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_segmented(col, segment_offsets, k, sort_order, stream, mr);
}

std::unique_ptr<column> top_k_segmented_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order sort_order,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_segmented_order(col, segment_offsets, k, sort_order, stream, mr);
}
}  // namespace cudf
