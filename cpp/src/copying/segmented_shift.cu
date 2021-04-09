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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {

namespace {

constexpr size_type SAFE_GATHER_IDX = 0;

/**
 * @brief Functor to determine the location to set `fill_value` for segmented shift.
 */
template <bool ForwardShift, typename BoundaryIterator>
struct segmented_shift_fill_functor {
  BoundaryIterator segment_bounds_begin;
  size_type offset;
  size_type segment_label, offset_to_bound;

  segmented_shift_fill_functor(BoundaryIterator segment_bounds_begin, size_type offset)
    : segment_bounds_begin(segment_bounds_begin), offset(offset)
  {
  }

  __device__ size_type operator()(size_type i)
  {
    if (ForwardShift) {  // offset > 0
      segment_label   = i / offset;
      offset_to_bound = i % offset;
    } else {  // offset < 0
      segment_label   = -i / offset;
      offset_to_bound = i % offset + offset + 1;
    }
    return *(segment_bounds_begin + segment_label) + offset_to_bound;
  }
};

}  // namespace

/**
 * @brief Implementation of segmented shift
 *
 * The first step is a global shift for `segmented_values`. The second step is to set the proper
 * locations to `fill_values`.
 *
 * @tparam BoundaryIterator Iterator type to the segment edge list
 *
 * @param segmented_values Segmented column to shift
 * @param segment_bound_begin Beginning of iterator range of the list that contains indices to the
 * segment's boundary. For forward shifts, the indices point to the segments' left boundaries, and
 * right boundaries otherwise
 * @param offset The offset by which to shift the input
 * @param fill_value Fill value for indeterminable outputs
 * @param num_segments The number of segments
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Column where values are shifted in each segment
 */
template <bool ForwardShift, typename BoundaryIterator>
std::unique_ptr<column> segmented_shift_impl(column_view const& segmented_values,
                                             BoundaryIterator segment_bound_begin,
                                             size_type offset,
                                             cudf::scalar const& fill_value,
                                             std::size_t num_segments,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  // Step 1: global shift
  auto shift_func = [col_size = segmented_values.size(), offset] __device__(size_type idx) {
    auto raw_shifted_idx = idx - offset;
    return static_cast<uint32_t>(
      raw_shifted_idx >= 0 and raw_shifted_idx < col_size ? raw_shifted_idx : SAFE_GATHER_IDX);
  };
  auto gather_iter_begin = cudf::detail::make_counting_transform_iterator(0, shift_func);

  auto shifted = cudf::detail::gather(table_view({segmented_values}),
                                      gather_iter_begin,
                                      gather_iter_begin + segmented_values.size(),
                                      out_of_bounds_policy::DONT_CHECK,
                                      stream);

  // Step 2: set `fill_value`
  auto scatter_map = make_numeric_column(
    data_type(type_id::UINT32), num_segments * std::abs(offset), mask_state::UNALLOCATED, stream);
  segmented_shift_fill_functor<ForwardShift, decltype(segment_bound_begin)> fill_func{
    segment_bound_begin, offset};
  auto scatter_map_iterator = cudf::detail::make_counting_transform_iterator(0, fill_func);
  thrust::copy(rmm::exec_policy(stream),
               scatter_map_iterator,
               scatter_map_iterator + scatter_map->view().size(),
               scatter_map->mutable_view().begin<size_type>());

  auto shifted_filled =
    cudf::detail::scatter({fill_value}, scatter_map->view(), shifted->view(), true, stream, mr);

  return std::move(shifted_filled->release()[0]);
}

std::unique_ptr<column> segmented_shift(column_view const& segmented_values,
                                        device_span<size_type const> segment_offsets,
                                        size_type offset,
                                        scalar const& fill_value,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (segmented_values.is_empty()) { return make_empty_column(segmented_values.type()); }

  if (offset > 0) {
    return segmented_shift_impl<true>(segmented_values,
                                      segment_offsets.begin(),
                                      offset,
                                      fill_value,
                                      segment_offsets.size() - 1,
                                      stream,
                                      mr);
  } else {
    auto rbound_iter = thrust::make_transform_iterator(segment_offsets.begin() + 1,
                                                       [] __device__(auto i) { return i - 1; });
    return segmented_shift_impl<false>(
      segmented_values, rbound_iter, offset, fill_value, segmented_values.size() - 1, stream, mr);
  }
}

}  // namespace detail
}  // namespace cudf
