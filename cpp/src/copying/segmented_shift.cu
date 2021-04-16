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

#include <cudf/debug_printers.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Functor to determine the index to gather from in result column,
 *        out-of-bounds index is given for undetermined rows
 */
template <bool forward_shift, typename OffsetIterator>
struct segmented_shift_gather_functor {
  OffsetIterator segment_offsets_begin;
  OffsetIterator segment_offsets_end;
  size_type offset;
  size_type out_of_bounds_index;

  segmented_shift_gather_functor(OffsetIterator segment_offsets_begin,
                                 OffsetIterator segment_offsets_end,
                                 size_type offset,
                                 size_type out_of_bound_index)
    : segment_offsets_begin(segment_offsets_begin),
      segment_offsets_end(segment_offsets_end),
      offset(offset),
      out_of_bounds_index(out_of_bound_index)
  {
  }

  __device__ size_type operator()(size_type i)
  {
    bool is_determined;
    if (forward_shift) {
      auto segment_bound_idx =
        thrust::upper_bound(thrust::seq, segment_offsets_begin, segment_offsets_end, i) - 1;
      is_determined = not(*segment_bound_idx <= i and i < *segment_bound_idx + offset);
    } else {
      auto segment_bound_idx =
        thrust::upper_bound(thrust::seq, segment_offsets_begin, segment_offsets_end, i);
      is_determined = not(*segment_bound_idx + offset <= i and i < *segment_bound_idx);
    }
    return is_determined ? i - offset : out_of_bounds_index;
  }
};

/**
 * @brief Functor to determine the location to set `fill_value` for segmented shift.
 */
template <bool forward_shift, typename BoundaryIterator>
struct segmented_shift_fill_functor {
  BoundaryIterator segment_bounds_begin;
  size_type offset;

  segmented_shift_fill_functor(BoundaryIterator segment_bounds_begin, size_type offset)
    : segment_bounds_begin(segment_bounds_begin), offset(offset)
  {
  }

  __device__ size_type operator()(size_type i)
  {
    return forward_shift ? *(segment_bounds_begin + i / offset) + (i % offset)
                         : *(segment_bounds_begin - i / offset) + (i % offset + offset + 1);
  }
};

}  // namespace

/**
 * @brief Implementation of segmented shift
 *
 * The first step is a global shift for `segmented_values`. The second step is to set the proper
 * locations to `fill_values`.
 *
 * @tparam forward_shift If true, shifts element to the end of the segment.
 * @tparam OffsetIterator TBA
 *
 * @param segmented_values Segmented column to shift
 * @param segment_offset_begin TBA
 * @param segment_offset_end TBA
 * @param offset The offset by which to shift the input
 * @param fill_value Fill value for indeterminable outputs
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Column where values are shifted in each segment
 */
template <bool forward_shift, typename OffsetIterator>
std::unique_ptr<column> segmented_shift_impl(column_view const& segmented_values,
                                             OffsetIterator segment_offset_begin,
                                             OffsetIterator segment_offset_end,
                                             size_type offset,
                                             cudf::scalar const& fill_value,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  segmented_shift_gather_functor<forward_shift, decltype(segment_offset_begin)>
    shift_gather_functor(segment_offset_begin, segment_offset_end, offset, segmented_values.size());
  auto gather_iter_begin = cudf::detail::make_counting_transform_iterator(0, shift_gather_functor);

  if (not fill_value.is_valid(stream)) {
    auto result = cudf::detail::gather(table_view({segmented_values}),
                                       gather_iter_begin,
                                       gather_iter_begin + segmented_values.size(),
                                       out_of_bounds_policy::NULLIFY,
                                       stream,
                                       mr);
    return std::move(result->release()[0]);
  } else {
    auto shifted = cudf::detail::gather(table_view({segmented_values}),
                                        gather_iter_begin,
                                        gather_iter_begin + segmented_values.size(),
                                        out_of_bounds_policy::NULLIFY,
                                        stream);

    auto num_segments = cudf::distance(segment_offset_begin, segment_offset_end) - 1;
    // Worst case scenario, `fill_value` can go to all locations of the input column
    auto scatter_map_size = std::min(num_segments * std::abs(offset), segmented_values.size());
    auto scatter_map      = make_numeric_column(
      data_type(type_id::INT32), scatter_map_size, mask_state::UNALLOCATED, stream);
    auto scatter_map_mutable_itr = scatter_map->mutable_view().template begin<int32_t>();

    if (forward_shift) {
      segmented_shift_fill_functor<forward_shift, decltype(segment_offset_begin)> fill_func{
        segment_offset_begin, offset};
      auto scatter_map_iterator = cudf::detail::make_counting_transform_iterator(0, fill_func);

      thrust::copy(rmm::exec_policy(stream),
                   scatter_map_iterator,
                   scatter_map_iterator + scatter_map->view().size(),
                   scatter_map_mutable_itr);
    } else {
      auto segment_bound_begin = thrust::make_transform_iterator(
        segment_offset_begin + 1, [] __device__(auto i) { return i - 1; });
      segmented_shift_fill_functor<forward_shift, decltype(segment_bound_begin)> fill_func{
        segment_bound_begin, offset};
      auto scatter_map_iterator = cudf::detail::make_counting_transform_iterator(0, fill_func);
      thrust::copy(rmm::exec_policy(stream),
                   scatter_map_iterator,
                   scatter_map_iterator + scatter_map->view().size(),
                   scatter_map_mutable_itr);
    }

    auto result =
      cudf::detail::scatter({fill_value}, scatter_map->view(), shifted->view(), false, stream, mr);
    return std::move(result->release()[0]);
  }
}

std::unique_ptr<column> segmented_shift(column_view const& segmented_values,
                                        device_span<size_type const> segment_offsets,
                                        size_type offset,
                                        scalar const& fill_value,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (segmented_values.is_empty()) { return empty_like(segmented_values); }

  if (offset > 0) {
    return segmented_shift_impl<true>(segmented_values,
                                      segment_offsets.begin(),
                                      segment_offsets.end(),
                                      offset,
                                      fill_value,
                                      stream,
                                      mr);
  } else {
    return segmented_shift_impl<false>(segmented_values,
                                       segment_offsets.begin(),
                                       segment_offsets.end(),
                                       offset,
                                       fill_value,
                                       stream,
                                       mr);
  }
}

}  // namespace detail
}  // namespace cudf
