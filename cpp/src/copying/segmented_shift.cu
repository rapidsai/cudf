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

#include <cudf/copying.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/indexalator.cuh>
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
 * @brief TBA
 */
template <bool forward_shift, typename BoundaryIterator>
struct filter_functor {
  BoundaryIterator segment_bounds_begin;
  size_type num_segments;
  size_type offset;

  filter_functor(BoundaryIterator segment_bounds_begin, size_type num_segments, size_type offset)
    : segment_bounds_begin(segment_bounds_begin), num_segments(num_segments), offset(offset)
  {
  }

  __device__ bool operator()(size_type i)
  {
    auto segment_bound_idx =
      thrust::lower_bound(segment_bounds_begin, segment_bounds_begin + num_segments, i);
    return forward_shift ? *segment_bound_idx <= i and i < *segment_bound_idx + offset
                         : *segment_bound_idx + offset <= i and i < *segment_bound_idx;
  }
};

}  // anonymous namespace

/**
 * @brief Implementation of segmented shift
 *
 * TBA
 *
 * @tparam forward_shift If true, shifts element to the end of the segment.
 * @tparam BoudnaryIterator TBA
 *
 * @param segmented_values Segmented column to shift
 * @param segment_bound_begin TBA
 * @param num_segments TBA
 * @param offset The offset by which to shift the input
 * @param fill_value Fill value for indeterminable outputs
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Column where values are shifted in each segment
 */
template <bool forward_shift, typename BoundaryIterator>
std::unique_ptr<column> segmented_shift_impl(column_view const& segmented_values,
                                             BoundaryIterator segment_boundary_begin,
                                             std::size_t num_segments,
                                             size_type const& offset,
                                             scalar const& fill_value,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto input_pair_iterator =
    cudf::detail::indexalator_factory::make_input_pair_iterator(segmented_values) - offset;
  auto fill_pair_iterator = cudf::detail::indexalator_factory::make_input_pair_iterator(fill_value);
  auto filter =
    filter_functor<forward_shift, BoundaryIterator>{segment_boundary_begin, num_segments, offset};

  bool nullable = not fill_value.is_valid() or segmented_values.nullable();
  return copy_if_else(nullable,
                      input_pair_iterator,
                      input_pair_iterator + segmented_values.size(),
                      fill_pair_iterator,
                      filter,
                      segmented_values.type(),
                      stream,
                      mr);
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
                                      segment_offsets.size() - 1,
                                      offset,
                                      fill_value,
                                      stream,
                                      mr);
  } else {
    auto right_boundary_iterator =
      thrust::make_transform_iterator(segment_offsets.begin() + 1, [](auto i) { return i - 1; });
    return segmented_shift_impl<false>(segmented_values,
                                       right_boundary_iterator,
                                       segment_offsets.size() - 1,
                                       offset,
                                       fill_value,
                                       stream,
                                       mr);
  }
}

}  // namespace detail
}  // namespace cudf
