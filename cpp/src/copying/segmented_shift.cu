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
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

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
template <bool forward_shift, typename OffsetIterator>
struct filter_functor {
  OffsetIterator segment_offset_begin;
  OffsetIterator segment_offset_end;
  size_type offset;

  filter_functor(OffsetIterator segment_offset_begin,
                 OffsetIterator segment_offset_end,
                 size_type offset)
    : segment_offset_begin(segment_offset_begin),
      segment_offset_end(segment_offset_end),
      offset(offset)
  {
  }

  __device__ bool operator()(size_type const& i)
  {
    if (forward_shift) {
      auto segment_bound_idx =
        thrust::upper_bound(thrust::seq, segment_offset_begin, segment_offset_end, i) - 1;
      return not(*segment_bound_idx <= i and i < *segment_bound_idx + offset);
    } else {
      auto segment_bound_idx =
        thrust::upper_bound(thrust::seq, segment_offset_begin, segment_offset_end, i);
      return not(*segment_bound_idx + offset <= i and i < *segment_bound_idx);
    }
  }
};

template <typename PairIterator, typename ScalarIterator>
std::unique_ptr<column> segmented_shift_rep_impl(PairIterator input_pair_iterator,
                                                 ScalarIterator fill_pair_iterator,
                                                 bool nullable,
                                                 size_type offset,
                                                 device_span<size_type const> segment_offsets,
                                                 data_type value_type,
                                                 size_type column_size,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  size_type num_segments = segment_offsets.size() - 1;
  if (offset > 0) {
    filter_functor<true, decltype(segment_offsets.begin())> filter{
      segment_offsets.begin(), segment_offsets.end(), offset};
    return copy_if_else(nullable,
                        input_pair_iterator,
                        input_pair_iterator + column_size,
                        fill_pair_iterator,
                        filter,
                        value_type,
                        stream,
                        mr);
  } else {
    filter_functor<false, decltype(segment_offsets.begin())> filter{
      segment_offsets.begin(), segment_offsets.end(), offset};
    return copy_if_else(nullable,
                        input_pair_iterator,
                        input_pair_iterator + column_size,
                        fill_pair_iterator,
                        filter,
                        value_type,
                        stream,
                        mr);
  }
}

// template<typename PairIterator, typename ScalarIterator>
// std::unique_ptr<column> segmented_shift_string_impl(
//   PairIterator input_pair_iterator,
//   ScalarIterator fill_pair_iterator,
//   size_type offset,
//   device_span<size_type const> segment_offsets,
//   size_type column_size,
//   rmm::cuda_stream_view stream,
//   rmm::mr::device_memory_resource* mr
// )
// {
//   size_type num_segments = segment_offsets.size() - 1;
//   if (offset > 0) {
//     filter_functor<true, decltype(segment_offsets.begin())> filter{segment_offsets.begin(),
//     segment_offsets.end(), offset}; return strings::detail::copy_if_else(
//         input_pair_iterator,
//         input_pair_iterator + column_size,
//         fill_pair_iterator,
//         filter,
//         stream,
//         mr);
//   }
//   else {
//     filter_functor<false, decltype(segment_offsets.begin())> filter{segment_offsets.begin(),
//     segment_offsets.end(), offset}; return strings::detail::copy_if_else(
//       input_pair_iterator,
//       input_pair_iterator + column_size,
//       fill_pair_iterator,
//       filter,
//       stream,
//       mr);
//   }
// }

template <typename T, typename Enable = void>
struct segmented_shift_functor {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type for segmented_shift.");
  }
};

template <typename T>
struct segmented_shift_functor<T, std::enable_if_t<is_rep_layout_compatible<T>()>> {
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto values_device_view = column_device_view::create(segmented_values, stream);
    auto fill_pair_iterator = make_pair_iterator<T>(fill_value);

    bool nullable = not fill_value.is_valid() or segmented_values.nullable();

    if (segmented_values.has_nulls()) {
      auto input_pair_iterator = make_pair_iterator<T, true>(*values_device_view) - offset;
      return segmented_shift_rep_impl(input_pair_iterator,
                                      fill_pair_iterator,
                                      nullable,
                                      offset,
                                      segment_offsets,
                                      segmented_values.type(),
                                      segmented_values.size(),
                                      stream,
                                      mr);
    } else {
      auto input_pair_iterator = make_pair_iterator<T, false>(*values_device_view) - offset;
      return segmented_shift_rep_impl(input_pair_iterator,
                                      fill_pair_iterator,
                                      nullable,
                                      offset,
                                      segment_offsets,
                                      segmented_values.type(),
                                      segmented_values.size(),
                                      stream,
                                      mr);
    }
  }
};

template <>
struct segmented_shift_functor<string_view> {
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    // using T = string_view;

    // auto values_device_view = column_device_view::create(segmented_values, stream);
    // auto fill_pair_iterator = make_pair_iterator<T>(fill_value);
    // if (segmented_values.has_nulls()) {
    //   auto input_pair_iterator = make_pair_iterator<T, true>(*values_device_view) - offset;
    //   return segmented_shift_string_impl(
    //     input_pair_iterator, fill_pair_iterator, offset, segment_offsets,
    //     segmented_values.size(), stream, mr
    //   );
    // }
    // else {
    //   auto input_pair_iterator = make_pair_iterator<T, false>(*values_device_view) - offset;
    //   return segmented_shift_string_impl(
    //     input_pair_iterator, fill_pair_iterator, offset, segment_offsets,
    //     segmented_values.size(), stream, mr
    //   );
    // }
    CUDF_FAIL("segmented_shift does not support string_view yet");
  }
};

template <>
struct segmented_shift_functor<list_view> {
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("segmented_shift does not support list_view yet");
  }
};

template <>
struct segmented_shift_functor<struct_view> {
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("segmented_shift does not support struct_view yet");
  }
};

struct segmented_shift_functor_forwarder {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    segmented_shift_functor<T> shifter;
    return shifter(segmented_values, segment_offsets, offset, fill_value, stream, mr);
  }
};

}  // anonymous namespace

/**
 * @brief Implementation of segmented shift
 *
 * TBA
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
std::unique_ptr<column> segmented_shift(column_view const& segmented_values,
                                        device_span<size_type const> segment_offsets,
                                        size_type offset,
                                        scalar const& fill_value,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (segmented_values.is_empty()) { return empty_like(segmented_values); }

  return type_dispatcher<dispatch_storage_type>(segmented_values.type(),
                                                segmented_shift_functor_forwarder{},
                                                segmented_values,
                                                segment_offsets,
                                                offset,
                                                fill_value,
                                                stream,
                                                mr);
}

}  // namespace detail
}  // namespace cudf
