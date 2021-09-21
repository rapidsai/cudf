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

#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/copy_if_else.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Helper function to invoke general `copy_if_else`
 */
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
  if (offset > 0) {
    auto filter = [segment_offsets, offset] __device__(auto const& i) {
      auto segment_bound_idx =
        thrust::upper_bound(thrust::seq, segment_offsets.begin(), segment_offsets.end(), i) - 1;
      return not(*segment_bound_idx <= i and i < *segment_bound_idx + offset);
    };
    return copy_if_else(nullable,
                        input_pair_iterator,
                        input_pair_iterator + column_size,
                        fill_pair_iterator,
                        filter,
                        value_type,
                        stream,
                        mr);
  } else {
    auto filter = [segment_offsets, offset] __device__(auto const& i) {
      auto segment_bound_idx =
        thrust::upper_bound(thrust::seq, segment_offsets.begin(), segment_offsets.end(), i);
      return not(*segment_bound_idx + offset <= i and i < *segment_bound_idx);
    };
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

/**
 * @brief Helper function to invoke string specialization of `copy_if_else`
 */
template <typename InputIterator, typename ScalarIterator>
std::unique_ptr<column> segmented_shift_string_impl(InputIterator input_iterator,
                                                    ScalarIterator fill_iterator,
                                                    size_type offset,
                                                    device_span<size_type const> segment_offsets,
                                                    size_type column_size,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  auto filter = [segment_offsets, offset] __device__(auto const& i) {
    auto const segment_bound_idx =
      thrust::upper_bound(thrust::seq, segment_offsets.begin(), segment_offsets.end(), i) -
      (offset > 0);
    auto const left_idx  = *segment_bound_idx + (offset < 0 ? offset : 0);
    auto const right_idx = *segment_bound_idx + (offset > 0 ? offset : 0);
    return not(left_idx <= i and i < right_idx);
  };
  return strings::detail::copy_if_else(
    input_iterator, input_iterator + column_size, fill_iterator, filter, stream, mr);
}

template <typename T, typename Enable = void>
struct segmented_shift_functor {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type for segmented_shift.");
  }
};

/**
 * @brief Segmented shift specialization for representation layout compatible types.
 */
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
    bool nullable           = not fill_value.is_valid() or segmented_values.nullable();

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

/**
 * @brief Segmented shift specialization for `string_view`.
 */
template <>
struct segmented_shift_functor<string_view> {
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto values_device_view = column_device_view::create(segmented_values, stream);
    auto input_iterator     = make_optional_iterator<cudf::string_view>(
      *values_device_view, contains_nulls::DYNAMIC{}, segmented_values.has_nulls());
    auto fill_iterator =
      make_optional_iterator<cudf::string_view>(fill_value, contains_nulls::YES{});
    return segmented_shift_string_impl(input_iterator - offset,
                                       fill_iterator,
                                       offset,
                                       segment_offsets,
                                       segmented_values.size(),
                                       stream,
                                       mr);
  }
};

/**
 * @brief Functor to instantiate the specializations for segmented shift and
 * forward arguments.
 */
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

}  // namespace

std::unique_ptr<column> segmented_shift(column_view const& segmented_values,
                                        device_span<size_type const> segment_offsets,
                                        size_type offset,
                                        scalar const& fill_value,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (segmented_values.is_empty()) { return empty_like(segmented_values); }
  if (offset == 0) { return std::make_unique<column>(segmented_values); };

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
