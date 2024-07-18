/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>

#include <memory>
#include <utility>

namespace cudf {
namespace detail {

enum class which : bool {
  PRECEDING,
  FOLLOWING,
};

template <which Which>
struct window_offset_impl {
  template <typename T,
            CUDF_ENABLE_IF(!(cudf::is_timestamp<T>() or
                             (cudf::is_index_type<T>() and !cudf::is_unsigned<T>())))>
  std::unique_ptr<column> operator()(column_view const&,
                                     scalar const& length,
                                     scalar const& offset,
                                     window_type const window_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    CUDF_FAIL("Unsupported rolling window type.");
  }

  template <typename InputType, typename OffsetType, typename StrictWeakOrdering>
  struct distance_kernel {
    OffsetType const* length;
    OffsetType const* offset;
    cudf::column_device_view::const_iterator<InputType> begin;
    cudf::column_device_view::const_iterator<InputType> end;
    __device__ size_type operator()(size_type i)
    {
      if constexpr (Which == which::PRECEDING) {
        return 1 + thrust::distance(
                     thrust::lower_bound(
                       thrust::seq, begin, end, *(begin + i) + *offset, StrictWeakOrdering{}),
                     begin + i);
      } else {
        return thrust::distance(begin + i,
                                thrust::lower_bound(thrust::seq,
                                                    begin,
                                                    end,
                                                    *(begin + i) + (*offset + *length),
                                                    StrictWeakOrdering{})) -
               1;
      }
    }
  };

  template <typename T, CUDF_ENABLE_IF(cudf::is_timestamp<T>())>
  std::unique_ptr<column> operator()(column_view const& input,
                                     scalar const& length,
                                     scalar const& offset,
                                     window_type const window_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    using OffsetType = typename T::duration;
    CUDF_EXPECTS(cudf::is_duration(length.type()), "Length and offset must be duration types.");
    CUDF_EXPECTS(length.type().id() == type_to_id<OffsetType>(),
                 "Length must have same resolution as input.");
    auto result = cudf::make_numeric_column(
      cudf::data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
    auto const d_input_device_view = cudf::column_device_view::create(input, stream);
    auto input_begin               = d_input_device_view->begin<T>();
    auto input_end                 = d_input_device_view->end<T>();
    auto const d_length = dynamic_cast<duration_scalar<OffsetType> const&>(length).data();
    auto const d_offset = dynamic_cast<duration_scalar<OffsetType> const&>(offset).data();
    switch (window_type) {
      case window_type::LEFT_CLOSED:
      case window_type::OPEN:
        thrust::copy_n(rmm::exec_policy(stream),
                       cudf::detail::make_counting_transform_iterator(
                         0,
                         distance_kernel<T, OffsetType, thrust::less<T>>{
                           d_length, d_offset, input_begin, input_end}),
                       input.size(),
                       result->mutable_view().begin<size_type>());
      case window_type::RIGHT_CLOSED:
      case window_type::CLOSED:
        thrust::copy_n(rmm::exec_policy(stream),
                       cudf::detail::make_counting_transform_iterator(
                         0,
                         distance_kernel<T, OffsetType, thrust::less_equal<T>>{
                           d_length, d_offset, input_begin, input_end}),
                       input.size(),
                       result->mutable_view().begin<size_type>());
    }
    return result;
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_index_type<T>() and !cudf::is_unsigned<T>())>
  std::unique_ptr<column> operator()(column_view const& input,
                                     scalar const& length,
                                     scalar const& offset,
                                     window_type const window_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(have_same_types(input, length),
                 "Input column, length, and offset must have same type.");
    auto result = cudf::make_numeric_column(
      cudf::data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
    auto const d_input_device_view = cudf::column_device_view::create(input, stream);
    auto input_begin               = d_input_device_view->begin<T>();
    auto input_end                 = d_input_device_view->end<T>();
    auto const d_length            = dynamic_cast<numeric_scalar<T> const&>(length).data();
    auto const d_offset            = dynamic_cast<numeric_scalar<T> const&>(offset).data();
    switch (window_type) {
      case window_type::LEFT_CLOSED:
      case window_type::OPEN:
        thrust::copy_n(
          rmm::exec_policy(stream),
          cudf::detail::make_counting_transform_iterator(
            0, distance_kernel<T, T, thrust::less<T>>{d_length, d_offset, input_begin, input_end}),
          input.size(),
          result->mutable_view().begin<size_type>());
      case window_type::RIGHT_CLOSED:
      case window_type::CLOSED:
        thrust::copy_n(rmm::exec_policy(stream),
                       cudf::detail::make_counting_transform_iterator(
                         0,
                         distance_kernel<T, T, thrust::less_equal<T>>{
                           d_length, d_offset, input_begin, input_end}),
                       input.size(),
                       result->mutable_view().begin<size_type>());
    }
    return result;
  }
};

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> windows_from_offset(
  column_view const& input,
  scalar const& length,
  scalar const& offset,
  window_type const window_type,
  bool only_preceding,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!input.has_nulls(), "Input column cannot have nulls.");
  CUDF_EXPECTS(have_same_types(length, offset), "Length and offset must have the same type.");
  auto preceding = type_dispatcher(input.type(),
                                   window_offset_impl<which::PRECEDING>{},
                                   input,
                                   length,
                                   offset,
                                   window_type,
                                   stream,
                                   mr);
  if (only_preceding) {
    return {std::move(preceding), nullptr};
  } else {
    auto following = type_dispatcher(input.type(),
                                     window_offset_impl<which::FOLLOWING>{},
                                     input,
                                     length,
                                     offset,
                                     window_type,
                                     stream,
                                     mr);
    return {std::move(preceding), std::move(following)};
  }
}
}  // namespace detail
}  // namespace cudf
