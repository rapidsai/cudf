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
#include <cudf/detail/nvtx/ranges.hpp>
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

/*
 * Select the appropriate ordering comparator for the window type and if we're computing the
 * preceding or following window.
 */
template <which Which, window_type WindowType>
struct op_impl {};

template <which Which>
struct op_impl<Which, window_type::LEFT_CLOSED> {
  using type = std::less<>;
};
template <which Which>
struct op_impl<Which, window_type::RIGHT_CLOSED> {
  using type = std::less_equal<>;
};
template <>
struct op_impl<which::PRECEDING, window_type::OPEN> {
  using type = std::less_equal<>;
};
template <>
struct op_impl<which::FOLLOWING, window_type::OPEN> {
  using type = std::less<>;
};
template <>
struct op_impl<which::PRECEDING, window_type::CLOSED> {
  using type = std::less<>;
};
template <>
struct op_impl<which::FOLLOWING, window_type::CLOSED> {
  using type = std::less_equal<>;
};
template <which Which, window_type WindowType>
using op_t = typename op_impl<Which, WindowType>::type;

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

  template <typename T, typename OffsetType, typename ScalarType>
  [[nodiscard]] std::unique_ptr<column> compute_window_bounds(
    column_view const& input,
    scalar const& length,
    scalar const& offset,
    window_type const window_type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    auto result = cudf::make_numeric_column(
      cudf::data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
    auto const input_device_view = cudf::column_device_view::create(input, stream);
    auto input_begin             = input_device_view->begin<T>();
    auto input_end               = input_device_view->end<T>();
    auto const d_length          = dynamic_cast<ScalarType const&>(length).data();
    auto const d_offset          = dynamic_cast<ScalarType const&>(offset).data();
    auto copy_n                  = [&](auto kernel) {
      thrust::copy_n(rmm::exec_policy(stream),
                     cudf::detail::make_counting_transform_iterator(0, kernel),
                     input.size(),
                     result->mutable_view().begin<size_type>());
    };
    if (window_type == window_type::LEFT_CLOSED) {
      copy_n(distance_kernel<T, OffsetType, op_t<Which, window_type::LEFT_CLOSED>>{
        d_length, d_offset, input_begin, input_end});
    } else if (window_type == window_type::OPEN) {
      copy_n(distance_kernel<T, OffsetType, op_t<Which, window_type::OPEN>>{
        d_length, d_offset, input_begin, input_end});
    } else if (window_type == window_type::RIGHT_CLOSED) {
      copy_n(distance_kernel<T, OffsetType, op_t<Which, window_type::RIGHT_CLOSED>>{
        d_length, d_offset, input_begin, input_end});
    } else if (window_type == window_type::CLOSED) {
      copy_n(distance_kernel<T, OffsetType, op_t<Which, window_type::CLOSED>>{
        d_length, d_offset, input_begin, input_end});
    } else {
      CUDF_FAIL("Unhandled window type.");
    }
    return result;
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_timestamp<T>())>
  [[nodiscard]] std::unique_ptr<column> operator()(column_view const& input,
                                                   scalar const& length,
                                                   scalar const& offset,
                                                   window_type const window_type,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
  {
    using OffsetType = typename T::duration;
    using ScalarType = duration_scalar<OffsetType>;
    CUDF_EXPECTS(cudf::is_duration(length.type()), "Length and offset must be duration types.");
    CUDF_EXPECTS(length.type().id() == type_to_id<OffsetType>(),
                 "Length must have same the resolution as the input.");
    return compute_window_bounds<T, OffsetType, ScalarType>(
      input, length, offset, window_type, stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_index_type<T>() and !cudf::is_unsigned<T>())>
  [[nodiscard]] std::unique_ptr<column> operator()(column_view const& input,
                                                   scalar const& length,
                                                   scalar const& offset,
                                                   window_type const window_type,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
  {
    using OffsetType = T;
    using ScalarType = numeric_scalar<OffsetType>;
    CUDF_EXPECTS(have_same_types(input, length),
                 "Input column, length, and offset must have the same type.");
    return compute_window_bounds<T, OffsetType, ScalarType>(
      input, length, offset, window_type, stream, mr);
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

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> windows_from_offset(
  column_view const& input,
  scalar const& length,
  scalar const& offset,
  window_type const window_type,
  bool only_preceding,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::windows_from_offset(
    input, length, offset, window_type, only_preceding, stream, mr);
}

}  // namespace cudf
