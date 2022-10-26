/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Returns a new strings column using the specified Filter to select
 * strings from the lhs iterator or the rhs iterator.
 *
 * ```
 * output[i] = filter_fn(i) ? lhs(i) : rhs(i)
 * ```
 *
 * @tparam StringIterLeft A random access iterator whose value_type is
 * `thrust::optional<string_view>` where the `optional` has a value iff the element is valid.
 * @tparam StringIterRight A random access iterator whose value_type is
 * `thrust::optional<string_view>` where the `optional` has a value iff the element is valid.
 * @tparam Filter Functor that takes an index and returns a boolean.
 *
 * @param lhs_begin Start of first set of data. Used when `filter_fn` returns true.
 * @param lhs_end End of first set of data.
 * @param rhs_begin Strings of second set of data. Used when `filter_fn` returns false.
 * @param filter_fn Called to determine which iterator to use for a specific row.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
template <typename StringIterLeft, typename StringIterRight, typename Filter>
std::unique_ptr<cudf::column> copy_if_else(StringIterLeft lhs_begin,
                                           StringIterLeft lhs_end,
                                           StringIterRight rhs_begin,
                                           Filter filter_fn,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto strings_count = std::distance(lhs_begin, lhs_end);
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  // create null mask
  auto valid_mask = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [lhs_begin, rhs_begin, filter_fn] __device__(size_type idx) {
      return filter_fn(idx) ? lhs_begin[idx].has_value() : rhs_begin[idx].has_value();
    },
    stream,
    mr);
  size_type null_count = valid_mask.second;
  auto null_mask       = (null_count > 0) ? std::move(valid_mask.first) : rmm::device_buffer{};

  // build offsets column
  auto offsets_transformer = [lhs_begin, rhs_begin, filter_fn] __device__(size_type idx) {
    auto const result = filter_fn(idx) ? lhs_begin[idx] : rhs_begin[idx];
    return result.has_value() ? result->size_bytes() : 0;
  };

  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), offsets_transformer);
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().template data<int32_t>();

  // build chars column
  auto const bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
  auto chars_column = create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().template data<char>();
  // fill in chars
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [lhs_begin, rhs_begin, filter_fn, d_offsets, d_chars] __device__(size_type idx) {
      auto const result = filter_fn(idx) ? lhs_begin[idx] : rhs_begin[idx];
      if (!result.has_value()) return;
      auto const d_str = *result;
      memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
    });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
