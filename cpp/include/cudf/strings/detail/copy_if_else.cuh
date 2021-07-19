/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Returns a new strings column using the specified Filter to select
 * strings from the lhs iterator or the rhs iterator.
 *
 * ```
 * output[i] = filter_fn(i) ? lhs(i).first : rhs(i).first
 * ```
 *
 * @tparam StringPairIterLeft Pair iterator returning thrust::pair<string_view,bool> where the
 *         bool parameter specifies if the string_view is valid (true) or not (false).
 * @tparam StringPairIterRight Pair iterator returning thrust::pair<string_view,bool> where the
 *         bool parameter specifies if the string_view is valid (true) or not (false).
 * @tparam Filter Functor that takes an index and returns a boolean.
 *
 * @param lhs_begin Start of first set of data. Used when filter_fn returns true.
 * @param lhs_end End of first set of data.
 * @param rhs_begin Strings of second set of data. Used when filter_fn returns false.
 * @param filter_fn Called to determine which iterator (lhs or rhs) to retrieve an entry for a
 * specific row.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New strings column.
 */
template <typename StringPairIterLeft, typename StringPairIterRight, typename Filter>
std::unique_ptr<cudf::column> copy_if_else(
  StringPairIterLeft lhs_begin,
  StringPairIterLeft lhs_end,
  StringPairIterRight rhs_begin,
  Filter filter_fn,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_count = std::distance(lhs_begin, lhs_end);
  if (strings_count == 0) return make_empty_column(data_type{type_id::STRING});

  // create null mask
  auto valid_mask = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [lhs_begin, rhs_begin, filter_fn] __device__(size_type idx) {
      return filter_fn(idx) ? thrust::get<1>(lhs_begin[idx]) : thrust::get<1>(rhs_begin[idx]);
    },
    stream,
    mr);
  size_type null_count = valid_mask.second;
  auto null_mask       = (null_count > 0) ? std::move(valid_mask.first) : rmm::device_buffer{};

  // build offsets column
  auto offsets_transformer = [lhs_begin, rhs_begin, filter_fn] __device__(size_type idx) {
    bool bfilter    = filter_fn(idx);
    size_type bytes = 0;
    if (bfilter ? thrust::get<1>(lhs_begin[idx]) : thrust::get<1>(rhs_begin[idx]))
      bytes = bfilter ? thrust::get<0>(lhs_begin[idx]).size_bytes()
                      : thrust::get<0>(rhs_begin[idx]).size_bytes();
    return bytes;
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
      auto bfilter = filter_fn(idx);
      if (bfilter ? !thrust::get<1>(lhs_begin[idx]) : !thrust::get<1>(rhs_begin[idx])) return;
      string_view d_str = bfilter ? thrust::get<0>(lhs_begin[idx]) : thrust::get<0>(rhs_begin[idx]);
      memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
    });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
