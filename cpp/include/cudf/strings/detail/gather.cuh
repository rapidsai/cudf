/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {

template <typename Iterator>
constexpr inline bool is_signed_iterator()
{
  return std::is_signed<typename std::iterator_traits<Iterator>::value_type>::value;
}

namespace strings {
namespace detail {
/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather<true>( s1, map.begin(), map.end() )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam NullifyOutOfBounds If true, indices outside the column's range are nullified.
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New strings column containing the gathered strings.
 */
template <bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<cudf::column> gather(
  strings_column_view const& strings,
  MapIterator begin,
  MapIterator end,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto output_count  = std::distance(begin, end);
  auto strings_count = strings.size();
  if (output_count == 0) return make_empty_strings_column(stream, mr);

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // build offsets column
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, output_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto d_offsets = offsets_column->mutable_view().template data<int32_t>();
  thrust::transform(rmm::exec_policy(stream),
                    begin,
                    end,
                    d_offsets,
                    [d_strings, strings_count] __device__(size_type idx) {
                      if (NullifyOutOfBounds && ((idx < 0) || (idx >= strings_count))) return 0;
                      if (d_strings.is_null(idx)) return 0;
                      return d_strings.element<string_view>(idx).size_bytes();
                    });

  // check total size is not too large
  size_t total_bytes = thrust::transform_reduce(
    rmm::exec_policy(stream),
    d_offsets,
    d_offsets + output_count,
    [] __device__(auto size) { return static_cast<size_t>(size); },
    size_t{0},
    thrust::plus<size_t>{});
  CUDF_EXPECTS(total_bytes < std::numeric_limits<size_type>::max(),
               "total size of output strings is too large for a cudf column");

  // create offsets from sizes
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + output_count + 1, d_offsets);

  // build chars column
  size_type bytes   = static_cast<size_type>(total_bytes);
  auto chars_column = create_chars_child_column(output_count, 0, bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().template data<char>();
  // fill in chars
  auto gather_chars =
    [d_strings, begin, strings_count, d_offsets, d_chars] __device__(size_type idx) {
      auto index = begin[idx];
      if (NullifyOutOfBounds) {
        if (is_signed_iterator<MapIterator>() ? ((index < 0) || (index >= strings_count))
                                              : (index >= strings_count))
          return;
      }
      if (d_strings.is_null(index)) return;
      string_view d_str = d_strings.element<string_view>(index);
      memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
    };
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     output_count,
                     gather_chars);

  return make_strings_column(output_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             0,
                             rmm::device_buffer{0, stream, mr},
                             stream,
                             mr);
}

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map.begin(), map.end(), true )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param nullify_out_of_bounds If true, indices outside the column's range are nullified.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New strings column containing the gathered strings.
 */
template <typename MapIterator>
std::unique_ptr<cudf::column> gather(
  strings_column_view const& strings,
  MapIterator begin,
  MapIterator end,
  bool nullify_out_of_bounds,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (nullify_out_of_bounds) return gather<true>(strings, begin, end, stream, mr);
  return gather<false>(strings, begin, end, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
