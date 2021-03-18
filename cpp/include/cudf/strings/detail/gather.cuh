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
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Returns a new chars column using the specified indices to select
 * strings from the input iterator.
 *
 * This uses a character-parallel gather CUDA kernel that performs very
 * well on a strings column with long strings (e.g. average > 64 bytes).
 *
 * @tparam StringIterator Iterator should produce `string_view` objects.
 * @tparam MapIterator Iterator for retrieving integer indices of the `StringIterator`.
 *
 * @param strings_begin Start of the iterator to retrieve `string_view` instances
 * @param map_begin Start of index iterator.
 * @param map_end End of index iterator.
 * @param offsets The offset values to be associated with the output chars column.
 * @param chars_bytes The total number of bytes for the output chars column.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New chars column fit for a strings column.
 */
template <typename StringIterator, typename MapIterator>
std::unique_ptr<cudf::column> gather_chars(StringIterator strings_begin,
                                           MapIterator map_begin,
                                           MapIterator map_end,
                                           cudf::device_span<int32_t const> const offsets,
                                           size_type chars_bytes,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto const output_count = std::distance(map_begin, map_end);
  if (output_count == 0) return make_empty_column(data_type{type_id::INT8});

  auto chars_column  = create_chars_child_column(output_count, 0, chars_bytes, stream, mr);
  auto const d_chars = chars_column->mutable_view().template data<char>();

  auto gather_chars_fn = [strings_begin, map_begin, offsets] __device__(size_type out_idx) -> char {
    auto const out_row =
      thrust::prev(thrust::upper_bound(thrust::seq, offsets.begin(), offsets.end(), out_idx));
    auto const row_idx = map_begin[thrust::distance(offsets.begin(), out_row)];  // get row index
    auto const d_str   = strings_begin[row_idx];                                 // get row's string
    auto const offset  = out_idx - *out_row;  // get string's char
    return d_str.data()[offset];
  };

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(chars_bytes),
                    d_chars,
                    gather_chars_fn);

  return chars_column;
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
  auto const output_count  = std::distance(begin, end);
  auto const strings_count = strings.size();
  if (output_count == 0) return make_empty_strings_column(stream, mr);

  // allocate offsets column and use memory to compute string size in each output row
  auto out_offsets_column = make_numeric_column(
    data_type{type_id::INT32}, output_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_out_offsets = out_offsets_column->mutable_view().template data<int32_t>();
  auto const d_in_offsets =
    (strings_count > 0) ? strings.offsets().data<int32_t>() + strings.offset() : nullptr;
  thrust::transform(rmm::exec_policy(stream),
                    begin,
                    end,
                    d_out_offsets,
                    [d_in_offsets, strings_count] __device__(size_type in_idx) {
                      if (NullifyOutOfBounds && (in_idx < 0 || in_idx >= strings_count)) return 0;
                      return d_in_offsets[in_idx + 1] - d_in_offsets[in_idx];
                    });

  // check total size is not too large
  size_t const total_bytes = thrust::transform_reduce(
    rmm::exec_policy(stream),
    d_out_offsets,
    d_out_offsets + output_count,
    [] __device__(auto size) { return static_cast<size_t>(size); },
    size_t{0},
    thrust::plus<size_t>{});
  CUDF_EXPECTS(total_bytes < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "total size of output strings is too large for a cudf column");

  // In-place convert output sizes into offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + output_count + 1, d_out_offsets);

  // build chars column
  cudf::device_span<int32_t const> const d_out_offsets_span(d_out_offsets, output_count + 1);
  auto const d_strings  = column_device_view::create(strings.parent(), stream);
  auto out_chars_column = gather_chars(d_strings->begin<string_view>(),
                                       begin,
                                       end,
                                       d_out_offsets_span,
                                       static_cast<size_type>(total_bytes),
                                       stream,
                                       mr);

  return make_strings_column(output_count,
                             std::move(out_offsets_column),
                             std::move(out_chars_column),
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
