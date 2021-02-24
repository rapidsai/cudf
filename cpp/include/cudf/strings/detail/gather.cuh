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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

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
  auto const output_count  = std::distance(begin, end);
  auto const strings_count = strings.size();
  if (output_count == 0) return make_empty_strings_column(stream, mr);

  // allocate offsets column
  auto out_offsets_column = make_numeric_column(
    data_type{type_id::INT32}, output_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_out_offsets = out_offsets_column->mutable_view().template data<int32_t>();

  if (strings_count == 0) {
    // Gathering from an empty strings column, so generate null offsets with no chars.
    CUDF_EXPECTS(NullifyOutOfBounds, "gathering from an empty strings column");
    CUDA_TRY(cudaMemsetAsync(
      d_out_offsets, 0, out_offsets_column->size() * sizeof(int32_t), stream.value()));
    return make_strings_column(output_count,
                               std::move(out_offsets_column),
                               create_chars_child_column(0, 0, 0, stream, mr),
                               0,
                               rmm::device_buffer{0, stream, mr},
                               stream,
                               mr);
  }

  // use output offsets memory to compute string size in each output row
  auto const d_in_offsets = strings.offsets().data<int32_t>() + strings.offset();
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
  size_type const out_chars_bytes = static_cast<size_type>(total_bytes);
  auto out_chars_column  = create_chars_child_column(output_count, 0, out_chars_bytes, stream, mr);
  auto const d_out_chars = out_chars_column->mutable_view().template data<char>();

  // fill in chars
  cudf::detail::device_span<int32_t const> const d_out_offsets_span(d_out_offsets,
                                                                    output_count + 1);
  auto const d_in_chars = strings.chars().data<char>();
  auto gather_chars_fn =
    [d_out_offsets_span, begin, d_in_offsets, d_in_chars] __device__(size_type out_char_idx) {
      // find output row index for this output char index
      auto const next_row_ptr = thrust::upper_bound(
        thrust::seq, d_out_offsets_span.begin(), d_out_offsets_span.end(), out_char_idx);
      auto const out_row_idx     = thrust::distance(d_out_offsets_span.begin(), next_row_ptr) - 1;
      auto const str_char_offset = out_char_idx - d_out_offsets_span[out_row_idx];
      auto const in_row_idx      = begin[out_row_idx];
      auto const in_char_offset  = d_in_offsets[in_row_idx] + str_char_offset;
      return d_in_chars[in_char_offset];
    };
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(out_chars_bytes),
                    d_out_chars,
                    gather_chars_fn);

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
