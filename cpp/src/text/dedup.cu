/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/dedup.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace nvtext {
namespace detail {
namespace {

using string_index = cudf::strings::detail::string_index_pair;

/**
 * @brief Comparator used for building the suffix array
 *
 * Each string is created using the substring of the chars_span
 * between the given index/offset (lhs or rhs) and the end of the
 * chars_span.
 */
struct sort_comparator_fn {
  cudf::device_span<char const> chars_span;
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const
  {
    constexpr auto max_size = cuda::std::numeric_limits<cudf::size_type>::max();
    auto const chars_size   = static_cast<cudf::size_type>(chars_span.size());

    auto const lhs_size = cuda::std::min(max_size, chars_size - lhs);
    auto const rhs_size = cuda::std::min(max_size, chars_size - rhs);
    auto const lh_str   = cudf::string_view(chars_span.data() + lhs, lhs_size);
    auto const rh_str   = cudf::string_view(chars_span.data() + rhs, rhs_size);
    return lh_str < rh_str;
  }
};

/**
 * @brief Utility counts the number of common bytes between the 2 given strings
 */
__device__ cudf::size_type count_common_bytes(cudf::string_view lhs, cudf::string_view rhs)
{
  auto const size1 = lhs.size_bytes();
  auto const size2 = rhs.size_bytes();
  auto const* ptr1 = lhs.data();
  auto const* ptr2 = rhs.data();

  cudf::size_type idx = 0;
  while ((idx < size1) && (idx < size2) && (*ptr1++ == *ptr2++)) {
    ++idx;
  }
  return idx;
}

/**
 * @brief Uses the sorted array of indices (suffix array) to compare common
 * bytes between adjacent strings and return the count if greater than width.
 */
struct find_adjacent_duplicates_fn {
  cudf::device_span<char const> chars_span;
  cudf::size_type width;
  cudf::size_type const* d_indices;
  __device__ int16_t operator()(cudf::size_type idx) const
  {
    if (idx == 0) { return 0; }
    constexpr auto max_size = cuda::std::numeric_limits<cudf::size_type>::max();
    auto const chars_size   = static_cast<cudf::size_type>(chars_span.size());

    auto const lhs      = d_indices[idx - 1];
    auto const rhs      = d_indices[idx];
    auto const lhs_size = cuda::std::min(max_size, chars_size - lhs);
    auto const rhs_size = cuda::std::min(max_size, chars_size - rhs);

    auto const lh_str = cudf::string_view(chars_span.data() + lhs, lhs_size);
    auto const rh_str = cudf::string_view(chars_span.data() + rhs, rhs_size);

    constexpr auto max_run_length =
      static_cast<cudf::size_type>(cuda::std::numeric_limits<int16_t>::max());

    auto const size = cuda::std::min(count_common_bytes(lh_str, rh_str), max_run_length);

    return size >= width ? static_cast<int16_t>(size) : 0;
  }
};

/**
 * @brief Resolves any overlapping duplicate strings found in find_adjacent_duplicates_fn
 *
 * Adjacent strings with common bytes greater than width may produce a sliding window of
 * duplicates which can be collapsed into a single duplicate pattern instead.
 * Care is also taken to preserve adjacent strings which may be larger than max(int16).
 */
struct collapse_overlaps_fn {
  char const* d_chars;
  cudf::size_type const* d_offsets;
  int16_t const* d_sizes;
  __device__ string_index operator()(cudf::size_type idx) const
  {
    constexpr auto max_run_length =
      static_cast<cudf::size_type>(cuda::std::numeric_limits<int16_t>::max());

    auto size   = d_sizes[idx];
    auto offset = d_offsets[idx];
    if ((idx > 0) && ((offset - 1) == d_offsets[idx - 1])) {
      if (size < d_sizes[idx - 1]) { return string_index{nullptr, 0}; }
      if (size == d_sizes[idx - 1] && size == max_run_length) {
        // check if we are in the middle of a chain
        auto prev_idx    = idx - max_run_length;
        auto prev_offset = offset;
        while (prev_idx >= 0) {
          if (d_offsets[prev_idx] != (prev_offset - max_run_length)) {
            prev_idx = -1;
          } else {
            if (d_sizes[idx + 1] < size) { break; }  // final edge
            prev_offset = d_offsets[prev_idx];
            if ((prev_idx == 0) || ((prev_offset - 1) != d_offsets[prev_idx - 1])) { break; }
            prev_idx -= max_run_length;
          }
        }
        if (prev_idx < 0) { return string_index{nullptr, 0}; }
      }
    }

    auto d_ptr = d_chars + offset;
    return string_index(d_ptr, size);
  }
};

std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_suffix_array_fn(
  cudf::device_span<char const> chars_span,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const size = static_cast<cudf::size_type>(chars_span.size()) - min_width + (min_width > 0);
  auto indices    = rmm::device_uvector<cudf::size_type>(size, stream);

  auto const cmp_op = sort_comparator_fn{chars_span};
  auto const seq    = thrust::make_counting_iterator<cudf::size_type>(0);
  auto tmp_bytes    = std::size_t{0};
  cub::DeviceMergeSort::SortKeysCopy(
    nullptr, tmp_bytes, seq, indices.begin(), indices.size(), cmp_op, stream.value());
  auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
  cub::DeviceMergeSort::SortKeysCopy(
    tmp_stg.data(), tmp_bytes, seq, indices.begin(), indices.size(), cmp_op, stream.value());

  return std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(indices));
}

std::unique_ptr<cudf::column> resolve_duplicates_fn(
  cudf::device_span<char const> chars_span,
  cudf::device_span<cudf::size_type const> indices,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto sizes = rmm::device_uvector<int16_t>(indices.size(), stream);

  // locate candidate duplicates within the suffix array
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(indices.size()),
                    sizes.begin(),
                    find_adjacent_duplicates_fn{chars_span, min_width, indices.data()});

  auto const dup_count =
    sizes.size() - thrust::count(rmm::exec_policy(stream), sizes.begin(), sizes.end(), 0);
  auto dup_indices = rmm::device_uvector<cudf::size_type>(dup_count, stream);

  // remove the non-candidate entries from indices and sizes
  thrust::remove_copy_if(
    rmm::exec_policy(stream),
    indices.begin(),
    indices.end(),
    thrust::counting_iterator<cudf::size_type>(0),
    dup_indices.begin(),
    [d_sizes = sizes.data()] __device__(cudf::size_type idx) -> bool { return d_sizes[idx] == 0; });
  auto end = thrust::remove(rmm::exec_policy(stream), sizes.begin(), sizes.end(), 0);
  sizes.resize(cuda::std::distance(sizes.begin(), end), stream);

  // sort the resulting indices/sizes for overlap filtering
  thrust::sort_by_key(
    rmm::exec_policy_nosync(stream), dup_indices.begin(), dup_indices.end(), sizes.begin());

  // produce final duplicates for make_strings_column and collapse any overlapping candidates
  auto duplicates =
    rmm::device_uvector<cudf::strings::detail::string_index_pair>(dup_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(dup_indices.size()),
                    duplicates.begin(),
                    collapse_overlaps_fn{chars_span.data(), dup_indices.data(), sizes.data()});

  // filter out the remaining non-viable candidates
  duplicates.resize(
    cuda::std::distance(
      duplicates.begin(),
      thrust::remove(
        rmm::exec_policy(stream), duplicates.begin(), duplicates.end(), string_index{nullptr, 0})),
    stream);

  // sort the result by size descending (should be very fast)
  thrust::sort(rmm::exec_policy_nosync(stream),
               duplicates.begin(),
               duplicates.end(),
               [] __device__(auto lhs, auto rhs) -> bool { return lhs.second > rhs.second; });

  // ironically remove duplicates from the sorted list
  duplicates.resize(
    cuda::std::distance(duplicates.begin(),
                        thrust::unique(rmm::exec_policy(stream),
                                       duplicates.begin(),
                                       duplicates.end(),
                                       [] __device__(auto lhs, auto rhs) -> bool {
                                         return cudf::string_view(lhs.first, lhs.second) ==
                                                cudf::string_view(rhs.first, rhs.second);
                                       })),
    stream);

  return cudf::strings::detail::make_strings_column(
    duplicates.begin(), duplicates.end(), stream, mr);
}

}  // namespace

std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_suffix_array(
  cudf::strings_column_view const& input,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);

  auto const d_input_chars = input.chars_begin(stream) + first_offset;
  auto const chars_size    = last_offset - first_offset;
  CUDF_EXPECTS(min_width < chars_size, "min_width value cannot exceed the input size");
  CUDF_EXPECTS(chars_size < std::numeric_limits<cudf::size_type>::max(),
               "input size cannot exceed the 2GB");

  auto const chars_span = cudf::device_span<char const>(d_input_chars, chars_size);
  return build_suffix_array_fn(chars_span, min_width, stream, mr);
}

std::unique_ptr<cudf::column> substring_duplicates(cudf::strings_column_view const& input,
                                                   cudf::size_type min_width,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(min_width > 8, "min_width should be at least 8");

  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const d_input_chars = input.chars_begin(stream) + first_offset;
  auto const chars_size    = last_offset - first_offset;
  CUDF_EXPECTS(min_width < chars_size, "min_width value cannot exceed the input size");

  auto const chars_span = cudf::device_span<char const>(d_input_chars, chars_size);

  auto indices = build_suffix_array_fn(chars_span, min_width, stream, mr);

  return resolve_duplicates_fn(chars_span, *indices, min_width, stream, mr);
}
}  // namespace detail

std::unique_ptr<cudf::column> substring_duplicates(cudf::strings_column_view const& input,
                                                   cudf::size_type min_width,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::substring_duplicates(input, min_width, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_suffix_array(
  cudf::strings_column_view const& input,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::build_suffix_array(input, min_width, stream, mr);
}

}  // namespace nvtext
