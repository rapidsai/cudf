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
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/dedup.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cmath>

namespace nvtext {
namespace detail {
namespace {

using string_index = cudf::strings::detail::string_index_pair;

struct bitonic_sort_comparator_fn {
  cudf::device_span<char const> chars_span;
  int64_t size;
  __device__ bool operator()(int64_t lhs, int64_t rhs) const
  {
    if (lhs >= size || rhs >= size) { return lhs < rhs; }
    constexpr int64_t max_size = cuda::std::numeric_limits<cudf::size_type>::max();
    auto const chars_size      = static_cast<int64_t>(chars_span.size());

    auto const lhs_size = static_cast<cudf::size_type>(cuda::std::min(max_size, chars_size - lhs));
    auto const rhs_size = static_cast<cudf::size_type>(cuda::std::min(max_size, chars_size - rhs));
    auto const lh_str   = cudf::string_view(chars_span.data() + lhs, lhs_size);
    auto const rh_str   = cudf::string_view(chars_span.data() + rhs, rhs_size);
    return lh_str < rh_str;
  }
};

__global__ void bitonic_sort_step(bitonic_sort_comparator_fn scfn,
                                  cudf::size_type* d_indices,
                                  int64_t size2,  // size2 is the power of 2 greater than size
                                  cudf::size_type right,
                                  cudf::size_type left)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= size2) { return; }
  auto const dit = tid ^ right;
  if (dit <= tid) { return; }

  auto const asc = tid & left;
  auto const idx = d_indices[tid];
  auto const jdx = d_indices[dit];
  if (((asc == 0) && (scfn(jdx, idx))) || ((asc != 0) && (scfn(idx, jdx)))) {
    auto const temp = idx;  // swap entries
    d_indices[tid]  = jdx;
    d_indices[dit]  = temp;
  }
}

__device__ cudf::size_type count_common_bytes(cudf::string_view lhs, cudf::string_view rhs)
{
  auto const size1 = lhs.size_bytes();
  auto const size2 = rhs.size_bytes();
  auto const* ptr1 = lhs.data();
  auto const* ptr2 = rhs.data();

  cudf::size_type idx = 0;
  for (; (idx < size1) && (idx < size2); ++idx) {
    if (*ptr1 != *ptr2) { break; }
    ++ptr1;
    ++ptr2;
  }
  return idx;
}

struct find_duplicates_fn {
  char const* d_chars;
  int64_t chars_size;
  cudf::size_type width;
  cudf::size_type const* d_indices;
  __device__ int16_t operator()(int64_t idx) const
  {
    if (idx == 0) { return 0; }
    constexpr int64_t max_size = cuda::std::numeric_limits<cudf::size_type>::max();

    auto const lhs      = d_indices[idx - 1];
    auto const rhs      = d_indices[idx];
    auto const lhs_size = static_cast<cudf::size_type>(cuda::std::min(max_size, chars_size - lhs));
    auto const rhs_size = static_cast<cudf::size_type>(cuda::std::min(max_size, chars_size - rhs));

    auto const lh_str = cudf::string_view(d_chars + lhs, lhs_size);
    auto const rh_str = cudf::string_view(d_chars + rhs, rhs_size);

    constexpr auto max_run_length =
      static_cast<cudf::size_type>(cuda::std::numeric_limits<int16_t>::max());

    auto const size = cuda::std::min(count_common_bytes(lh_str, rh_str), max_run_length);

    return size >= width ? static_cast<int16_t>(size) : 0;
  }
};

struct collapse_overlaps_fn {
  char const* d_chars;
  cudf::size_type const* d_offsets;
  int16_t const* d_sizes;
  __device__ string_index operator()(int64_t idx) const
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
}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<int16_t>>>
build_suffix_array_fn(cudf::device_span<char const> chars_span,
                      cudf::size_type min_width,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  auto const size  = static_cast<int64_t>(chars_span.size()) - min_width + 1;
  auto const size2 = 1L << static_cast<int32_t>(std::ceil(std::log2(size)));

  auto indices = rmm::device_uvector<cudf::size_type>(size2, stream);
  auto sizes   = rmm::device_uvector<int16_t>(indices.size(), stream);

  thrust::sequence(rmm::exec_policy_nosync(stream), indices.begin(), indices.end());
  auto const cmp_op = bitonic_sort_comparator_fn{chars_span, size};
  auto const grid   = cudf::detail::grid_1d(size2, 512);
  for (auto left = 2L; left <= size2; left <<= 1) {
    for (auto right = left >> 1; right > 0; right >>= 1) {
      bitonic_sort_step<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        cmp_op, indices.data(), size2, right, left);
    }
  }
  indices.resize(size, stream);
  sizes.resize(size, stream);

  return std::make_pair(std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(indices)),
                        std::make_unique<rmm::device_uvector<int16_t>>(std::move(sizes)));
}

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
  return std::get<0>(build_suffix_array_fn(chars_span, min_width, stream, mr));
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

  auto d_input_chars = input.chars_begin(stream) + first_offset;
  auto chars_size    = last_offset - first_offset;
  CUDF_EXPECTS(min_width < chars_size, "min_width value cannot exceed the input size");

  auto const chars_span = cudf::device_span<char const>(d_input_chars, chars_size);
  auto [indices, sizes] = build_suffix_array_fn(chars_span, min_width, stream, mr);

  // locate candidate duplicates within the suffixes produced by sort
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(indices->size()),
                    sizes->begin(),
                    find_duplicates_fn{d_input_chars, chars_size, min_width, indices->data()});

  // remove the non-candidate entries from indices and sizes
  thrust::remove_if(
    rmm::exec_policy(stream),
    indices->begin(),
    indices->end(),
    thrust::counting_iterator<int64_t>(0),
    [d_sizes = sizes->data()] __device__(int64_t idx) -> bool { return d_sizes[idx] == 0; });
  auto end = thrust::remove(rmm::exec_policy(stream), sizes->begin(), sizes->end(), 0);
  sizes->resize(thrust::distance(sizes->begin(), end), stream);
  indices->resize(sizes->size(), stream);

  // sort the resulting indices/sizes for overlap filtering
  thrust::sort_by_key(
    rmm::exec_policy_nosync(stream), indices->begin(), indices->end(), sizes->begin());

  // produce final duplicates for make_strings_column and collapse any overlapping candidates
  auto duplicates =
    rmm::device_uvector<cudf::strings::detail::string_index_pair>(indices->size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(indices->size()),
                    duplicates.begin(),
                    collapse_overlaps_fn{d_input_chars, indices->data(), sizes->data()});

  // filter out the remaining non-viable candidates
  duplicates.resize(
    thrust::distance(
      duplicates.begin(),
      thrust::remove(
        rmm::exec_policy(stream), duplicates.begin(), duplicates.end(), string_index{nullptr, 0})),
    stream);

  // sort result by size descending (should be very fast)
  thrust::sort(rmm::exec_policy_nosync(stream),
               duplicates.begin(),
               duplicates.end(),
               [] __device__(auto lhs, auto rhs) -> bool { return lhs.second > rhs.second; });

  return cudf::strings::detail::make_strings_column(
    duplicates.begin(), duplicates.end(), stream, mr);
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
