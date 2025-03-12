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

namespace nvtext {
namespace detail {
namespace {

using string_index = cudf::strings::detail::string_index_pair;

struct sort_comparator_fn {
  char const* d_chars;
  int64_t chars_size;
  __device__ bool operator()(int64_t lhs, int64_t rhs) const
  {
    constexpr int64_t max_size = cuda::std::numeric_limits<cudf::size_type>::max();

    auto const lhs_size = static_cast<cudf::size_type>(cuda::std::min(max_size, chars_size - lhs));
    auto const rhs_size = static_cast<cudf::size_type>(cuda::std::min(max_size, chars_size - rhs));
    auto const lh_str   = cudf::string_view(d_chars + lhs, lhs_size);
    auto const rh_str   = cudf::string_view(d_chars + rhs, rhs_size);
    return lh_str < rh_str;
  }
};

#if 0
__global__ void bitonic_sort_step(
  sort_comparator_fn scfn, int64_t* d_indices, int64_t size, int64_t j, int64_t k)
{
  auto const i   = cudf::detail::grid_1d::global_thread_id();
  auto const ixj = i ^ j;

  if (i >= size || ixj >= size) { return; }

  if ((ixj) > i) {
    if ((i & k) == 0) {
      if (scfn(d_indices[ixj], d_indices[i])) {  //(dev_values[i] > dev_values[ixj])
        auto const temp = d_indices[i];          // dev_values[i];
        d_indices[i]    = d_indices[ixj];        // dev_values[i]   = dev_values[ixj];
        d_indices[ixj]  = temp;                  // dev_values[ixj] = temp;
      }
    } else {
      if (scfn(d_indices[i], d_indices[ixj])) {  //(dev_values[i] < dev_values[ixj])
        auto const temp = d_indices[i];          // dev_values[i];
        d_indices[i]    = d_indices[ixj];        // dev_values[i]   = dev_values[ixj];
        d_indices[ixj]  = temp;                  // dev_values[ixj] = temp;
      }
    }
  }
}
#endif

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
  int64_t const* d_indices;
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
  int64_t const* d_offsets;
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

template <typename Iterator, typename Stencil, typename Predicate>
Iterator remove_if_safe(
  Iterator first, Iterator last, Stencil stencil, Predicate const& fn, rmm::cuda_stream_view stream)
{
  auto const size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                             static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto result = first;
  auto itr    = first;
  while (itr != last) {
    auto end = static_cast<std::size_t>(std::distance(itr, last)) <= size ? last : itr + size;
    result   = thrust::remove_if(rmm::exec_policy(stream), itr, end, stencil, fn);
    itr      = end;
  }
  return result;
}

// handles ranges above int32 max
template <typename Iterator, typename T>
Iterator remove_safe(Iterator first, Iterator last, T const& value, rmm::cuda_stream_view stream)
{
  auto const size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                             static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto result = first;
  auto itr    = first;
  while (itr != last) {
    auto end = static_cast<std::size_t>(std::distance(itr, last)) <= size ? last : itr + size;
    result   = thrust::remove(rmm::exec_policy(stream), itr, end, value);
    itr      = end;
  }
  return result;
}
}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<int64_t>>,
          std::unique_ptr<rmm::device_uvector<int16_t>>>
build_suffix_array(cudf::strings_column_view const& input,
                   cudf::size_type min_width,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
{
  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);

  auto d_input_chars = input.chars_begin(stream) + first_offset;
  auto chars_size    = last_offset - first_offset;
  CUDF_EXPECTS(min_width < chars_size, "min_width value cannot exceed the input size");

  auto indices = rmm::device_uvector<int64_t>(chars_size - min_width + 1, stream);
  auto sizes   = rmm::device_uvector<int16_t>(indices.size(), stream);

  {
    auto const cmp_op = sort_comparator_fn{d_input_chars, chars_size};
    auto const seq    = thrust::make_counting_iterator<int64_t>(0);
    auto tmp_bytes    = std::size_t{0};
    cub::DeviceMergeSort::SortKeysCopy(
      nullptr, tmp_bytes, seq, indices.begin(), indices.size(), cmp_op, stream.value());
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    // std::cout << indices.size() * sizeof(int64_t) << "/" << tmp_bytes << std::endl;
    cub::DeviceMergeSort::SortKeysCopy(
      tmp_stg.data(), tmp_bytes, seq, indices.begin(), indices.size(), cmp_op, stream.value());
  }
#if 0
  {
    thrust::sequence(rmm::exec_policy_nosync(stream), indices.begin(), indices.end());
    auto const cmp_op = sort_comparator_fn{d_input_chars, chars_size};
    auto size2        = 1 << static_cast<int>(std::log2(chars_size) + 1.0);
    for (auto k = 2L; k <= size2; k <<= 1) {
      for (auto j = k >> 1; j > 0; j = j >> 1) {
        auto grid = cudf::detail::grid_1d(chars_size, 512);
        bitonic_sort_step<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
          cmp_op, indices.data(), (int64_t)indices.size(), j, k);
      }
    }
    std::cout << "bitonic-sort " << (int)cudaStreamSynchronize(stream.value()) << std::endl;
  }
#endif

  return std::make_pair(std::make_unique<rmm::device_uvector<int64_t>>(std::move(indices)),
                        std::make_unique<rmm::device_uvector<int16_t>>(std::move(sizes)));
}

std::unique_ptr<cudf::column> substring_deduplicate(cudf::strings_column_view const& input,
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

  auto [indices, sizes] = build_suffix_array(input, min_width, stream, mr);

  // locate candidate duplicates within the suffixes produced by sort
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(indices->size()),
                    sizes->begin(),
                    find_duplicates_fn{d_input_chars, chars_size, min_width, indices->data()});

  // remove the non-candidate entries from indices and sizes
  remove_if_safe(
    indices->begin(),
    indices->end(),
    thrust::counting_iterator<int64_t>(0),
    [d_sizes = sizes->data()] __device__(int64_t idx) -> bool { return d_sizes[idx] == 0; },
    stream);
  auto end = remove_safe(sizes->begin(), sizes->end(), 0, stream);
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

std::unique_ptr<cudf::column> substring_deduplicate(cudf::strings_column_view const& input,
                                                    cudf::size_type min_width,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::substring_deduplicate(input, min_width, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<int64_t>>,
          std::unique_ptr<rmm::device_uvector<int16_t>>>
build_suffix_array(cudf::strings_column_view const& input,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::build_suffix_array(input, 8, stream, mr);
}
}  // namespace nvtext
