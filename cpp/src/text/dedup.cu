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
    auto size   = d_sizes[idx];
    auto offset = d_offsets[idx];
    if ((idx > 0) && ((offset - 1) == d_offsets[idx - 1]) && (size < d_sizes[idx - 1])) {
      // TODO: need to handle chains longer than max<int16_t>
      return string_index{nullptr, 0};
    }
    auto d_ptr = d_chars + offset;
    return string_index(d_ptr, size);
  }
};

}  // namespace

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

  auto indices = rmm::device_uvector<int64_t>(chars_size - min_width + 1, stream);
  auto sizes   = rmm::device_uvector<int16_t>(indices.size(), stream);

  thrust::sequence(rmm::exec_policy_nosync(stream), indices.begin(), indices.end());
  // note: thrust::sort may be limited to a 32-bit range
  thrust::sort(rmm::exec_policy_nosync(stream),
               indices.begin(),
               indices.end(),
               sort_comparator_fn{d_input_chars, chars_size});

  // locate candidate duplicates within the suffixes produced by sort
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(indices.size()),
                    sizes.begin(),
                    find_duplicates_fn{d_input_chars, chars_size, min_width, indices.data()});

  // remove the non-candidate entries from indices and sizes
  thrust::remove_if(
    rmm::exec_policy_nosync(stream),
    indices.begin(),
    indices.end(),
    thrust::counting_iterator<int64_t>(0),
    [d_sizes = sizes.data()] __device__(int64_t idx) -> bool { return d_sizes[idx] == 0; });
  auto end = thrust::remove(rmm::exec_policy(stream), sizes.begin(), sizes.end(), 0);
  sizes.resize(thrust::distance(sizes.begin(), end), stream);
  indices.resize(sizes.size(), stream);

  // sort the resulting indices/sizes for overlap filtering
  thrust::sort_by_key(
    rmm::exec_policy_nosync(stream), indices.begin(), indices.end(), sizes.begin());

  // produce final duplicates for make_strings_column and collapse any overlapping candidates
  auto duplicates =
    rmm::device_uvector<cudf::strings::detail::string_index_pair>(indices.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(indices.size()),
                    duplicates.begin(),
                    collapse_overlaps_fn{d_input_chars, indices.data(), sizes.data()});

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

}  // namespace nvtext
