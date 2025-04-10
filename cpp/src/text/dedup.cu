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

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cuda/std/limits>
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

struct sort_comparator_fn {
  cudf::device_span<char const> chars_span;
  __device__ bool operator()(int32_t lhs, int32_t rhs) const
  {
    constexpr auto max_size = cuda::std::numeric_limits<int32_t>::max();
    auto const chars_size   = static_cast<int32_t>(chars_span.size());

    auto const lhs_size = cuda::std::min(max_size, chars_size - lhs);
    auto const rhs_size = cuda::std::min(max_size, chars_size - rhs);
    auto const lh_str   = cudf::string_view(chars_span.data() + lhs, lhs_size);
    auto const rh_str   = cudf::string_view(chars_span.data() + rhs, rhs_size);
    return lh_str < rh_str;
  }
};

__device__ int32_t count_common_bytes(cudf::string_view lhs, cudf::string_view rhs)
{
  auto const size1 = lhs.size_bytes();
  auto const size2 = rhs.size_bytes();
  auto const* ptr1 = lhs.data();
  auto const* ptr2 = rhs.data();

  int32_t idx = 0;
  while ((idx < size1) && (idx < size2) && (*ptr1++ == *ptr2++)) {
    ++idx;
  }
  return idx;
}

struct find_duplicates_fn {
  cudf::device_span<char const> chars_span;
  int32_t width;
  int32_t const* d_indices;
  __device__ int16_t operator()(int32_t idx) const
  {
    if (idx == 0) { return 0; }
    constexpr auto max_size = cuda::std::numeric_limits<int32_t>::max();
    auto const chars_size   = static_cast<int32_t>(chars_span.size());

    auto const lhs      = d_indices[idx - 1];
    auto const rhs      = d_indices[idx];
    auto const lhs_size = cuda::std::min(max_size, chars_size - lhs);
    auto const rhs_size = cuda::std::min(max_size, chars_size - rhs);

    auto const lh_str = cudf::string_view(chars_span.data() + lhs, lhs_size);
    auto const rh_str = cudf::string_view(chars_span.data() + rhs, rhs_size);

    constexpr auto max_run_length = static_cast<int32_t>(cuda::std::numeric_limits<int16_t>::max());

    auto const size = cuda::std::min(count_common_bytes(lh_str, rh_str), max_run_length);

    return size >= width ? static_cast<int16_t>(size) : 0;
  }
};

struct collapse_overlaps_fn {
  char const* d_chars;
  int32_t const* d_offsets;
  int16_t const* d_sizes;
  __device__ string_index operator()(int32_t idx) const
  {
    constexpr auto max_run_length = static_cast<int32_t>(cuda::std::numeric_limits<int16_t>::max());

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

std::unique_ptr<rmm::device_uvector<int32_t>> build_suffix_array_fn(
  cudf::device_span<char const> chars_span,
  int32_t min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const size = static_cast<int32_t>(chars_span.size()) - min_width + 1;
  auto indices    = rmm::device_uvector<int32_t>(size, stream);

  auto const cmp_op = sort_comparator_fn{chars_span};
  auto const seq    = thrust::make_counting_iterator<int32_t>(0);
  auto tmp_bytes    = std::size_t{0};
  cub::DeviceMergeSort::SortKeysCopy(
    nullptr, tmp_bytes, seq, indices.begin(), indices.size(), cmp_op, stream.value());
  auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
  cub::DeviceMergeSort::SortKeysCopy(
    tmp_stg.data(), tmp_bytes, seq, indices.begin(), indices.size(), cmp_op, stream.value());

  return std::make_unique<rmm::device_uvector<int32_t>>(std::move(indices));
}

std::unique_ptr<cudf::column> resolve_duplicates_fn(cudf::device_span<char const> chars_span,
                                                    cudf::device_span<int32_t const> indices,
                                                    int32_t min_width,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  auto sizes = rmm::device_uvector<int16_t>(indices.size(), stream);

  // locate candidate duplicates within the suffix array
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(indices.size()),
                    sizes.begin(),
                    find_duplicates_fn{chars_span, min_width, indices.data()});

  auto const dup_count =
    sizes.size() - thrust::count(rmm::exec_policy(stream), sizes.begin(), sizes.end(), 0);
  auto dup_indices = rmm::device_uvector<int32_t>(dup_count, stream);

  // remove the non-candidate entries from indices and sizes
  thrust::remove_copy_if(
    rmm::exec_policy(stream),
    indices.begin(),
    indices.end(),
    thrust::counting_iterator<int32_t>(0),
    dup_indices.begin(),
    [d_sizes = sizes.data()] __device__(int32_t idx) -> bool { return d_sizes[idx] == 0; });
  auto end = thrust::remove(rmm::exec_policy(stream), sizes.begin(), sizes.end(), 0);
  sizes.resize(thrust::distance(sizes.begin(), end), stream);

  // sort the resulting indices/sizes for overlap filtering
  thrust::sort_by_key(
    rmm::exec_policy_nosync(stream), dup_indices.begin(), dup_indices.end(), sizes.begin());

  // produce final duplicates for make_strings_column and collapse any overlapping candidates
  auto duplicates =
    rmm::device_uvector<cudf::strings::detail::string_index_pair>(dup_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int32_t>(0),
                    thrust::counting_iterator<int32_t>(dup_indices.size()),
                    duplicates.begin(),
                    collapse_overlaps_fn{chars_span.data(), dup_indices.data(), sizes.data()});

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

  // ironically remove duplicates from the sorted list
  duplicates.resize(
    thrust::distance(duplicates.begin(),
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

std::unique_ptr<rmm::device_uvector<int32_t>> build_suffix_array(
  cudf::strings_column_view const& input,
  int32_t min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);

  auto const d_input_chars = input.chars_begin(stream) + first_offset;
  auto const chars_size    = last_offset - first_offset;
  CUDF_EXPECTS(min_width < chars_size, "min_width value cannot exceed the input size");
  CUDF_EXPECTS(chars_size < std::numeric_limits<int32_t>::max(),
               "input size cannot exceed the 2GB");

  auto const chars_span = cudf::device_span<char const>(d_input_chars, chars_size);
  return build_suffix_array_fn(chars_span, min_width, stream, mr);
}

std::unique_ptr<cudf::column> resolve_duplicates(cudf::strings_column_view const& input,
                                                 cudf::device_span<int32_t const> indices,
                                                 int32_t min_width,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(min_width > 8, "min_width should be at least 8");

  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const d_input_chars = input.chars_begin(stream) + first_offset;
  auto const chars_size    = last_offset - first_offset;
  auto const chars_span    = cudf::device_span<char const>(d_input_chars, chars_size);

  return resolve_duplicates_fn(chars_span, indices, min_width, stream, mr);
}

std::unique_ptr<cudf::column> substring_duplicates(cudf::strings_column_view const& input,
                                                   int32_t min_width,
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

namespace {

struct find_duplicates2_fn {
  cudf::device_span<char const> chars_span1;
  cudf::device_span<char const> chars_span2;
  int32_t width;
  cudf::device_span<int32_t const> d_indices1;
  cudf::device_span<int32_t const> d_indices2;
  cudf::device_span<int32_t const> lb_indices;
  cudf::device_span<int32_t const> ub_indices;

  __device__ int16_t operator()(int32_t idx) const
  {
    auto const lhs_idx  = d_indices1[idx];
    auto const lhs_size = static_cast<cudf::size_type>(chars_span1.size() - lhs_idx);
    auto const lh_str   = cudf::string_view(chars_span1.data() + lhs_idx, lhs_size);

    auto const itr = cudf::detail::make_counting_transform_iterator(
      0, [d_chars = chars_span2.data(), d_indices = d_indices2, cs = chars_span2.size()](auto idx) {
        auto const rhs_idx  = d_indices[idx];
        auto const rhs_size = static_cast<cudf::size_type>(cs - rhs_idx);
        return cudf::string_view(d_chars + rhs_idx, rhs_size);
      });
    auto const lb_idx = lb_indices[idx];
    auto const ub_idx = ub_indices[idx];
    auto const begin  = itr + lb_idx;
    auto const end    = itr + ub_idx + (ub_idx < d_indices2.size());

    auto const fnd = thrust::lower_bound(thrust::seq, begin, end, lh_str);
    if (fnd == end) { return 0; }
    auto size = count_common_bytes(lh_str, *fnd);

    // check for lower_bound edge case
    auto const ridx = lb_idx + thrust::distance(begin, fnd) - (lb_idx > 0);
    if (size < width && ridx >= 0) {
      auto const rhs_idx  = d_indices2[ridx];
      auto const rhs_size = static_cast<cudf::size_type>(chars_span2.size() - rhs_idx);
      auto const rh_str   = cudf::string_view(chars_span2.data() + rhs_idx, rhs_size);

      size = count_common_bytes(lh_str, rh_str);
    }

    constexpr auto max_run_length = static_cast<int32_t>(cuda::std::numeric_limits<int16_t>::max());

    size = cuda::std::min(size, max_run_length);
    return size >= width ? static_cast<int16_t>(size) : 0;
  }
};

struct lb_fn {
  cudf::device_span<char const> chars;
  __device__ cudf::string_view operator()(cudf::size_type idx) const
  {
    auto const size = static_cast<cudf::size_type>(chars.size() - idx);
    return cudf::string_view(chars.data() + idx, size);
  }
};

struct prefix_string_fn {
  __device__ uint32_t operator()(cudf::string_view str) const
  {
    uint32_t data   = 0;
    auto const size = cuda::std::min(static_cast<size_t>(str.size_bytes()), sizeof(uint32_t));
    memcpy(&data, str.data(), size);
    return __byte_perm(data, 0, 0x0123);
  }
};

struct mini_fn {
  cudf::device_span<char const> chars;
  __device__ uint32_t operator()(cudf::size_type idx) const
  {
    auto const size = cuda::std::min((chars.size() - idx), sizeof(uint32_t));
    uint32_t data   = 0;
    memcpy(&data, chars.data() + idx, size);
    return __byte_perm(data, 0, 0x0123);
  }
};

std::unique_ptr<cudf::column> resolve_duplicates_pair_impl(
  cudf::strings_column_view const& input1,
  cudf::device_span<int32_t const> indices1,
  cudf::strings_column_view const& input2,
  cudf::device_span<int32_t const> indices2,
  int32_t min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(min_width > 8, "min_width should be at least 8");
  auto d_strings1 = cudf::column_device_view::create(input1.parent(), stream);
  auto d_strings2 = cudf::column_device_view::create(input2.parent(), stream);

  auto [first_offset1, last_offset1] =
    cudf::strings::detail::get_first_and_last_offset(input1, stream);
  auto [first_offset2, last_offset2] =
    cudf::strings::detail::get_first_and_last_offset(input2, stream);

  auto d_input_chars1 = input1.chars_begin(stream) + first_offset1;
  auto d_input_chars2 = input2.chars_begin(stream) + first_offset2;
  auto chars_size1    = last_offset1 - first_offset1;
  auto chars_size2    = last_offset2 - first_offset2;
  CUDF_EXPECTS(min_width < chars_size1, "min_width value cannot exceed the input size");
  CUDF_EXPECTS(min_width < chars_size2, "min_width value cannot exceed the input size");

  auto const chars_span1 = cudf::device_span<char const>(d_input_chars1, chars_size1);
  auto const chars_span2 = cudf::device_span<char const>(d_input_chars2, chars_size2);

  auto itr1 = thrust::make_transform_iterator(indices1.begin(), mini_fn{chars_span1});
  auto end1 = itr1 + indices1.size();
  auto itr2 = thrust::make_transform_iterator(indices2.begin(), lb_fn{chars_span2});
  auto end2 = itr2 + indices2.size();

  auto pd1 = rmm::device_uvector<uint32_t>(indices2.size(), stream);  // 4x
  thrust::transform(rmm::exec_policy_nosync(stream), itr2, end2, pd1.begin(), prefix_string_fn{});

  auto lb1 = rmm::device_uvector<int32_t>(indices1.size(), stream);  // 4x
  thrust::lower_bound(
    rmm::exec_policy_nosync(stream), pd1.begin(), pd1.end(), itr1, end1, lb1.begin());
  auto ub1 = rmm::device_uvector<int32_t>(indices1.size(), stream);  // 4x
  thrust::upper_bound(
    rmm::exec_policy_nosync(stream), pd1.begin(), pd1.end(), itr1, end1, ub1.begin());

  auto fd2 = find_duplicates2_fn{chars_span1, chars_span2, min_width, indices1, indices2, lb1, ub1};
  auto sizes = rmm::device_uvector<int16_t>(indices1.size(), stream);  // 2x
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int32_t>(0),
                    thrust::counting_iterator<int32_t>(sizes.size()),
                    sizes.begin(),
                    fd2);

  // candidate duplicates from input1 have matched the input2 data at least min_width;
  // this means any duplicates in both inputs should be reflected in indices1/sizes;
  // so we should be able to filter/collapse the results using only indices1/sizes
  auto const dup_count =
    sizes.size() - thrust::count(rmm::exec_policy(stream), sizes.begin(), sizes.end(), 0);
  auto dup_indices = rmm::device_uvector<int32_t>(dup_count, stream);

  // remove the non-candidate entries from indices and sizes
  thrust::remove_copy_if(
    rmm::exec_policy(stream),
    indices1.begin(),
    indices1.end(),
    thrust::counting_iterator<int32_t>(0),
    dup_indices.begin(),
    [d_sizes = sizes.data()] __device__(int32_t idx) -> bool { return d_sizes[idx] == 0; });
  auto end = thrust::remove(rmm::exec_policy(stream), sizes.begin(), sizes.end(), 0);
  sizes.resize(thrust::distance(sizes.begin(), end), stream);

  // sort the resulting indices/sizes for overlap filtering
  thrust::sort_by_key(
    rmm::exec_policy_nosync(stream), dup_indices.begin(), dup_indices.end(), sizes.begin());

  // produce final duplicates for make_strings_column and collapse any overlapping candidates
  auto duplicates =
    rmm::device_uvector<cudf::strings::detail::string_index_pair>(dup_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<int32_t>(0),
                    thrust::counting_iterator<int32_t>(dup_indices.size()),
                    duplicates.begin(),
                    collapse_overlaps_fn{chars_span1.data(), dup_indices.data(), sizes.data()});

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

  // ironically remove duplicates from the sorted list
  duplicates.resize(
    thrust::distance(duplicates.begin(),
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

std::unique_ptr<cudf::column> resolve_duplicates_pair(cudf::strings_column_view const& input1,
                                                      cudf::device_span<int32_t const> indices1,
                                                      cudf::strings_column_view const& input2,
                                                      cudf::device_span<int32_t const> indices2,
                                                      int32_t min_width,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  // check for duplicates between the 2 suffix arrays
  // force the 2nd input to be the smaller one
  return (indices1.size() < indices2.size())
           ? resolve_duplicates_pair_impl(input2, indices2, input1, indices1, min_width, stream, mr)
           : resolve_duplicates_pair_impl(
               input1, indices1, input2, indices2, min_width, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> substring_duplicates(cudf::strings_column_view const& input,
                                                   int32_t min_width,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::substring_duplicates(input, min_width, stream, mr);
}

std::unique_ptr<rmm::device_uvector<int32_t>> build_suffix_array(
  cudf::strings_column_view const& input,
  int32_t min_width,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::build_suffix_array(input, min_width, stream, mr);
}

std::unique_ptr<cudf::column> resolve_duplicates(cudf::strings_column_view const& input,
                                                 cudf::device_span<int32_t const> indices,
                                                 int32_t min_width,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::resolve_duplicates(input, indices, min_width, stream, mr);
}

std::unique_ptr<cudf::column> resolve_duplicates_pair(cudf::strings_column_view const& input1,
                                                      cudf::device_span<int32_t const> indices1,
                                                      cudf::strings_column_view const& input2,
                                                      cudf::device_span<int32_t const> indices2,
                                                      int32_t min_width,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::resolve_duplicates_pair(input1, indices1, input2, indices2, min_width, stream, mr);
}

}  // namespace nvtext
