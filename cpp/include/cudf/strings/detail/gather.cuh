/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace strings {
namespace detail {

// Helper function for loading 16B from a potentially unaligned memory location to registers.
__forceinline__ __device__ uint4 load_uint4(char const* ptr)
{
  auto const offset       = reinterpret_cast<std::uintptr_t>(ptr) % 4;
  auto const* aligned_ptr = reinterpret_cast<unsigned int const*>(ptr - offset);
  auto const shift        = offset * 8;

  uint4 regs = {aligned_ptr[0], aligned_ptr[1], aligned_ptr[2], aligned_ptr[3]};
  uint tail  = 0;
  if (shift) tail = aligned_ptr[4];

  regs.x = __funnelshift_r(regs.x, regs.y, shift);
  regs.y = __funnelshift_r(regs.y, regs.z, shift);
  regs.z = __funnelshift_r(regs.z, regs.w, shift);
  regs.w = __funnelshift_r(regs.w, tail, shift);

  return regs;
}

/**
 * @brief Gather characters from the input iterator, with string parallel strategy.
 *
 * This strategy assigns strings to warps so that each warp can cooperatively copy from the input
 * location of the string to the corresponding output location. Large datatype (uint4) is used for
 * stores. This strategy is best suited for large strings.
 *
 * @tparam StringIterator Iterator should produce `string_view` objects.
 * @tparam MapIterator Iterator for retrieving integer indices of the `StringIterator`.
 *
 * @param strings_begin Start of the iterator to retrieve `string_view` instances.
 * @param out_chars Output buffer for gathered characters.
 * @param out_offsets The offset values associated with the output buffer.
 * @param string_indices Start of index iterator.
 * @param total_out_strings Number of output strings to be gathered.
 */
template <typename StringIterator, typename MapIterator>
CUDF_KERNEL void gather_chars_fn_string_parallel(StringIterator strings_begin,
                                                 char* out_chars,
                                                 cudf::detail::input_offsetalator const out_offsets,
                                                 MapIterator string_indices,
                                                 size_type total_out_strings)
{
  constexpr size_t out_datatype_size = sizeof(uint4);
  constexpr size_t in_datatype_size  = sizeof(uint);

  auto const global_thread_id = cudf::detail::grid_1d::global_thread_id();
  auto const global_warp_id   = global_thread_id / cudf::detail::warp_size;
  auto const warp_lane        = global_thread_id % cudf::detail::warp_size;
  auto const nwarps           = cudf::detail::grid_1d::grid_stride() / cudf::detail::warp_size;

  auto const alignment_offset = reinterpret_cast<std::uintptr_t>(out_chars) % out_datatype_size;
  uint4* out_chars_aligned    = reinterpret_cast<uint4*>(out_chars - alignment_offset);

  for (auto istring = global_warp_id; istring < total_out_strings; istring += nwarps) {
    auto const out_start = out_offsets[istring];
    auto const out_end   = out_offsets[istring + 1];

    // This check is necessary because string_indices[istring] may be out of bound.
    if (out_start == out_end) continue;

    char const* in_start = strings_begin[string_indices[istring]].data();

    // Both `out_start_aligned` and `out_end_aligned` are indices into `out_chars`.
    // `out_start_aligned` is the first 16B aligned memory location after `out_start + 4`.
    // `out_end_aligned` is the last 16B aligned memory location before `out_end - 4`. Characters
    // between `[out_start_aligned, out_end_aligned)` will be copied using uint4.
    // `out_start + 4` and `out_end - 4` are used instead of `out_start` and `out_end` to avoid
    // `load_uint4` reading beyond string boundaries.
    int64_t const out_start_aligned =
      (out_start + in_datatype_size + alignment_offset + out_datatype_size - 1) /
        out_datatype_size * out_datatype_size -
      alignment_offset;
    int64_t const out_end_aligned =
      (out_end - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size -
      alignment_offset;

    for (int64_t ichar = out_start_aligned + warp_lane * out_datatype_size; ichar < out_end_aligned;
         ichar += cudf::detail::warp_size * out_datatype_size) {
      *(out_chars_aligned + (ichar + alignment_offset) / out_datatype_size) =
        load_uint4(in_start + ichar - out_start);
    }

    // Copy characters of the current string outside [out_start_aligned, out_end_aligned)
    if (out_end_aligned <= out_start_aligned) {
      // In this case, `[out_start_aligned, out_end_aligned)` is an empty set, and we copy the
      // entire string.
      for (auto ichar = out_start + warp_lane; ichar < out_end; ichar += cudf::detail::warp_size) {
        out_chars[ichar] = in_start[ichar - out_start];
      }
    } else {
      // Copy characters in range `[out_start, out_start_aligned)`.
      if (out_start + warp_lane < out_start_aligned) {
        out_chars[out_start + warp_lane] = in_start[warp_lane];
      }
      // Copy characters in range `[out_end_aligned, out_end)`.
      auto const ichar = out_end_aligned + warp_lane;
      if (ichar < out_end) { out_chars[ichar] = in_start[ichar - out_start]; }
    }
  }
}

/**
 * @brief Gather characters from the input iterator, with char parallel strategy.
 *
 * This strategy assigns characters to threads, and uses binary search for getting the string
 * index. To improve the binary search performance, fixed number of strings per threadblock is
 * used. This strategy is best suited for small strings.
 *
 * @tparam StringIterator Iterator should produce `string_view` objects.
 * @tparam MapIterator Iterator for retrieving integer indices of the `StringIterator`.
 *
 * @param strings_begin Start of the iterator to retrieve `string_view` instances.
 * @param out_chars Output buffer for gathered characters.
 * @param out_offsets The offset values associated with the output buffer.
 * @param string_indices Start of index iterator.
 * @param total_out_strings Number of output strings to be gathered.
 */
template <int strings_per_threadblock, typename StringIterator, typename MapIterator>
CUDF_KERNEL void gather_chars_fn_char_parallel(StringIterator strings_begin,
                                               char* out_chars,
                                               cudf::detail::input_offsetalator const out_offsets,
                                               MapIterator string_indices,
                                               size_type total_out_strings)
{
  __shared__ int64_t out_offsets_threadblock[strings_per_threadblock + 1];

  // Current thread block will process output strings starting at `begin_out_string_idx`.
  size_type begin_out_string_idx = blockIdx.x * strings_per_threadblock;

  // Number of strings to be processed by the current threadblock.
  size_type strings_current_threadblock =
    min(strings_per_threadblock, total_out_strings - begin_out_string_idx);

  if (strings_current_threadblock <= 0) return;

  // Collectively load offsets of strings processed by the current thread block.
  for (size_type idx = threadIdx.x; idx <= strings_current_threadblock; idx += blockDim.x) {
    out_offsets_threadblock[idx] = out_offsets[idx + begin_out_string_idx];
  }
  __syncthreads();

  for (int64_t out_ibyte = threadIdx.x + out_offsets_threadblock[0];
       out_ibyte < out_offsets_threadblock[strings_current_threadblock];
       out_ibyte += blockDim.x) {
    // binary search for the string index corresponding to out_ibyte
    auto const string_idx_iter =
      thrust::prev(thrust::upper_bound(thrust::seq,
                                       out_offsets_threadblock,
                                       out_offsets_threadblock + strings_current_threadblock,
                                       out_ibyte));
    size_type string_idx = thrust::distance(out_offsets_threadblock, string_idx_iter);

    // calculate which character to load within the string
    auto const icharacter = out_ibyte - out_offsets_threadblock[string_idx];

    size_type in_string_idx = string_indices[begin_out_string_idx + string_idx];
    out_chars[out_ibyte]    = strings_begin[in_string_idx].data()[icharacter];
  }
}

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
 * @param strings_begin Start of the iterator to retrieve `string_view` instances.
 * @param map_begin Start of index iterator.
 * @param map_end End of index iterator.
 * @param offsets The offset values to be associated with the output chars column.
 * @param chars_bytes The total number of bytes for the output chars column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New chars column fit for a strings column.
 */
template <typename StringIterator, typename MapIterator>
rmm::device_uvector<char> gather_chars(StringIterator strings_begin,
                                       MapIterator map_begin,
                                       MapIterator map_end,
                                       cudf::detail::input_offsetalator const offsets,
                                       int64_t chars_bytes,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const output_count = std::distance(map_begin, map_end);
  if (output_count == 0) return rmm::device_uvector<char>(0, stream, mr);

  auto chars_data = rmm::device_uvector<char>(chars_bytes, stream, mr);
  cudf::experimental::prefetch::detail::prefetch("gather", chars_data, stream);
  auto d_chars = chars_data.data();

  constexpr int warps_per_threadblock = 4;
  // String parallel strategy will be used if average string length is above this threshold.
  // Otherwise, char parallel strategy will be used.
  constexpr int64_t string_parallel_threshold = 32;

  int64_t const average_string_length = chars_bytes / output_count;

  if (average_string_length > string_parallel_threshold) {
    constexpr int max_threadblocks = 65536;
    gather_chars_fn_string_parallel<<<
      min((static_cast<int>(output_count) + warps_per_threadblock - 1) / warps_per_threadblock,
          max_threadblocks),
      warps_per_threadblock * cudf::detail::warp_size,
      0,
      stream.value()>>>(strings_begin, d_chars, offsets, map_begin, output_count);
  } else {
    constexpr int strings_per_threadblock = 32;
    gather_chars_fn_char_parallel<strings_per_threadblock>
      <<<(output_count + strings_per_threadblock - 1) / strings_per_threadblock,
         warps_per_threadblock * cudf::detail::warp_size,
         0,
         stream.value()>>>(strings_begin, d_chars, offsets, map_begin, output_count);
  }

  return chars_data;
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
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column containing the gathered strings.
 */
template <bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<cudf::column> gather(strings_column_view const& strings,
                                     MapIterator begin,
                                     MapIterator end,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const output_count = std::distance(begin, end);
  if (output_count == 0) return make_empty_column(type_id::STRING);

  // build offsets column
  auto const d_strings    = column_device_view::create(strings.parent(), stream);
  auto const d_in_offsets = cudf::detail::offsetalator_factory::make_input_iterator(
    strings.is_empty() ? make_empty_column(type_id::INT32)->view() : strings.offsets(),
    strings.offset());

  auto sizes_itr = thrust::make_transform_iterator(
    begin,
    cuda::proclaim_return_type<size_type>(
      [d_strings = *d_strings, d_in_offsets] __device__(size_type idx) {
        if (NullifyOutOfBounds && (idx < 0 || idx >= d_strings.size())) { return 0; }
        if (not d_strings.is_valid(idx)) { return 0; }
        return static_cast<size_type>(d_in_offsets[idx + 1] - d_in_offsets[idx]);
      }));
  auto [out_offsets_column, total_bytes] = cudf::strings::detail::make_offsets_child_column(
    sizes_itr, sizes_itr + output_count, stream, mr);

  // build chars column
  auto const offsets_view =
    cudf::detail::offsetalator_factory::make_input_iterator(out_offsets_column->view());
  cudf::experimental::prefetch::detail::prefetch(
    "gather", strings.chars_begin(stream), strings.chars_size(stream), stream);
  auto out_chars_data = gather_chars(
    d_strings->begin<string_view>(), begin, end, offsets_view, total_bytes, stream, mr);

  return make_strings_column(output_count,
                             std::move(out_offsets_column),
                             out_chars_data.release(),
                             0,  // caller sets these
                             rmm::device_buffer{});
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
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column containing the gathered strings.
 */
template <typename MapIterator>
std::unique_ptr<cudf::column> gather(strings_column_view const& strings,
                                     MapIterator begin,
                                     MapIterator end,
                                     bool nullify_out_of_bounds,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  if (nullify_out_of_bounds) return gather<true>(strings, begin, end, stream, mr);
  return gather<false>(strings, begin, end, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
