/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace strings {
namespace detail {

// Helper function for loading 16B from a potentially unaligned memory location to registers.
__forceinline__ __device__ uint4 load_uint4(const char* ptr)
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
__global__ void gather_chars_fn_string_parallel(StringIterator strings_begin,
                                                char* out_chars,
                                                cudf::device_span<int32_t const> const out_offsets,
                                                MapIterator string_indices,
                                                size_type total_out_strings)
{
  constexpr size_t out_datatype_size = sizeof(uint4);
  constexpr size_t in_datatype_size  = sizeof(uint);

  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int global_warp_id   = global_thread_id / cudf::detail::warp_size;
  int warp_lane        = global_thread_id % cudf::detail::warp_size;
  int nwarps           = gridDim.x * blockDim.x / cudf::detail::warp_size;

  auto const alignment_offset = reinterpret_cast<std::uintptr_t>(out_chars) % out_datatype_size;
  uint4* out_chars_aligned    = reinterpret_cast<uint4*>(out_chars - alignment_offset);

  for (size_type istring = global_warp_id; istring < total_out_strings; istring += nwarps) {
    auto const out_start = out_offsets[istring];
    auto const out_end   = out_offsets[istring + 1];

    // This check is necessary because string_indices[istring] may be out of bound.
    if (out_start == out_end) continue;

    const char* in_start = strings_begin[string_indices[istring]].data();

    // Both `out_start_aligned` and `out_end_aligned` are indices into `out_chars`.
    // `out_start_aligned` is the first 16B aligned memory location after `out_start + 4`.
    // `out_end_aligned` is the last 16B aligned memory location before `out_end - 4`. Characters
    // between `[out_start_aligned, out_end_aligned)` will be copied using uint4.
    // `out_start + 4` and `out_end - 4` are used instead of `out_start` and `out_end` to avoid
    // `load_uint4` reading beyond string boundaries.
    int32_t out_start_aligned =
      (out_start + in_datatype_size + alignment_offset + out_datatype_size - 1) /
        out_datatype_size * out_datatype_size -
      alignment_offset;
    int32_t out_end_aligned =
      (out_end - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size -
      alignment_offset;

    for (size_type ichar = out_start_aligned + warp_lane * out_datatype_size;
         ichar < out_end_aligned;
         ichar += cudf::detail::warp_size * out_datatype_size) {
      *(out_chars_aligned + (ichar + alignment_offset) / out_datatype_size) =
        load_uint4(in_start + ichar - out_start);
    }

    // Tail logic: copy characters of the current string outside `[out_start_aligned,
    // out_end_aligned)`.
    if (out_end_aligned <= out_start_aligned) {
      // In this case, `[out_start_aligned, out_end_aligned)` is an empty set, and we copy the
      // entire string.
      for (int32_t ichar = out_start + warp_lane; ichar < out_end;
           ichar += cudf::detail::warp_size) {
        out_chars[ichar] = in_start[ichar - out_start];
      }
    } else {
      // Copy characters in range `[out_start, out_start_aligned)`.
      if (out_start + warp_lane < out_start_aligned) {
        out_chars[out_start + warp_lane] = in_start[warp_lane];
      }
      // Copy characters in range `[out_end_aligned, out_end)`.
      int32_t ichar = out_end_aligned + warp_lane;
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
__global__ void gather_chars_fn_char_parallel(StringIterator strings_begin,
                                              char* out_chars,
                                              cudf::device_span<int32_t const> const out_offsets,
                                              MapIterator string_indices,
                                              size_type total_out_strings)
{
  __shared__ int32_t out_offsets_threadblock[strings_per_threadblock + 1];

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

  for (int32_t out_ibyte = threadIdx.x + out_offsets_threadblock[0];
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
    int32_t icharacter = out_ibyte - out_offsets_threadblock[string_idx];

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
std::unique_ptr<cudf::column> gather_chars(StringIterator strings_begin,
                                           MapIterator map_begin,
                                           MapIterator map_end,
                                           cudf::device_span<int32_t const> const offsets,
                                           size_type chars_bytes,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto const output_count = std::distance(map_begin, map_end);
  if (output_count == 0) return make_empty_column(type_id::INT8);

  auto chars_column  = create_chars_child_column(chars_bytes, stream, mr);
  auto const d_chars = chars_column->mutable_view().template data<char>();

  constexpr int warps_per_threadblock = 4;
  // String parallel strategy will be used if average string length is above this threshold.
  // Otherwise, char parallel strategy will be used.
  constexpr size_type string_parallel_threshold = 32;

  size_type average_string_length = chars_bytes / output_count;

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
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column containing the gathered strings.
 */
template <bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<cudf::column> gather(strings_column_view const& strings,
                                     MapIterator begin,
                                     MapIterator end,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  auto const output_count  = std::distance(begin, end);
  auto const strings_count = strings.size();
  if (output_count == 0) return make_empty_column(type_id::STRING);

  // allocate offsets column and use memory to compute string size in each output row
  auto out_offsets_column = make_numeric_column(
    data_type{type_id::INT32}, output_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_out_offsets = out_offsets_column->mutable_view().template data<int32_t>();
  auto const d_in_offsets  = (strings_count > 0) ? strings.offsets_begin() : nullptr;
  auto const d_strings     = column_device_view::create(strings.parent(), stream);
  thrust::transform(
    rmm::exec_policy(stream),
    begin,
    end,
    d_out_offsets,
    [d_strings = *d_strings, d_in_offsets, strings_count] __device__(size_type in_idx) {
      if (NullifyOutOfBounds && (in_idx < 0 || in_idx >= strings_count)) return 0;
      if (not d_strings.is_valid(in_idx)) return 0;
      return d_in_offsets[in_idx + 1] - d_in_offsets[in_idx];
    });

  // check total size is not too large
  size_t const total_bytes = thrust::transform_reduce(
    rmm::exec_policy(stream),
    d_out_offsets,
    d_out_offsets + output_count,
    [] __device__(auto size) { return static_cast<size_t>(size); },
    size_t{0},
    thrust::plus{});
  CUDF_EXPECTS(total_bytes < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "total size of output strings is too large for a cudf column");

  // In-place convert output sizes into offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + output_count + 1, d_out_offsets);

  // build chars column
  cudf::device_span<int32_t const> const d_out_offsets_span(d_out_offsets, output_count + 1);
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
                                     rmm::mr::device_memory_resource* mr)
{
  if (nullify_out_of_bounds) return gather<true>(strings, begin, end, stream, mr);
  return gather<false>(strings, begin, end, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
