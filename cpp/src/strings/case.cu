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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/case.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the warp-parallel function is used to compute the output sizes.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the strings_lengths benchmark results.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Utility functions for converting characters to upper or lower case
 */
struct convert_char_fn {
  character_flags_table_type case_flag;
  character_flags_table_type const* d_flags;
  character_cases_table_type const* d_case_table;
  special_case_mapping const* d_special_case_mapping;

  // compute size or copy the bytes representing the special case mapping for this codepoint
  __device__ size_type handle_special_case_bytes(uint32_t code_point,
                                                 detail::character_flags_table_type flag,
                                                 char* d_buffer = nullptr) const
  {
    special_case_mapping m = d_special_case_mapping[get_special_case_hash_index(code_point)];

    size_type bytes   = 0;
    auto const count  = IS_LOWER(flag) ? m.num_upper_chars : m.num_lower_chars;
    auto const* chars = IS_LOWER(flag) ? m.upper : m.lower;
    for (uint16_t idx = 0; idx < count; idx++) {
      bytes += d_buffer
                 ? detail::from_char_utf8(detail::codepoint_to_utf8(chars[idx]), d_buffer + bytes)
                 : detail::bytes_in_char_utf8(detail::codepoint_to_utf8(chars[idx]));
    }
    return bytes;
  }

  // this is called for converting any UTF-8 characters
  __device__ size_type process_character(char_utf8 chr, char* d_buffer = nullptr) const
  {
    auto const code_point = detail::utf8_to_codepoint(chr);

    detail::character_flags_table_type flag = code_point <= 0x00'FFFF ? d_flags[code_point] : 0;

    // we apply special mapping in two cases:
    // - uncased characters with the special mapping flag: always
    // - cased characters with the special mapping flag: when matching the input case_flag
    if (IS_SPECIAL(flag) && ((flag & case_flag) || !IS_UPPER_OR_LOWER(flag))) {
      return handle_special_case_bytes(code_point, case_flag, d_buffer);
    }

    char_utf8 const new_char =
      (flag & case_flag) ? detail::codepoint_to_utf8(d_case_table[code_point]) : chr;
    return (d_buffer) ? detail::from_char_utf8(new_char, d_buffer)
                      : detail::bytes_in_char_utf8(new_char);
  }

  // special function for converting ASCII-only characters
  __device__ char process_ascii(char chr)
  {
    return (case_flag & d_flags[chr]) ? static_cast<char>(d_case_table[chr]) : chr;
  }
};

/**
 * @brief Per string logic for case conversion functions
 *
 * This can be used in calls to make_strings_children.
 */
struct upper_lower_fn {
  convert_char_fn converter;
  column_device_view d_strings;
  size_type* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str = d_strings.element<string_view>(idx);
    size_type bytes  = 0;
    char* d_buffer   = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto const size = converter.process_character(*itr, d_buffer);
      if (d_buffer) {
        d_buffer += size;
      } else {
        bytes += size;
      }
    }
    if (!d_buffer) { d_offsets[idx] = bytes; }
  }
};

/**
 * @brief Count output bytes in warp-parallel threads
 *
 * This executes as one warp per string and just computes the output sizes.
 */
struct count_bytes_fn {
  convert_char_fn converter;
  column_device_view d_strings;
  size_type* d_offsets;

  __device__ void operator()(size_type idx) const
  {
    auto const str_idx  = idx / cudf::detail::warp_size;
    auto const lane_idx = idx % cudf::detail::warp_size;

    // initialize the output for the atomicAdd
    if (lane_idx == 0) { d_offsets[str_idx] = 0; }
    __syncwarp();

    if (d_strings.is_null(str_idx)) { return; }
    auto const d_str   = d_strings.element<string_view>(str_idx);
    auto const str_ptr = d_str.data();

    size_type size = 0;
    for (auto i = lane_idx; i < d_str.size_bytes(); i += cudf::detail::warp_size) {
      auto const chr = str_ptr[i];
      if (is_utf8_continuation_char(chr)) { continue; }
      char_utf8 u8 = 0;
      to_char_utf8(str_ptr + i, u8);
      size += converter.process_character(u8);
    }
    // this is every so slightly faster than using the cub::warp_reduce
    if (size > 0) {
      cuda::atomic_ref<size_type, cuda::thread_scope_block> ref{*(d_offsets + str_idx)};
      ref.fetch_add(size, cuda::std::memory_order_relaxed);
    }
  }
};

/**
 * @brief Special functor for processing ASCII-only data
 */
struct ascii_converter_fn {
  convert_char_fn converter;
  __device__ char operator()(char chr) { return converter.process_ascii(chr); }
};

/**
 * @brief Utility method for converting upper and lower case characters
 * in a strings column
 *
 * @param input Strings to convert
 * @param case_flag The character type to convert (upper, lower, or both)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with characters converted
 */
std::unique_ptr<column> convert_case(strings_column_view const& input,
                                     character_flags_table_type case_flag,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  if (input.size() == input.null_count()) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_flags   = get_character_flags_table();
  auto const d_cases   = get_character_cases_table();
  auto const d_special = get_special_case_mapping_table();

  convert_char_fn ccfn{case_flag, d_flags, d_cases, d_special};
  upper_lower_fn converter{ccfn, *d_strings};

  // For smaller strings, use the regular string-parallel algorithm
  if ((input.chars_size(stream) / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    auto [offsets, chars] =
      cudf::strings::detail::make_strings_children(converter, input.size(), stream, mr);
    return make_strings_column(input.size(),
                               std::move(offsets),
                               std::move(chars),
                               input.null_count(),
                               cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  // Check if the input contains any multi-byte characters.
  // This check incurs ~20% performance hit for smaller strings and so we only use it
  // after the threshold check above. The check makes very little impact for larger strings
  // but results in a large performance gain when the input contains only single-byte characters.
  // The count_if is faster than any_of or all_of: https://github.com/NVIDIA/thrust/issues/1016
  bool const multi_byte_chars =
    thrust::count_if(rmm::exec_policy(stream),
                     input.chars_begin(stream),
                     input.chars_end(stream),
                     cuda::proclaim_return_type<bool>(
                       [] __device__(auto chr) { return is_utf8_continuation_char(chr); })) > 0;
  if (!multi_byte_chars) {
    // optimization for ASCII-only case: copy the input column and inplace replace each character
    auto result           = std::make_unique<column>(input.parent(), stream, mr);
    auto d_chars          = result->mutable_view().head<char>();
    auto const chars_size = strings_column_view(result->view()).chars_size(stream);
    thrust::transform(
      rmm::exec_policy(stream), d_chars, d_chars + chars_size, d_chars, ascii_converter_fn{ccfn});
    result->set_null_count(input.null_count());
    return result;
  }

  // This will use a warp-parallel algorithm to compute the output sizes for each string
  // and then uses the normal string parallel functor to build the output.
  auto offsets = make_numeric_column(
    data_type{type_to_id<size_type>()}, input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  auto d_offsets = offsets->mutable_view().data<size_type>();

  // first pass, compute output sizes
  // note: tried to use segmented-reduce approach instead here and it was consistently slower
  count_bytes_fn counter{ccfn, *d_strings, d_offsets};
  auto const count_itr = thrust::make_counting_iterator<size_type>(0);
  thrust::for_each_n(
    rmm::exec_policy(stream), count_itr, input.size() * cudf::detail::warp_size, counter);

  // convert sizes to offsets
  auto const bytes =
    cudf::detail::sizes_to_offsets(d_offsets, d_offsets + input.size() + 1, d_offsets, stream);
  CUDF_EXPECTS(bytes <= std::numeric_limits<size_type>::max(),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  auto chars = create_chars_child_column(static_cast<size_type>(bytes), stream, mr);
  // second pass, write output
  converter.d_offsets = d_offsets;
  converter.d_chars   = chars->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream), count_itr, input.size(), converter);

  return make_strings_column(input.size(),
                             std::move(offsets),
                             std::move(chars),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> to_lower(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  character_flags_table_type case_flag = IS_UPPER(0xFF);  // convert only upper case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> to_upper(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  character_flags_table_type case_flag = IS_LOWER(0xFF);  // convert only lower case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> swapcase(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  // convert only upper or lower case characters
  character_flags_table_type case_flag = IS_LOWER(0xFF) | IS_UPPER(0xFF);
  return convert_case(strings, case_flag, stream, mr);
}

}  // namespace detail

// APIs

std::unique_ptr<column> to_lower(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_lower(strings, stream, mr);
}

std::unique_ptr<column> to_upper(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_upper(strings, stream, mr);
}

std::unique_ptr<column> swapcase(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::swapcase(strings, stream, mr);
}

}  // namespace strings
}  // namespace cudf
