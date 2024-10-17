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
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/case.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/merge.h>
#include <thrust/transform.h>

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
struct base_upper_lower_fn {
  convert_char_fn converter;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  base_upper_lower_fn(convert_char_fn converter) : converter(converter) {}

  __device__ inline void process_string(string_view d_str, size_type idx) const
  {
    size_type bytes = 0;
    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (auto itr = d_str.data(); itr < (d_str.data() + d_str.size_bytes()); ++itr) {
      if (is_utf8_continuation_char(static_cast<u_char>(*itr))) continue;
      char_utf8 chr = 0;
      to_char_utf8(itr, chr);
      auto const size = converter.process_character(chr, d_buffer);
      if (d_buffer) {
        d_buffer += size;
      } else {
        bytes += size;
      }
    }
    if (!d_buffer) { d_sizes[idx] = bytes; }
  }
};

struct upper_lower_fn : public base_upper_lower_fn {
  column_device_view d_strings;

  upper_lower_fn(convert_char_fn converter, column_device_view const& d_strings)
    : base_upper_lower_fn{converter}, d_strings{d_strings}
  {
  }

  __device__ void operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str = d_strings.element<string_view>(idx);
    process_string(d_str, idx);
  }
};

// Long strings are divided into smaller strings using this value as a guide.
// Generally strings are split into sub-blocks of bytes of this size but
// care is taken to not sub-block in the middle of a multi-byte character.
constexpr size_type LS_SUB_BLOCK_SIZE = 32;

/**
 * @brief Produces sub-offsets for the chars in the given strings column
 */
struct sub_offset_fn {
  char const* d_input_chars;
  int64_t first_offset;
  int64_t last_offset;

  __device__ int64_t operator()(int64_t idx) const
  {
    auto const end = d_input_chars + last_offset;
    auto position  = (idx + 1) * LS_SUB_BLOCK_SIZE;
    auto begin     = d_input_chars + first_offset + position;
    while ((begin < end) && is_utf8_continuation_char(static_cast<u_char>(*begin))) {
      ++begin;
      ++position;
    }
    return (begin < end) ? position + first_offset : last_offset;
  }
};

/**
 * @brief Specialized case conversion for long strings
 *
 * This is needed since the offset count can exceed size_type.
 * Also, nulls are ignored since this purely builds the output chars.
 * The d_offsets are only temporary to help address the sub-blocks.
 */
struct upper_lower_ls_fn : public base_upper_lower_fn {
  convert_char_fn converter;
  char const* d_input_chars;
  int64_t* d_input_offsets;  // includes column offset

  upper_lower_ls_fn(convert_char_fn converter, char const* d_input_chars, int64_t* d_input_offsets)
    : base_upper_lower_fn{converter}, d_input_chars{d_input_chars}, d_input_offsets{d_input_offsets}
  {
  }

  // idx is row index
  __device__ void operator()(size_type idx) const
  {
    auto const offset = d_input_offsets[idx];
    auto const d_str  = string_view{d_input_chars + offset,
                                   static_cast<size_type>(d_input_offsets[idx + 1] - offset)};
    process_string(d_str, idx);
  }
};

/**
 * @brief Count output bytes in warp-parallel threads
 *
 * This executes as one warp per string and just computes the output sizes.
 */
CUDF_KERNEL void count_bytes_kernel(convert_char_fn converter,
                                    column_device_view d_strings,
                                    size_type* d_sizes)
{
  auto idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= (d_strings.size() * cudf::detail::warp_size)) { return; }

  auto const str_idx  = idx / cudf::detail::warp_size;
  auto const lane_idx = idx % cudf::detail::warp_size;

  // initialize the output for the atomicAdd
  if (lane_idx == 0) { d_sizes[str_idx] = 0; }
  __syncwarp();

  if (d_strings.is_null(str_idx)) { return; }
  auto const d_str   = d_strings.element<string_view>(str_idx);
  auto const str_ptr = d_str.data();

  // each thread processes 4 bytes
  size_type size = 0;
  for (auto i = lane_idx * 4; i < d_str.size_bytes(); i += cudf::detail::warp_size * 4) {
    for (auto j = i; (j < (i + 4)) && (j < d_str.size_bytes()); j++) {
      auto const chr = str_ptr[j];
      if (is_utf8_continuation_char(chr)) { continue; }
      char_utf8 u8 = 0;
      to_char_utf8(str_ptr + j, u8);
      size += converter.process_character(u8);
    }
  }
  // this is slightly faster than using the cub::warp_reduce
  if (size > 0) {
    cuda::atomic_ref<size_type, cuda::thread_scope_block> ref{*(d_sizes + str_idx)};
    ref.fetch_add(size, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Special functor for processing ASCII-only data
 */
struct ascii_converter_fn {
  convert_char_fn converter;
  __device__ char operator()(char chr) { return converter.process_ascii(chr); }
};

constexpr int64_t block_size       = 512;
constexpr int64_t bytes_per_thread = 8;

/**
 * @brief Checks the chars data for any multibyte characters
 *
 * The output count is not accurate but it is only checked for > 0.
 */
CUDF_KERNEL void has_multibytes_kernel(char const* d_input_chars,
                                       int64_t first_offset,
                                       int64_t last_offset,
                                       int64_t* d_output)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  // read only every 2nd byte; all bytes in a multibyte char have high bit set
  auto const byte_idx = (static_cast<int64_t>(idx) * bytes_per_thread) + first_offset;
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  using block_reduce = cub::BlockReduce<int64_t, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  // each thread processes 8 bytes (only 4 need to be checked)
  int64_t mb_count = 0;
  for (auto i = byte_idx; (i < (byte_idx + bytes_per_thread)) && (i < last_offset); i += 2) {
    u_char const chr = static_cast<u_char>(d_input_chars[i]);
    mb_count += ((chr & 0x80) > 0);
  }
  auto const mb_total = block_reduce(temp_storage).Reduce(mb_count, cub::Sum());

  if ((lane_idx == 0) && (mb_total > 0)) {
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref{*d_output};
    ref.fetch_add(mb_total, cuda::std::memory_order_relaxed);
  }
}

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
                                     rmm::device_async_resource_ref mr)
{
  if (input.size() == input.null_count()) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_flags   = get_character_flags_table();
  auto const d_cases   = get_character_cases_table();
  auto const d_special = get_special_case_mapping_table();

  auto const first_offset = (input.offset() == 0) ? 0L
                                                  : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset =
    cudf::strings::detail::get_offset_value(input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size  = last_offset - first_offset;
  auto const input_chars = input.chars_begin(stream);

  convert_char_fn ccfn{case_flag, d_flags, d_cases, d_special};
  upper_lower_fn converter{ccfn, *d_strings};

  // For smaller strings, use the regular string-parallel algorithm
  if ((chars_size / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    auto [offsets, chars] = make_strings_children(converter, input.size(), stream, mr);
    return make_strings_column(input.size(),
                               std::move(offsets),
                               chars.release(),
                               input.null_count(),
                               cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  // Check if the input contains any multi-byte characters.
  // This check incurs ~20% performance hit for smaller strings and so we only use it
  // after the threshold check above. The check makes very little impact for long strings
  // but results in a large performance gain when the input contains only single-byte characters.
  cudf::detail::device_scalar<int64_t> mb_count(0, stream);
  // cudf::detail::grid_1d is limited to size_type elements
  auto const num_blocks = util::div_rounding_up_safe(chars_size / bytes_per_thread, block_size);
  // we only need to check every other byte since either will contain high bit
  has_multibytes_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
    input_chars, first_offset, last_offset, mb_count.data());
  if (mb_count.value(stream) == 0) {
    // optimization for ASCII-only case: copy the input column and inplace replace each character
    auto result  = std::make_unique<column>(input.parent(), stream, mr);
    auto d_chars = result->mutable_view().head<char>();
    thrust::transform(
      rmm::exec_policy(stream), d_chars, d_chars + chars_size, d_chars, ascii_converter_fn{ccfn});
    result->set_null_count(input.null_count());
    return result;
  }

  // This will use a warp-parallel algorithm to compute the output sizes for each string
  // note: tried to use segmented-reduce approach instead here and it was consistently slower
  auto [offsets, bytes] = [&] {
    rmm::device_uvector<size_type> sizes(input.size(), stream);
    // cudf::detail::grid_1d is limited to size_type threads
    auto const num_blocks = util::div_rounding_up_safe(
      static_cast<int64_t>(input.size()) * cudf::detail::warp_size, block_size);
    count_bytes_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
      ccfn, *d_strings, sizes.data());
    // convert sizes to offsets
    return cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  }();

  // build sub-offsets
  auto const sub_count = chars_size / LS_SUB_BLOCK_SIZE;
  auto tmp_offsets     = rmm::device_uvector<int64_t>(sub_count + input.size() + 1, stream);
  {
    rmm::device_uvector<int64_t> sub_offsets(sub_count, stream);
    auto const count_itr = thrust::make_counting_iterator<int64_t>(0);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      count_itr,
                      count_itr + sub_count,
                      sub_offsets.data(),
                      sub_offset_fn{input_chars, first_offset, last_offset});

    // merge them with input offsets
    auto input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    thrust::merge(rmm::exec_policy_nosync(stream),
                  input_offsets,
                  input_offsets + input.size() + 1,
                  sub_offsets.begin(),
                  sub_offsets.end(),
                  tmp_offsets.begin());
    stream.synchronize();  // protect against destruction of sub_offsets
  }

  // run case conversion over the new sub-strings
  auto const tmp_size = static_cast<size_type>(tmp_offsets.size()) - 1;
  upper_lower_ls_fn sub_conv{ccfn, input_chars, tmp_offsets.data()};
  auto chars = std::get<1>(make_strings_children(sub_conv, tmp_size, stream, mr));

  return make_strings_column(input.size(),
                             std::move(offsets),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> to_lower(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  character_flags_table_type case_flag = IS_UPPER(0xFF);  // convert only upper case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> to_upper(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  character_flags_table_type case_flag = IS_LOWER(0xFF);  // convert only lower case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> swapcase(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  // convert only upper or lower case characters
  character_flags_table_type case_flag = IS_LOWER(0xFF) | IS_UPPER(0xFF);
  return convert_case(strings, case_flag, stream, mr);
}

}  // namespace detail

// APIs

std::unique_ptr<column> to_lower(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_lower(strings, stream, mr);
}

std::unique_ptr<column> to_upper(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_upper(strings, stream, mr);
}

std::unique_ptr<column> swapcase(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::swapcase(strings, stream, mr);
}

}  // namespace strings
}  // namespace cudf
