/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/detail/get_value.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <cstring>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param input Data to copy from.
 * @param bytes Number of bytes to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_and_increment(char* buffer, const char* input, size_type bytes)
{
  std::memcpy(buffer, input, bytes);
  return buffer + bytes;
}

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param d_string String to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_string(char* buffer, const string_view& d_string)
{
  return copy_and_increment(buffer, d_string.data(), d_string.size_bytes());
}

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output.
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must also have members d_offsets and d_chars which are set to
 *         memory containing the offsets and chars columns during write.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string.
 *        After that, the d_offsets and d_chars are set and this is called again to fill in the
 *        chars memory.
 * @param strings_count Number of strings.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return offsets child column and chars child column for a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(
  SizeAndExecuteFunction size_and_exec_fn,
  size_type strings_count,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view          = offsets_column->mutable_view();
  auto d_offsets             = offsets_view.template data<int32_t>();
  size_and_exec_fn.d_offsets = d_offsets;

  // This is called twice -- once for offsets and once for chars.
  // Reducing the number of places size_and_exec_fn is inlined speeds up compile time.
  auto for_each_fn = [strings_count, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       size_and_exec_fn);
  };

  // Compute the offsets values
  for_each_fn(size_and_exec_fn);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // Now build the chars column
  auto const bytes = cudf::detail::get_value<int32_t>(offsets_view, strings_count, stream);
  std::unique_ptr<column> chars_column =
    create_chars_child_column(strings_count, bytes, stream, mr);
  size_and_exec_fn.d_chars = chars_column->mutable_view().template data<char>();
  for_each_fn(size_and_exec_fn);

  return std::make_pair(std::move(offsets_column), std::move(chars_column));
}

/**
 * @brief Creates child offsets, chars columns and null mask, null count of a strings column by
 * applying the template function that can be used for computing the output size of each string as
 * well as create the output.
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must have members d_offsets and d_chars which are set to memory containing
 *         the offsets and chars columns during write. In addition, it must also output negative
 *         index values to the d_offsets array to specify that the corresponding string rows are
 *         null elements.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string.
 *        After that, the d_offsets and d_chars are set and this is called again to fill in the
 *        chars memory.
 * @param strings_count Number of strings.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return offsets child column and chars child column for a strings column
 */
template <typename SizeAndExecuteFunction>
std::tuple<std::unique_ptr<column>, std::unique_ptr<column>, rmm::device_buffer, size_type>
make_strings_children_with_null_mask(
  SizeAndExecuteFunction size_and_exec_fn,
  size_type strings_count,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view          = offsets_column->mutable_view();
  auto d_offsets             = offsets_view.template data<int32_t>();
  size_and_exec_fn.d_offsets = d_offsets;

  // This is called twice -- once for offsets and once for chars.
  auto for_each_fn = [strings_count, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       size_and_exec_fn);
  };

  // Compute the string sizes, storing in d_offsets (negative output sizes mean null strings)
  for_each_fn(size_and_exec_fn);

  // Use the string sizes to compute null_mask and null_count of the output strings column
  auto [null_mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_offsets] __device__(auto const idx) { return d_offsets[idx] >= 0; },
    stream,
    mr);

  // Compute the offsets
  auto const iter_trans_begin = thrust::make_transform_iterator(
    d_offsets, [] __device__(auto const size) { return size < 0 ? 0 : size; });
  thrust::exclusive_scan(
    rmm::exec_policy(stream), iter_trans_begin, iter_trans_begin + strings_count + 1, d_offsets);

  // Now build the chars column
  auto const bytes = cudf::detail::get_value<int32_t>(offsets_view, strings_count, stream);
  std::unique_ptr<column> chars_column =
    create_chars_child_column(strings_count, bytes, stream, mr);
  size_and_exec_fn.d_chars = chars_column->mutable_view().template data<char>();

  // If all the strings are empty or null, the d_chars pointer will has nullptr value.
  // Thus, we need to set an arbitrary pointer value to d_chars to prevent the string sizes to be
  // computed again. It is safe to do so, because in this case the string column has all empty, or
  // all null string elements so nothing will be copied onto d_chars.
  if (!size_and_exec_fn.d_chars) { size_and_exec_fn.d_chars = reinterpret_cast<char*>(0x1); }
  for_each_fn(size_and_exec_fn);

  return std::make_tuple(
    std::move(offsets_column), std::move(chars_column), std::move(null_mask), null_count);
}

/**
 * @brief Converts a single UTF-8 character into a code-point value that
 * can be used for lookup in the character flags or the character case tables.
 *
 * @param utf8_char Single UTF-8 character to convert.
 * @return Code-point for the UTF-8 character.
 */
__device__ inline uint32_t utf8_to_codepoint(cudf::char_utf8 utf8_char)
{
  uint32_t unchr = 0;
  if (utf8_char < 0x00000080)  // single-byte pass thru
    unchr = utf8_char;
  else if (utf8_char < 0x0000E000)  // two bytes
  {
    unchr = (utf8_char & 0x1F00) >> 2;  // shift and
    unchr |= (utf8_char & 0x003F);      // unmask
  } else if (utf8_char < 0x00F00000)    // three bytes
  {
    unchr = (utf8_char & 0x0F0000) >> 4;         // get upper 4 bits
    unchr |= (utf8_char & 0x003F00) >> 2;        // shift and
    unchr |= (utf8_char & 0x00003F);             // unmask
  } else if (utf8_char <= (unsigned)0xF8000000)  // four bytes
  {
    unchr = (utf8_char & 0x03000000) >> 6;   // upper 3 bits
    unchr |= (utf8_char & 0x003F0000) >> 4;  // next 6 bits
    unchr |= (utf8_char & 0x00003F00) >> 2;  // next 6 bits
    unchr |= (utf8_char & 0x0000003F);       // unmask
  }
  return unchr;
}

/**
 * @brief Converts a character code-point value into a UTF-8 character.
 *
 * @param unchr Character code-point to convert.
 * @return Single UTF-8 character.
 */
__host__ __device__ inline cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
{
  cudf::char_utf8 utf8 = 0;
  if (unchr < 0x00000080)  // single byte utf8
    utf8 = unchr;
  else if (unchr < 0x00000800)  // double byte utf8
  {
    utf8 = (unchr << 2) & 0x1F00;  // shift bits for
    utf8 |= (unchr & 0x3F);        // utf8 encoding
    utf8 |= 0x0000C080;
  } else if (unchr < 0x00010000)  // triple byte utf8
  {
    utf8 = (unchr << 4) & 0x0F0000;   // upper 4 bits
    utf8 |= (unchr << 2) & 0x003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);           // last 6 bits
    utf8 |= 0x00E08080;
  } else if (unchr < 0x00110000)  // quadruple byte utf8
  {
    utf8 = (unchr << 6) & 0x07000000;   // upper 3 bits
    utf8 |= (unchr << 4) & 0x003F0000;  // next 6 bits
    utf8 |= (unchr << 2) & 0x00003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);             // last 6 bits
    utf8 |= (unsigned)0xF0808080;
  }
  return utf8;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
