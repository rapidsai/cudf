/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <text/subword/detail/cp_data.h>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/error.hpp>
#include <text/subword/detail/codepoint_metadata.ah>
#include <text/subword/detail/data_normalizer.hpp>
#include <text/subword/detail/tokenizer_utils.cuh>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <string>
#include <vector>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Bit used to filter out invalid code points.
 *
 * When normalizing characters to code point values, if this bit is set,
 * the code point should be filtered out before returning from the normalizer.
 */
constexpr uint32_t FILTER_BIT = 22;

/**
 * @brief Retrieve new code point from metadata value.
 *
 * @param metadata Value from the codepoint_metadata table.
 * @return The replacement character if appropriate.
 */
__device__ uint32_t get_first_cp(uint32_t metadata) { return metadata & NEW_CP_MASK; }

/**
 * @brief Retrieve token category from the metadata value.
 *
 * Category values are 0-5:
 * 0 - character should be padded
 * 1 - pad character if lower-case
 * 2 - character should be removed
 * 3 - remove character if lower-case
 * 4 - whitespace character -- always replace
 * 5 - uncategorized
 *
 * @param metadata Value from the codepoint_metadata table.
 * @return Category value.
 */
__device__ uint32_t extract_token_cat(uint32_t metadata)
{
  return (metadata >> TOKEN_CAT_SHIFT) & TOKEN_CAT_MASK;
}

/**
 * @brief Return true if category of metadata value specifies the character should be replaced.
 */
__device__ bool should_remove_cp(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_REMOVE_CHAR) || (lower_case && (cat == TOKEN_CAT_REMOVE_CHAR_IF_LOWER));
}

/**
 * @brief Return true if category of metadata value specifies the character should be padded.
 */
__device__ bool should_add_spaces(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_ADD_SPACE) || (lower_case && (cat == TOKEN_CAT_ADD_SPACE_IF_LOWER));
}

/**
 * @brief Return true if category of metadata value specifies the character should be replaced.
 */
__device__ bool always_replace(uint32_t metadata)
{
  return extract_token_cat(metadata) == TOKEN_CAT_ALWAYS_REPLACE;
}

/**
 * @brief Returns true if metadata value includes a multi-character transform bit equal to 1.
 */
__device__ bool is_multi_char_transform(uint32_t metadata)
{
  return (metadata >> MULTICHAR_SHIFT) & MULTICHAR_MASK;
}

/**
 * @brief Returns true if the byte passed in could be a valid head byte for
 * a utf8 character. That is, not binary `10xxxxxx`
 */
__device__ bool is_head_byte(unsigned char utf8_byte) { return (utf8_byte >> 6) != 2; }

/**
 * @brief Converts a UTF-8 character into a unicode code point value.
 *
 * If the byte at start_byte_for_thread is the first byte of a UTF-8 character (head byte),
 * the UTF-8 character is converted to a unicode code point and returned.
 *
 * If the byte at start_byte_for_thread is not a head byte, 0 is returned.
 *
 * All threads start reading bytes from the pointer denoted by strings.
 *
 * @param strings A pointer to the start of the sequence of characters to be analyzed.
 * @param start_byte_for_thread Which byte to start analyzing
 * @return New code point value for this byte.
 */
__device__ uint32_t extract_code_points_from_utf8(unsigned char const* strings,
                                                  size_t const total_bytes,
                                                  uint32_t const start_byte_for_thread)
{
  constexpr uint8_t max_utf8_blocks_for_char    = 4;
  uint8_t utf8_blocks[max_utf8_blocks_for_char] = {0};

#pragma unroll
  for (int i = 0; i < std::min(static_cast<size_t>(max_utf8_blocks_for_char),
                               total_bytes - start_byte_for_thread);
       ++i) {
    utf8_blocks[i] = strings[start_byte_for_thread + i];
  }

  const uint8_t length_encoding_bits = utf8_blocks[0] >> 3;
  // UTF-8 format is variable-width character encoding using up to 4 bytes.
  // If the first byte is:
  // - [x00-x7F] -- beginning of a 1-byte character (ASCII)
  // - [xC0-xDF] -- beginning of a 2-byte character
  // - [xE0-xEF] -- beginning of a 3-byte character
  // - [xF0-xF7] -- beginning of a 3-byte character
  // Anything else is an intermediate byte [x80-xBF].
  // So shifted by 3 bits this becomes
  // - [x00-x0F]  or leb < 16
  // - [x18-x1B]  or 24 <= leb <= 27
  // - [x1C-x1D]  or 28 <= leb <= 29
  // - [x1E-x1F]  or leb >= 30
  // The remaining bits are part of the value as specified by the mask
  // specified by x's below.
  // - b0xxxxxxx = x7F
  // - b110xxxxx = x1F
  // - b1110xxxx = x0F
  // - b11110xxx = x07
  using encoding_length_pair = thrust::pair<uint8_t, uint8_t>;
  // Set the number of characters and the top masks based on the length encoding bits.
  encoding_length_pair const char_encoding_length = [length_encoding_bits] {
    if (length_encoding_bits < 16) return encoding_length_pair{1, 0x7F};
    if (length_encoding_bits >= 24 && length_encoding_bits <= 27)
      return encoding_length_pair{2, 0x1F};
    if (length_encoding_bits == 28 || length_encoding_bits == 29)
      return encoding_length_pair{3, 0x0F};
    if (length_encoding_bits == 30) return encoding_length_pair{4, 0x07};
    return encoding_length_pair{0, 0};
  }();

  // Now pack up the bits into a uint32_t.
  // Move the first set of values into bits 19-24 in the 32-bit value.
  uint32_t code_point = (utf8_blocks[0] & char_encoding_length.second) << 18;
  // Move the remaining values which are 6 bits (mask b10xxxxxx = x3F)
  // from the remaining bytes into successive positions in the 32-bit result.
#pragma unroll
  for (int i = 1; i < max_utf8_blocks_for_char; ++i) {
    code_point |= ((utf8_blocks[i] & 0x3F) << (18 - 6 * i));
  }

  // Adjust the final result by shifting by the character length.
  uint8_t const shift_amt = 24 - 6 * char_encoding_length.first;
  code_point >>= shift_amt;
  return code_point;
}

/**
 * @brief Normalize the characters for the strings input.
 *
 * Characters are replaced, padded, or removed depending on the `do_lower_case` input
 * as well as the metadata values for each code point found in `cp_metadata`.
 *
 * First, each character is converted from UTF-8 to a unicode code point value.
 * This value is then looked up in the `cp_metadata` table to determine its fate.
 * The end result is a set of code point values for each character.
 * The normalized set of characters make it easier for the tokenizer to identify
 * tokens and match up token ids.
 *
 * @param[in] strings The input strings with characters to normalize to code point values.
 * @param[in] device_strings_offsets Offsets to individual strings in the input vector.
 * @param[in] total_bytes Total number of bytes in the input `strings` vector.
 * @param[in] cp_metadata The metadata lookup table for every unicode code point value.
 * @param[in] aux_table Aux table for mapping some multi-byte code point values.
 * @param[in] do_lower_case True if normalization should include lower-casing.
 * @param[in] num_strings Total number of strings in the input `strings` vector.
 * @param[out] code_points The resulting code point values from normalization.
 * @param[out] chars_per_thread Output number of code point values per string.
 */
__global__ void kernel_data_normalizer(unsigned char const* strings,
                                       uint32_t const* device_strings_offsets,
                                       size_t const total_bytes,
                                       uint32_t const* cp_metadata,
                                       uint64_t const* aux_table,
                                       bool const do_lower_case,
                                       uint32_t const num_strings,
                                       uint32_t* code_points,
                                       uint32_t* chars_per_thread)
{
  constexpr uint32_t init_val                     = (1 << FILTER_BIT);
  uint32_t replacement_code_points[MAX_NEW_CHARS] = {init_val, init_val, init_val};

  uint32_t const char_for_thread = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t num_new_chars         = 0;

  if (char_for_thread < total_bytes) {
    auto const code_point = extract_code_points_from_utf8(strings, total_bytes, char_for_thread);
    auto const metadata   = cp_metadata[code_point];

    if (is_head_byte(strings[char_for_thread]) && !should_remove_cp(metadata, do_lower_case)) {
      num_new_chars = 1;
      // Apply lower cases and accent stripping if necessary
      auto const new_cp =
        do_lower_case || always_replace(metadata) ? get_first_cp(metadata) : code_point;
      replacement_code_points[0] = new_cp == 0 ? code_point : new_cp;

      if (do_lower_case && is_multi_char_transform(metadata)) {
        auto const next_cps          = aux_table[code_point];
        replacement_code_points[1]   = static_cast<uint32_t>(next_cps >> 32);
        auto const potential_next_cp = static_cast<uint32_t>(next_cps);
        replacement_code_points[2] =
          potential_next_cp != 0 ? potential_next_cp : replacement_code_points[2];
        num_new_chars = 2 + (potential_next_cp != 0);
      }

      if (should_add_spaces(metadata, do_lower_case)) {
        // Need to shift all existing code-points up one
        // This is a rotate right. There is no thrust equivalent at this time.
        for (int loc = num_new_chars; loc > 0; --loc) {
          replacement_code_points[loc] = replacement_code_points[loc - 1];
        }

        // Write the required spaces at the end
        replacement_code_points[0]                 = SPACE_CODE_POINT;
        replacement_code_points[num_new_chars + 1] = SPACE_CODE_POINT;
        num_new_chars += 2;
      }
    }
  }

  chars_per_thread[char_for_thread] = num_new_chars;

  typedef cub::
    BlockStore<uint32_t, THREADS_PER_BLOCK, MAX_NEW_CHARS, cub::BLOCK_STORE_WARP_TRANSPOSE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage temp_storage;

  // Now we perform coalesced writes back to global memory using cub.
  uint32_t* block_base = code_points + blockIdx.x * blockDim.x * MAX_NEW_CHARS;
  BlockStore(temp_storage).Store(block_base, replacement_code_points);
}

/**
 * @brief Retrieve the code point metadata table.
 *
 * Build the code point metadata table in device memory
 * using the vector pieces from codepoint_metadata.ah
 */
const codepoint_metadata_type* get_codepoint_metadata(cudaStream_t stream)
{
  static cudf::strings::detail::thread_safe_per_context_cache<codepoint_metadata_type>
    g_codepoint_metadata;
  return g_codepoint_metadata.find_or_initialize([stream](void) {
    codepoint_metadata_type* table =
      static_cast<codepoint_metadata_type*>(rmm::mr::get_default_resource()->allocate(
        codepoint_metadata_size * sizeof(codepoint_metadata_type), stream));
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 table + cp_section1_end,
                 table + codepoint_metadata_size,
                 codepoint_metadata_default_value);
    CUDA_TRY(cudaMemcpyAsync(table,
                             codepoint_metadata,
                             cp_section1_end * sizeof(codepoint_metadata[0]),  // 1st section
                             cudaMemcpyHostToDevice,
                             stream));
    CUDA_TRY(cudaMemcpyAsync(
      table + cp_section2_begin,
      cp_metadata_917505_917999,
      (cp_section2_end - cp_section2_begin + 1) * sizeof(codepoint_metadata[0]),  // 2nd section
      cudaMemcpyHostToDevice,
      stream));
    return table;
  });
}

/**
 * @brief Retrieve the aux code point data table.
 *
 * Build the aux code point data table in device memory
 * using the vector pieces from codepoint_metadata.ah
 */
const aux_codepoint_data_type* get_aux_codepoint_data(cudaStream_t stream)
{
  static cudf::strings::detail::thread_safe_per_context_cache<aux_codepoint_data_type>
    g_aux_codepoint_data;
  return g_aux_codepoint_data.find_or_initialize([stream](void) {
    aux_codepoint_data_type* table =
      static_cast<aux_codepoint_data_type*>(rmm::mr::get_default_resource()->allocate(
        aux_codepoint_data_size * sizeof(aux_codepoint_data_type), stream));
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 table + aux_section1_end,
                 table + aux_codepoint_data_size,
                 aux_codepoint_default_value);
    CUDA_TRY(cudaMemcpyAsync(table,
                             aux_codepoint_data,
                             aux_section1_end * sizeof(aux_codepoint_data[0]),  // 1st section
                             cudaMemcpyHostToDevice,
                             stream));
    CUDA_TRY(cudaMemcpyAsync(
      table + aux_section2_begin,
      aux_cp_data_44032_55203,
      (aux_section2_end - aux_section2_begin + 1) * sizeof(aux_codepoint_data[0]),  // 2nd section
      cudaMemcpyHostToDevice,
      stream));
    CUDA_TRY(cudaMemcpyAsync(
      table + aux_section3_begin,
      aux_cp_data_70475_71099,
      (aux_section3_end - aux_section3_begin + 1) * sizeof(aux_codepoint_data[0]),  // 3rd section
      cudaMemcpyHostToDevice,
      stream));
    CUDA_TRY(cudaMemcpyAsync(
      table + aux_section4_begin,
      aux_cp_data_119134_119232,
      (aux_section4_end - aux_section4_begin + 1) * sizeof(aux_codepoint_data[0]),  // 4th section
      cudaMemcpyHostToDevice,
      stream));
    return table;
  });
}

}  // namespace

data_normalizer::data_normalizer(uint32_t max_num_strings,
                                 uint32_t max_num_chars,
                                 cudaStream_t stream,
                                 bool do_lower_case)
  : do_lower_case(do_lower_case),
    device_strings(static_cast<size_t>(max_num_chars), stream),
    device_strings_offsets(static_cast<size_t>(max_num_strings + 1), stream),
    device_code_points(0, stream),
    device_chars_per_thread(0, stream),
    cub_temp_storage(0, stream),
    device_num_selected(1, stream)
{
  d_cp_metadata = detail::get_codepoint_metadata(stream);
  d_aux_table   = detail::get_aux_codepoint_data(stream);

  cudf::detail::grid_1d const grid{static_cast<cudf::size_type>(max_num_chars), THREADS_PER_BLOCK};
  size_t const max_threads_on_device = grid.num_threads_per_block * grid.num_blocks;
  size_t const max_new_char_total    = MAX_NEW_CHARS * max_threads_on_device;
  device_code_points.resize(max_new_char_total, stream);
  device_chars_per_thread.resize(max_threads_on_device, stream);

  // Determine temporary device storage requirements for cub
  size_t temp_storage_scan_bytes    = 0;
  uint32_t* device_chars_per_thread = nullptr;
  cub::DeviceScan::InclusiveSum(nullptr,
                                temp_storage_scan_bytes,
                                device_chars_per_thread,
                                device_chars_per_thread,
                                max_threads_on_device,
                                stream);
  size_t temp_storage_select_bytes = 0;
  NotEqual const select_op((1 << FILTER_BIT));
  cub::DeviceSelect::If(nullptr,
                        temp_storage_select_bytes,
                        device_code_points.data(),
                        device_code_points.data(),
                        device_num_selected.data(),
                        max_new_char_total,
                        select_op,
                        stream);
  max_cub_storage_bytes = std::max(temp_storage_scan_bytes, temp_storage_select_bytes);
  cub_temp_storage.resize(max_cub_storage_bytes, stream);
}

std::pair<ptr_length_pair, ptr_length_pair> data_normalizer::normalize(char const* d_strings,
                                                                       uint32_t const* d_offsets,
                                                                       uint32_t num_strings,
                                                                       cudaStream_t stream)
{
  ptr_length_pair cp_and_length;
  ptr_length_pair offset_and_length;

  // copy offsets to working memory
  size_t const num_offsets = std::min(size_t{num_strings + 1}, device_strings_offsets.size());
  CUDA_TRY(cudaMemcpyAsync(device_strings_offsets.data(),
                           d_offsets,
                           sizeof(uint32_t) * num_offsets,
                           cudaMemcpyDeviceToDevice,
                           stream));
  uint32_t const bytes_count = device_strings_offsets.element(num_offsets - 1, stream);
  if (bytes_count == 0)  // if no bytes, nothing to do
    return std::make_pair(cp_and_length, offset_and_length);

  cudf::detail::grid_1d const grid{static_cast<cudf::size_type>(bytes_count), THREADS_PER_BLOCK, 1};
  size_t const threads_on_device  = grid.num_threads_per_block * grid.num_blocks;
  size_t const max_new_char_total = MAX_NEW_CHARS * threads_on_device;

  kernel_data_normalizer<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
    reinterpret_cast<const unsigned char*>(d_strings),
    device_strings_offsets.data(),
    bytes_count,
    d_cp_metadata,
    d_aux_table,
    do_lower_case,
    num_strings,
    device_code_points.data(),
    device_chars_per_thread.data());
  CHECK_CUDA(stream);

  NotEqual const select_op((1 << FILTER_BIT));
  cub::DeviceSelect::If(cub_temp_storage.data(),
                        max_cub_storage_bytes,
                        device_code_points.data(),
                        device_code_points.data(),
                        device_num_selected.data(),
                        max_new_char_total,
                        select_op,
                        stream);
  CHECK_CUDA(stream);

  // We also need to prefix sum the number of characters up to an including the current character in
  // order to get the new strings lengths.
  cub::DeviceScan::InclusiveSum(cub_temp_storage.data(),
                                max_cub_storage_bytes,
                                device_chars_per_thread.data(),
                                device_chars_per_thread.data(),
                                threads_on_device,
                                stream);
  CHECK_CUDA(stream);

  // this is like an in-place gather
  auto const execpol = rmm::exec_policy(stream);
  thrust::for_each_n(
    execpol->on(stream),
    thrust::make_counting_iterator<uint32_t>(1),
    num_strings,
    update_strings_lengths_fn{device_chars_per_thread.data(), device_strings_offsets.data()});

  offset_and_length.gpu_ptr = device_strings_offsets.data();
  offset_and_length.length  = num_strings + 1;

  uint32_t num_chars = 0;
  CUDA_TRY(cudaMemcpyAsync(&num_chars,
                           offset_and_length.gpu_ptr + num_strings,
                           sizeof(num_chars),
                           cudaMemcpyDeviceToHost,
                           stream));
  cp_and_length.gpu_ptr = device_code_points.data();
  cp_and_length.length  = num_chars;

  return std::make_pair(cp_and_length, offset_and_length);
}

}  // namespace detail
}  // namespace nvtext
