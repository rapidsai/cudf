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
#include <cudf/utilities/error.hpp>
#include <text/subword/detail/codepoint_metadata.ah>
#include <text/subword/detail/data_normalizer.hpp>
#include <text/subword/detail/tokenizer_utils.cuh>

#include <device_launch_parameters.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <string>
#include <vector>

namespace nvtext {
namespace detail {
namespace {

#define SORT_BIT 22
#define THREADS_PER_BLOCK 64

__device__ __forceinline__ uint32_t get_first_cp(uint32_t metadata)
{
  return metadata & NEW_CP_MASK;
}

__device__ __forceinline__ uint32_t extract_token_cat(uint32_t metadata)
{
  return (metadata >> TOKEN_CAT_SHIFT) & TOKEN_CAT_MASK;
}

__device__ __forceinline__ bool should_remove_cp(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_REMOVE_CHAR) || (lower_case && (cat == TOKEN_CAT_REMOVE_CHAR_IF_LOWER));
}

__device__ __forceinline__ bool should_add_spaces(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_ADD_SPACE) || (lower_case && (cat == TOKEN_CAT_ADD_SPACE_IF_LOWER));
}

__device__ __forceinline__ bool always_replace(uint32_t metadata)
{
  return extract_token_cat(metadata) == TOKEN_CAT_ALWAYS_REPLACE;
}

__device__ __forceinline__ uint32_t is_multi_char_transform(uint32_t metadata)
{
  return (metadata >> MULTICHAR_SHIFT) & MULTICHAR_MASK;
}

/**
 * @brief Returns true if the byte passed in could be a valid head byte for
 * a utf8 character. That is, not binary `10xxxxxx`
 */
__device__ __forceinline__ bool is_head_byte(unsigned char utf8_byte)
{
  return (utf8_byte >> 6) != 2;
}

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
__device__ __forceinline__ uint32_t
extract_code_points_from_utf8(const unsigned char* strings, const uint32_t start_byte_for_thread)
{
  constexpr uint8_t max_utf8_blocks_for_char = 4;
  uint8_t utf8_blocks[max_utf8_blocks_for_char];

#pragma unroll
  for (int i = 0; i < max_utf8_blocks_for_char; ++i) {
    utf8_blocks[i] = strings[start_byte_for_thread + i];
  }

  // We can have at most 5 bits encoding the length. We check those bits to infer the actual length
  const uint8_t length_encoding_bits = utf8_blocks[0] >> 3;

  // Set the number of characters and the top masks based on the
  // length encoding bits.
  uint8_t char_encoding_length = 0, top_mask = 0;
  if (length_encoding_bits < 16) {
    char_encoding_length = 1;
    top_mask             = 0x7F;
  } else if (length_encoding_bits >= 24 && length_encoding_bits <= 27) {
    char_encoding_length = 2;
    top_mask             = 0x1F;
  } else if (length_encoding_bits == 28 || length_encoding_bits == 29) {
    char_encoding_length = 3;
    top_mask             = 0x0F;
  } else if (length_encoding_bits == 30) {
    char_encoding_length = 4;
    top_mask             = 0x07;
  }

  // Now pack up the bits into a uint32_t. All threads will process 4 bytes
  // to reduce divergence.
  uint32_t code_point = (utf8_blocks[0] & top_mask) << 18;

#pragma unroll
  for (int i = 1; i < max_utf8_blocks_for_char; ++i) {
    code_point |= ((utf8_blocks[i] & 0x3F) << (18 - 6 * i));
  }

  // Zero out the bottom of code points with extra reads
  const uint8_t shift_amt = 24 - 6 * char_encoding_length;
  code_point >>= shift_amt;

  return code_point;
}

}  // namespace

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
  constexpr uint32_t init_val                     = (1 << SORT_BIT);
  uint32_t replacement_code_points[MAX_NEW_CHARS] = {init_val, init_val, init_val};

  const uint32_t char_for_thread = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t num_new_chars         = 0;

  if (char_for_thread < total_bytes) {
    auto const code_point = extract_code_points_from_utf8(strings, char_for_thread);
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
        // TODO: see if there is algorithm for this
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

data_normalizer::data_normalizer(uint32_t max_num_strings,
                                 uint32_t max_num_chars,
                                 bool do_lower_case,
                                 cudaStream_t stream)
  : do_lower_case(do_lower_case),
    device_strings_offsets(max_num_strings + 1),
    device_strings(max_num_chars),
    device_cp_metadata(codepoint_metadata_size),
    device_aux_table(aux_codepoint_data_size)
{
  // build the code point metadata tables in device memory
  // using the vector pieces from codepoint_metadata.ah
  auto execpol = rmm::exec_policy(stream);
  thrust::fill(execpol->on(stream),
               device_cp_metadata.begin() + cp_section1_end,
               device_cp_metadata.end(),
               codepoint_metadata_default_value);
  CUDA_TRY(cudaMemcpyAsync(device_cp_metadata.data().get(),
                           codepoint_metadata,
                           cp_section1_end * sizeof(codepoint_metadata[0]),  // 1st section
                           cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(cudaMemcpyAsync(
    device_cp_metadata.data().get() + cp_section2_begin,
    cp_metadata_917505_917999,
    (cp_section2_end - cp_section2_begin + 1) * sizeof(codepoint_metadata[0]),  // 2nd section
    cudaMemcpyHostToDevice,
    stream));

  thrust::fill(execpol->on(stream),
               device_aux_table.begin() + aux_section1_end,
               device_aux_table.end(),
               aux_codepoint_default_value);
  CUDA_TRY(cudaMemcpyAsync(device_aux_table.data().get(),
                           aux_codepoint_data,
                           aux_section1_end * sizeof(aux_codepoint_data[0]),  // 1st section
                           cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(cudaMemcpyAsync(
    device_aux_table.data().get() + aux_section2_begin,
    aux_cp_data_44032_55203,
    (aux_section2_end - aux_section2_begin + 1) * sizeof(aux_codepoint_data[0]),  // 2nd section
    cudaMemcpyHostToDevice,
    stream));
  CUDA_TRY(cudaMemcpyAsync(
    device_aux_table.data().get() + aux_section3_begin,
    aux_cp_data_70475_71099,
    (aux_section3_end - aux_section3_begin + 1) * sizeof(aux_codepoint_data[0]),  // 3rd section
    cudaMemcpyHostToDevice,
    stream));
  CUDA_TRY(cudaMemcpyAsync(
    device_aux_table.data().get() + aux_section4_begin,
    aux_cp_data_119134_119232,
    (aux_section4_end - aux_section4_begin + 1) * sizeof(aux_codepoint_data[0]),  // 4th section
    cudaMemcpyHostToDevice,
    stream));

  size_t max_BLOCKS               = (max_num_chars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  size_t max_threads_on_device    = max_BLOCKS * THREADS_PER_BLOCK;
  const size_t max_new_char_total = MAX_NEW_CHARS * max_threads_on_device;
  device_code_points.resize(max_new_char_total);
  device_chars_per_thread.resize(max_threads_on_device);

  // TODO: using thrust instead would alleviate this extra processing
  // Determine temporary device storage requirements for cub
  //  size_t temp_storage_scan_bytes    = 0;
  //  uint32_t* device_chars_per_thread = nullptr;
  //  cub::DeviceScan::InclusiveSum(nullptr,
  //                                temp_storage_scan_bytes,
  //                                device_chars_per_thread,
  //                                device_chars_per_thread,
  //                                max_threads_on_device,
  //                                stream);
  //  size_t temp_storage_select_bytes = 0;
  //  static NotEqual select_op((1 << SORT_BIT));
  //  cub::DeviceSelect::If(nullptr,
  //                        temp_storage_select_bytes,
  //                        device_code_points.data().get(),
  //                        device_code_points.data().get(),
  //                        device_num_selected.data().get(),
  //                        max_new_char_total,
  //                        select_op,
  //                        stream);
  //  max_cub_storage_bytes = std::max(temp_storage_scan_bytes, temp_storage_select_bytes);
  //  cub_temp_storage.resize(max_cub_storage_bytes);
  //  device_num_selected.resize(1);
}

std::pair<ptr_length_pair, ptr_length_pair> data_normalizer::normalize(const char* d_strings,
                                                                       const uint32_t* d_offsets,
                                                                       uint32_t num_strings,
                                                                       cudaStream_t stream)
{
  ptr_length_pair cp_and_length;
  ptr_length_pair offset_and_length;

  // copy offsets to working memory
  size_t num_offsets = std::min(size_t{num_strings + 1}, device_strings_offsets.size());
  CUDA_TRY(cudaMemcpyAsync(device_strings_offsets.data().get(),
                           d_offsets,
                           sizeof(uint32_t) * num_offsets,
                           cudaMemcpyDeviceToDevice,
                           stream));
  uint32_t bytes_count = device_strings_offsets[num_offsets - 1];

  size_t BLOCKS                   = (bytes_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const size_t max_new_char_total = MAX_NEW_CHARS * BLOCKS * THREADS_PER_BLOCK;
  size_t threads_on_device        = BLOCKS * THREADS_PER_BLOCK;

  kernel_data_normalizer<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
    reinterpret_cast<const unsigned char*>(d_strings),
    device_strings_offsets.data().get(),
    bytes_count,
    device_cp_metadata.data().get(),
    device_aux_table.data().get(),
    do_lower_case,
    num_strings,
    device_code_points.data().get(),
    device_chars_per_thread.data().get());
  CHECK_CUDA(stream);

  // TODO: see if thrust remove-if and inclusive_scan (below) performance is equivalent
  // static NotEqual select_op((1 << SORT_BIT));
  // cub::DeviceSelect::If(cub_temp_storage.data().get(),
  //                      max_cub_storage_bytes,
  //                      device_code_points.data().get(),
  //                      device_code_points.data().get(),
  //                      device_num_selected.data().get(),
  //                      max_new_char_total,
  //                      select_op,
  //                      stream);
  // CHECK_CUDA(stream);
  auto execpol = rmm::exec_policy(stream);
  thrust::remove_if(execpol->on(stream),
                    device_code_points.begin(),
                    device_code_points.begin() + max_new_char_total,
                    [] __device__(auto value) { return value == (1 << SORT_BIT); });

  // We also need to prefix sum the number of characters up to an including the current character in
  // order to get the new strings lengths.
  // cub::DeviceScan::InclusiveSum(cub_temp_storage.data().get(),
  //                              max_cub_storage_bytes,
  //                              device_chars_per_thread.data().get(),
  //                              device_chars_per_thread.data().get(),
  //                              threads_on_device,
  //                              stream);
  // CHECK_CUDA(stream);
  thrust::inclusive_scan(execpol->on(stream),
                         device_chars_per_thread.begin(),
                         device_chars_per_thread.begin() + threads_on_device,
                         device_chars_per_thread.begin());

  constexpr uint16_t SENTENCE_UPDATE_THREADS = 64;
  size_t SEN_KERNEL_BLOCKS = (num_strings + SENTENCE_UPDATE_THREADS - 1) / SENTENCE_UPDATE_THREADS;
  update_strings_lengths<<<SEN_KERNEL_BLOCKS, SENTENCE_UPDATE_THREADS, 0, stream>>>(
    device_strings_offsets.data().get(), device_chars_per_thread.data().get(), num_strings);
  CHECK_CUDA(stream);

  offset_and_length.gpu_ptr = device_strings_offsets.data().get();
  offset_and_length.length  = num_strings + 1;

  uint32_t num_chars = 0;
  CUDA_TRY(cudaMemcpyAsync(&num_chars,
                           offset_and_length.gpu_ptr + num_strings,
                           sizeof(num_chars),
                           cudaMemcpyDeviceToHost,
                           stream));
  cp_and_length.gpu_ptr = device_code_points.data().get();
  cp_and_length.length  = num_chars;

  return std::make_pair(cp_and_length, offset_and_length);
}

}  // namespace detail
}  // namespace nvtext
