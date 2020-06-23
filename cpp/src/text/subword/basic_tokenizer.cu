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

#include <cudf/utilities/error.hpp>
#include <text/subword/detail/cp_util.cuh>
#include <text/subword/detail/tokenizer_utils.cuh>
#include <text/subword/detail/tokenizers.hpp>

#include <cub/cub.cuh>
#include <string>
#include <vector>

namespace nvtext {
namespace detail {
namespace {

#define SORT_BIT 22
#define THREADS_PER_BLOCK 64

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
 * If the byte at start_byte_for_thread is a head byte, the unicode code-point encoded by
 * the utf8 character started at that byte is returned and the head_byte boolean passed in
 * is set to true.
 *
 * If the byte at start_byte_for_thread is not a head byte, 0 is returned AND the head_byte
 * boolean passed in is set to false.
 *
 * All threads start reading bytes from the pointer denoted by sentences.
 *
 * @param sentences A pointer to the start of the sequence of characters to be tokenized.
 */
__device__ __forceinline__ uint32_t extract_code_points_from_utf8(
  const unsigned char* sentences, const uint32_t start_byte_for_thread, bool& head_byte)
{
  constexpr uint8_t max_utf8_blocks_for_char = 4;
  uint8_t utf8_blocks[max_utf8_blocks_for_char];

#pragma unroll
  for (int i = 0; i < max_utf8_blocks_for_char; ++i) {
    utf8_blocks[i] = sentences[start_byte_for_thread + i];
  }

  // We can have at most 5 bits encoding the length. We check those bits to infer the actual length
  const uint8_t length_encoding_bits = utf8_blocks[0] >> 3;

  head_byte = is_head_byte(utf8_blocks[0]);

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
  for (int i = 1; i < 4; ++i) { code_point |= ((utf8_blocks[i] & 0x3F) << (18 - 6 * i)); }

  // Zero out the bottom of code points with extra reads
  const uint8_t shift_amt = 24 - 6 * char_encoding_length;
  code_point >>= shift_amt;

  return code_point;
}
}  // namespace

__global__ void kernel_basic_tokenizer(const unsigned char* sentences,
                                       uint32_t* device_sentence_offsets,
                                       const size_t total_bytes,
                                       uint32_t* cp_metadata,
                                       uint64_t* aux_table,
                                       uint32_t* code_points,
                                       uint32_t* chars_per_thread,
                                       bool do_lower_case,
                                       uint32_t num_sentences)
{
  constexpr uint32_t init_val                     = (1 << SORT_BIT);
  uint32_t replacement_code_points[MAX_NEW_CHARS] = {init_val, init_val, init_val};

  bool head_byte                 = false;
  const uint32_t char_for_thread = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t num_new_chars         = 0;

  if (char_for_thread < total_bytes) {
    const uint32_t code_point =
      extract_code_points_from_utf8(sentences, char_for_thread, head_byte);
    const uint32_t metadata = get_cp_metadata(cp_metadata, code_point);

    if (head_byte && !should_remove_cp(metadata, do_lower_case)) {
      num_new_chars = 1;
      // Apply lower cases and accent stripping if necessary
      const bool replacement_needed = do_lower_case || always_replace(metadata);
      uint32_t new_cp               = replacement_needed ? get_first_cp(metadata) : code_point;
      new_cp                        = new_cp == 0 ? code_point : new_cp;

      replacement_code_points[0] = new_cp;
      if (do_lower_case && is_multi_char_transform(metadata)) {
        uint64_t next_cps                = get_extra_cps(aux_table, code_point);
        replacement_code_points[1]       = static_cast<uint32_t>(next_cps >> 32);
        const uint32_t potential_next_cp = static_cast<uint32_t>(next_cps);
        replacement_code_points[2] =
          potential_next_cp != 0 ? potential_next_cp : replacement_code_points[2];
        num_new_chars = 2 + (potential_next_cp != 0);
      }

      if (should_add_spaces(metadata, do_lower_case)) {
        // Need to shift all existing code-points up one
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

basic_tokenizer::basic_tokenizer(uint32_t max_num_sentences,
                                 uint32_t max_num_chars,
                                 std::vector<uint32_t> const& cp_metadata,
                                 std::vector<uint64_t> const& aux_table,
                                 bool do_lower_case,
                                 cudaStream_t stream)
  : do_lower_case(do_lower_case),
    device_sentence_offsets(max_num_sentences + 1),
    device_sentences(max_num_chars),
    device_cp_metadata{cp_metadata},
    device_aux_table{aux_table}
{
  size_t max_BLOCKS               = (max_num_chars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  size_t max_threads_on_device    = max_BLOCKS * THREADS_PER_BLOCK;
  const size_t max_new_char_total = MAX_NEW_CHARS * max_threads_on_device;
  device_code_points.resize(max_new_char_total);
  device_chars_per_thread.resize(max_threads_on_device);

  // Determine temporary device storage requirements for cub
  size_t temp_storage_scan_bytes    = 0;
  uint32_t* device_chars_per_thread = nullptr;
  cub::DeviceScan::InclusiveSum(nullptr,
                                temp_storage_scan_bytes,
                                device_chars_per_thread,
                                device_chars_per_thread,
                                max_threads_on_device);
  size_t temp_storage_select_bytes = 0;
  static NotEqual select_op((1 << SORT_BIT));
  cub::DeviceSelect::If(nullptr,
                        temp_storage_select_bytes,
                        thrust::raw_pointer_cast(device_code_points.data()),
                        thrust::raw_pointer_cast(device_code_points.data()),
                        thrust::raw_pointer_cast(device_num_selected.data()),
                        max_new_char_total,
                        select_op);
  max_cub_storage_bytes = std::max(temp_storage_scan_bytes, temp_storage_select_bytes);
  cub_temp_storage.resize(max_cub_storage_bytes);
  device_num_selected.resize(1);
}

std::pair<ptr_length_pair, ptr_length_pair> basic_tokenizer::tokenize(const char* device_sentences_,
                                                                      const uint32_t* offsets,
                                                                      uint32_t offset_size,
                                                                      cudaStream_t stream)
{
  ptr_length_pair cp_and_length;
  ptr_length_pair offset_and_length;

  size_t num_offsets = std::min(size_t{offset_size + 1}, device_sentence_offsets.size());
  CUDA_TRY(cudaMemcpyAsync(device_sentence_offsets.data().get(),
                           offsets,
                           sizeof(uint32_t) * num_offsets,
                           cudaMemcpyDeviceToDevice,
                           stream));
  uint32_t sentences_size = device_sentence_offsets[num_offsets - 1];

  static NotEqual select_op((1 << SORT_BIT));

  size_t BLOCKS                   = (sentences_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const size_t max_new_char_total = MAX_NEW_CHARS * BLOCKS * THREADS_PER_BLOCK;
  size_t threads_on_device        = BLOCKS * THREADS_PER_BLOCK;

  kernel_basic_tokenizer<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
    (unsigned char*)device_sentences_,
    thrust::raw_pointer_cast(device_sentence_offsets.data()),
    sentences_size,  // sentence_offsets[offset_size],
    thrust::raw_pointer_cast(device_cp_metadata.data()),
    thrust::raw_pointer_cast(device_aux_table.data()),
    thrust::raw_pointer_cast(device_code_points.data()),
    thrust::raw_pointer_cast(device_chars_per_thread.data()),
    do_lower_case,
    offset_size);
  CHECK_CUDA(stream);

  cub::DeviceSelect::If(thrust::raw_pointer_cast(cub_temp_storage.data()),
                        max_cub_storage_bytes,
                        thrust::raw_pointer_cast(device_code_points.data()),
                        thrust::raw_pointer_cast(device_code_points.data()),
                        thrust::raw_pointer_cast(device_num_selected.data()),
                        max_new_char_total,
                        select_op);
  CHECK_CUDA(stream);

  // We also need to prefix sum the number of characters up to an including the current character in
  // order to get the new sentence lengths.
  cub::DeviceScan::InclusiveSum(thrust::raw_pointer_cast(cub_temp_storage.data()),
                                max_cub_storage_bytes,
                                thrust::raw_pointer_cast(device_chars_per_thread.data()),
                                thrust::raw_pointer_cast(device_chars_per_thread.data()),
                                threads_on_device);
  CHECK_CUDA(stream);

  constexpr uint16_t SENTENCE_UPDATE_THREADS = 64;
  size_t SEN_KERNEL_BLOCKS = (offset_size + SENTENCE_UPDATE_THREADS - 1) / SENTENCE_UPDATE_THREADS;
  update_sentence_lengths<<<SEN_KERNEL_BLOCKS, SENTENCE_UPDATE_THREADS, 0, stream>>>(
    thrust::raw_pointer_cast(device_sentence_offsets.data()),
    thrust::raw_pointer_cast(device_chars_per_thread.data()),
    offset_size);
  CHECK_CUDA(stream);

  offset_and_length.gpu_ptr = thrust::raw_pointer_cast(device_sentence_offsets.data());
  offset_and_length.length  = offset_size + 1;

  uint32_t num_chars = 0;
  CUDA_TRY(cudaMemcpyAsync(&num_chars,
                           offset_and_length.gpu_ptr + offset_size,
                           sizeof(num_chars),
                           cudaMemcpyDeviceToHost,
                           stream));
  cp_and_length.gpu_ptr = thrust::raw_pointer_cast(device_code_points.data());
  cp_and_length.length  = num_chars;

  return std::make_pair(cp_and_length, offset_and_length);
}

}  // namespace detail
}  // namespace nvtext
