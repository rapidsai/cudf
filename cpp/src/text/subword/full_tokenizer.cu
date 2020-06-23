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
#include <text/subword/detail/hash_utils.cuh>
#include <text/subword/detail/tokenizer_utils.cuh>
#include <text/subword/detail/tokenizers.hpp>

#include <iostream>
#include "detail/cp_data_vec.ah"

namespace nvtext {
namespace detail {

full_tokenizer::full_tokenizer(std::string const& vocab_file,
                               uint32_t max_num_sentences,
                               uint32_t max_num_chars,
                               uint32_t max_rows_final_tensor,
                               uint32_t max_sequence_length,
                               uint32_t stride,
                               bool do_truncate,
                               bool do_lower_case,
                               cudaStream_t stream,
                               int max_inp_chars_per_word)
  : max_sequence_length(max_sequence_length),
    stride(stride),
    do_truncate(do_truncate),
    normalizer(max_num_sentences, max_num_chars, cp_data, aux_data, do_lower_case, stream)
// tokenizer(vocab_file, max_num_chars, max_inp_chars_per_word, stream)
// tensor_tokenIDS(max_rows_final_tensor * max_sequence_length),
// attention_mask(max_rows_final_tensor * max_sequence_length),
// metadata(max_rows_final_tensor * 3),
// device_row2log(max_rows_final_tensor),
// device_row2row_within_log(max_rows_final_tensor)
{
  detail::transfer_hash_info_to_device(vocab_file,
                                       device_hash_table,
                                       device_bin_coefficients,
                                       device_bin_offsets,
                                       unk_token_id,
                                       first_tok_id,
                                       sep_tok_id,
                                       outer_hash_a_param,
                                       outer_hash_b_param,
                                       num_outer_bins);

  max_word_length = max_inp_chars_per_word;

  const size_t max_new_char_total = MAX_NEW_CHARS * max_num_chars;
  device_token_ids.resize(max_new_char_total);
  const size_t device_word_indices_count = 2 * max_new_char_total;
  device_word_indices.resize(device_word_indices_count);

  const size_t four_byte_cp_chunks = 1 + (max_new_char_total - 1) / sizeof(uint32_t);
  const size_t rounded_num_cps     = sizeof(uint32_t) * four_byte_cp_chunks;
  device_tokens_per_word.resize(rounded_num_cps);

  // Determine temporary device storage requirements for cub
  static NotEqual select_op(std::numeric_limits<uint32_t>::max());
  size_t temp_storage_bytes = 0, temp_storage_bytes_2 = 0;
  cub::DeviceSelect::If(nullptr,
                        temp_storage_bytes,
                        device_word_indices.data().get(),
                        device_word_indices.data().get(),
                        device_num_selected.data().get(),
                        2 * max_new_char_total,
                        select_op);
  cub::DeviceScan::InclusiveSum(nullptr,
                                temp_storage_bytes_2,
                                device_tokens_per_word.data().get(),
                                device_word_indices.data().get(),
                                max_new_char_total);
  max_cub_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_2);
  cub_temp_storage.resize(max_cub_storage_bytes);
  device_num_selected.resize(1);
}

std::pair<uint32_t*, uint32_t*> full_tokenizer::tokenize(const char* d_strings,
                                                         const uint32_t* d_offsets,
                                                         uint32_t num_strings,
                                                         cudaStream_t stream)
{
  auto cps_and_offsets = normalizer.normalize(d_strings, d_offsets, num_strings, stream);
  tokenize(cps_and_offsets.first, cps_and_offsets.second, stream);
  // return cps_and_offsets;
  return std::make_pair(cps_and_offsets.first.gpu_ptr, cps_and_offsets.second.gpu_ptr);
#if 0  
  uint32_t* device_token_ids = cps_and_offsets.first.gpu_ptr;
  uint32_t* device_offsets   = cps_and_offsets.second.gpu_ptr;

  // copy log offsets to host
  std::vector<uint32_t> host_offsets;
  host_offsets.resize(num_strings + 1);
  CUDA_TRY(cudaMemcpyAsync(host_offsets.data(),
                           device_offsets,
                           sizeof(uint32_t) * (num_strings + 1),
                           cudaMemcpyDeviceToHost,
                           stream));

  // compute number of rows required for final tensor
  nrows_tensor_tokenIDS = 0;
  std::vector<uint32_t> nrows_per_log;
  nrows_per_log.resize(num_strings);
  for (uint32_t i = 0; i < num_strings; i++) {
    uint32_t ntokens = host_offsets[i + 1] - host_offsets[i];
    if (do_truncate || ntokens <= max_sequence_length)
      nrows_per_log[i] = 1;
    else {
      ntokens -= max_sequence_length;
      nrows_per_log[i] = 1 + (ntokens / stride);
      if (ntokens % stride) nrows_per_log[i]++;
    }
    nrows_tensor_tokenIDS += nrows_per_log[i];
  }
  // compute global_row to log, and global_row to within_log_row correspondence
  std::vector<uint32_t> host_row2log;
  std::vector<uint32_t> host_row2row_within_log;
  host_row2log.resize(nrows_tensor_tokenIDS);
  host_row2row_within_log.resize(nrows_tensor_tokenIDS);
  int row_id = 0;
  for (uint32_t i = 0; i < num_strings; i++) {
    for (uint32_t j = 0; j < nrows_per_log[i]; j++) {
      host_row2log[row_id]            = i;
      host_row2row_within_log[row_id] = j;
      row_id++;
    }
  }

  // copy info to GPU
  device_row2log            = host_row2log;
  device_row2row_within_log = host_row2row_within_log;

  // compute final-tensor, mask, and metadata
  compute_tensor_metadata_kernel<<<nrows_tensor_tokenIDS, max_sequence_length, 0, stream>>>(
    device_token_ids,
    device_offsets,
    thrust::raw_pointer_cast(device_row2log.data()),
    thrust::raw_pointer_cast(device_row2row_within_log.data()),
    max_sequence_length,
    stride,
    do_truncate,
    thrust::raw_pointer_cast(tensor_tokenIDS.data()),
    thrust::raw_pointer_cast(attention_mask.data()),
    thrust::raw_pointer_cast(metadata.data()));
#endif
}

// uint32_t full_tokenizer::get_nrows_tensor_tokenIDS() { return nrows_tensor_tokenIDS; }
//
// uint32_t* full_tokenizer::get_tensor_tokenIDS()
//{
//  return thrust::raw_pointer_cast(tensor_tokenIDS.data());
//}
//
// uint32_t* full_tokenizer::get_attention_mask()
//{
//  return thrust::raw_pointer_cast(attention_mask.data());
//}
//
// uint32_t* full_tokenizer::get_tensor_metadata()
//{
//  return thrust::raw_pointer_cast(metadata.data());
//}

}  // namespace detail
}  // namespace nvtext
