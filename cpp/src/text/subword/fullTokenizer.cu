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

#include <text/subword/data_transfer_utils.cuh>
#include <text/subword/hash_utils.cuh>
#include <text/subword/tokenizer_utils.cuh>
#include <text/subword/tokenizers.hpp>

#include <iostream>
#include "cp_data_vec.ah"

namespace nvtext {

__global__ void compute_tensor_metadata_kernel(  // input
  uint32_t* token_ids,
  uint32_t* offsets,
  uint32_t* row2log,
  uint32_t* row2row_within_log,
  uint32_t max_sequence_length,
  uint32_t stride,
  bool do_truncate,
  // output
  uint32_t* final_tensor,
  uint32_t* attn_mask,
  uint32_t* metadata)
{
  uint32_t absolute_row_id      = blockIdx.x;
  uint32_t log_id               = row2log[absolute_row_id];
  uint32_t row_within_log       = row2row_within_log[absolute_row_id];
  uint32_t offset_token_ids_log = offsets[log_id];
  uint32_t n_tokens_log         = offsets[log_id + 1] - offset_token_ids_log;
  bool last_row_of_log =
    (absolute_row_id == gridDim.x - 1) || (row2log[absolute_row_id + 1] != log_id);

  uint32_t row_offset_token_ids = offset_token_ids_log;
  if (row_within_log) row_offset_token_ids += max_sequence_length;
  for (int i = 1; i < row_within_log; i++) row_offset_token_ids += stride;

  if (row_within_log == 0) {
    if (threadIdx.x < n_tokens_log) {
      // copy token ids
      final_tensor[absolute_row_id * max_sequence_length + threadIdx.x] =
        token_ids[row_offset_token_ids + threadIdx.x];
      attn_mask[absolute_row_id * max_sequence_length + threadIdx.x] = 1;
    } else {
      // pad with 0
      final_tensor[absolute_row_id * max_sequence_length + threadIdx.x] = 0;
      attn_mask[absolute_row_id * max_sequence_length + threadIdx.x]    = 0;
    }
  } else {
    uint32_t n_replicates = max_sequence_length - stride;
    if ((row_offset_token_ids - n_replicates + threadIdx.x) <
        (offset_token_ids_log + n_tokens_log)) {
      // replicate elements or copy new tokens
      final_tensor[absolute_row_id * max_sequence_length + threadIdx.x] =
        token_ids[row_offset_token_ids - n_replicates + threadIdx.x];
      attn_mask[absolute_row_id * max_sequence_length + threadIdx.x] = 1;
    } else {
      // pad with 0
      final_tensor[absolute_row_id * max_sequence_length + threadIdx.x] = 0;
      attn_mask[absolute_row_id * max_sequence_length + threadIdx.x]    = 0;
    }
  }

  // write metadata
  if (threadIdx.x == 0) {
    metadata[absolute_row_id * 3] = log_id;
    if (row_within_log == 0)
      metadata[absolute_row_id * 3 + 1] = 0;
    else
      metadata[absolute_row_id * 3 + 1] = (max_sequence_length - stride) / 2;
    if (last_row_of_log) {
      if (n_tokens_log < max_sequence_length)
        metadata[absolute_row_id * 3 + 2] = n_tokens_log - 1;
      else {
        if (!do_truncate)
          metadata[absolute_row_id * 3 + 2] =
            (max_sequence_length - stride) + (n_tokens_log - max_sequence_length) % stride - 1;
        else
          // truncate
          metadata[absolute_row_id * 3 + 2] = (max_sequence_length - 1);
      }
    } else
      metadata[absolute_row_id * 3 + 2] =
        max_sequence_length - (max_sequence_length - stride) / 2 - 1;
  }
}

GpuFullTokenizer::GpuFullTokenizer(std::string const& vocab_file,
                                   uint32_t max_num_sentences,
                                   uint32_t max_num_chars,
                                   uint32_t max_rows_final_tensor,
                                   uint32_t max_sequence_length,
                                   uint32_t stride,
                                   bool do_truncate,
                                   bool do_lower_case,
                                   int max_inp_chars_per_word)
  : max_sequence_length(max_sequence_length),
    stride(stride),
    do_truncate(do_truncate),
    basic_tokenizer(max_num_sentences, max_num_chars, cp_data, aux_data, do_lower_case),
    word_piece_tokenizer(vocab_file, max_num_chars, max_inp_chars_per_word),
    tensor_tokenIDS(max_rows_final_tensor * max_sequence_length),
    attention_mask(max_rows_final_tensor * max_sequence_length),
    metadata(max_rows_final_tensor * 3),
    device_row2log(max_rows_final_tensor),
    device_row2row_within_log(max_rows_final_tensor)
{
}

void GpuFullTokenizer::tokenize(const char* device_sentences,
                                const uint32_t* offsets,
                                uint32_t offset_size)
{
  auto cps_and_offsets = basic_tokenizer.tokenize(device_sentences, offsets, offset_size);
  word_piece_tokenizer.tokenize(cps_and_offsets.first, cps_and_offsets.second);
  // return cps_and_offsets;
  uint32_t* device_token_ids = cps_and_offsets.first.gpu_ptr;
  uint32_t* device_offsets   = cps_and_offsets.second.gpu_ptr;

  // copy log offsets to host
  std::vector<uint32_t> host_offsets;
  host_offsets.resize(offset_size + 1);
  cudaMemcpy(host_offsets.data(),
             device_offsets,
             sizeof(uint32_t) * (offset_size + 1),
             cudaMemcpyDeviceToHost);

  // compute number of rows required for final tensor
  nrows_tensor_tokenIDS = 0;
  std::vector<uint32_t> nrows_per_log;
  nrows_per_log.resize(offset_size);
  for (uint32_t i = 0; i < offset_size; i++) {
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
  for (uint32_t i = 0; i < offset_size; i++) {
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
  compute_tensor_metadata_kernel<<<nrows_tensor_tokenIDS, max_sequence_length>>>(
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
}

uint32_t GpuFullTokenizer::get_nrows_tensor_tokenIDS() { return nrows_tensor_tokenIDS; }

uint32_t* GpuFullTokenizer::get_tensor_tokenIDS()
{
  return thrust::raw_pointer_cast(tensor_tokenIDS.data());
}

uint32_t* GpuFullTokenizer::get_attention_mask()
{
  return thrust::raw_pointer_cast(attention_mask.data());
}

uint32_t* GpuFullTokenizer::get_tensor_metadata()
{
  return thrust::raw_pointer_cast(metadata.data());
}

GpuFullTokenizer::~GpuFullTokenizer() {}

}  // namespace nvtext
