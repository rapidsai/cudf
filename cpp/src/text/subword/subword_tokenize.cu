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

//#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>
#include <nvtext/subword_tokenize.hpp>

#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <text/subword/detail/wordpiece_tokenizer.hpp>

namespace nvtext {
namespace detail {
namespace {

__global__ void compute_tensor_metadata_kernel(  // input
  uint32_t const* token_ids,
  uint32_t const* offsets,
  uint32_t const* row2log,
  uint32_t const* row2row_within_log,
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

}  // namespace

tokenizer_result subword_tokenize(cudf::strings_column_view const& strings,
                                  std::string const& filename_hashed_vocabulary,
                                  uint32_t max_sequence_length,
                                  uint32_t stride,
                                  bool do_lower,
                                  bool do_truncate,
                                  uint32_t max_num_strings,
                                  uint32_t max_num_chars,
                                  uint32_t max_rows_tensor,
                                  cudaStream_t stream,
                                  rmm::mr::device_memory_resource* mr)
{
  auto strings_count = strings.size();
  auto offsets       = strings.offsets();
  auto d_offsets     = offsets.data<uint32_t>() + strings.offset();
  // auto offset        = cudf::detail::get_value<int32_t>(offsets, strings.offset(), stream);
  auto offset = thrust::device_pointer_cast(d_offsets)[0];
  // auto chars_bytes = cudf::detail::get_value<int32_t>(offsets, strings.offset() +
  // strings_count, stream) - offset;
  auto chars_bytes = thrust::device_pointer_cast(d_offsets)[strings_count] - offset;
  auto d_chars     = strings.chars().data<char>() + offset;

  // Create tokenizer
  nvtxRangePushA("create tokenizer");
  wordpiece_tokenizer tokenizer(filename_hashed_vocabulary,
                                max_num_strings,
                                max_num_chars,
                                max_rows_tensor,
                                max_sequence_length,
                                stride,
                                do_truncate,
                                do_lower,
                                stream);
  nvtxRangePop();

  // Run tokenizer
  nvtxRangePushA("tokenize");
  std::pair<uint32_t*, uint32_t*> tokens =
    tokenizer.tokenize(d_chars, d_offsets, strings_count, stream);
  nvtxRangePop();
  uint32_t* device_token_ids = tokens.first;
  uint32_t* device_offsets   = tokens.second;

  // Format output from tokenizer
  nvtx3::thread_range rt{"tokenizer output"};
  // copy log offsets to host
  std::vector<uint32_t> host_offsets(strings_count + 1);
  CUDA_TRY(cudaMemcpyAsync(host_offsets.data(),
                           device_offsets,
                           sizeof(uint32_t) * host_offsets.size(),
                           cudaMemcpyDeviceToHost,
                           stream));

  // compute number of rows required for final tensor
  uint32_t nrows_tensor_tokenIDS = 0;
  std::vector<uint32_t> nrows_per_log(strings_count);
  for (auto i = 0; i < strings_count; i++) {
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
  std::vector<uint32_t> host_row2log(nrows_tensor_tokenIDS);
  std::vector<uint32_t> host_row2row_within_log(nrows_tensor_tokenIDS);
  int row_id = 0;
  for (auto i = 0; i < strings_count; i++) {
    for (uint32_t j = 0; j < nrows_per_log[i]; j++) {
      host_row2log[row_id]            = i;
      host_row2row_within_log[row_id] = j;
      row_id++;
    }
  }

  // copy info to GPU
  // correspondence between each row of tensor_tokenIDS and log_id
  rmm::device_vector<uint32_t> device_row2log = host_row2log;
  // correspondence between each row of tensor_tokenIDS and row number within a specific log
  rmm::device_vector<uint32_t> device_row2row_within_log = host_row2row_within_log;
  // output data
  rmm::device_buffer d_tensor_token_ids(
    nrows_tensor_tokenIDS * max_sequence_length * sizeof(uint32_t), stream, mr);
  rmm::device_buffer d_attention_mask(
    nrows_tensor_tokenIDS * max_sequence_length * sizeof(uint32_t), stream, mr);
  rmm::device_buffer d_tensor_metadata(nrows_tensor_tokenIDS * 3 * sizeof(uint32_t), stream, mr);

  // compute final-tensor, mask, and metadata
  compute_tensor_metadata_kernel<<<nrows_tensor_tokenIDS, max_sequence_length, 0, stream>>>(
    device_token_ids,
    device_offsets,
    device_row2log.data().get(),
    device_row2row_within_log.data().get(),
    max_sequence_length,
    stride,
    do_truncate,
    reinterpret_cast<uint32_t*>(d_tensor_token_ids.data()),
    reinterpret_cast<uint32_t*>(d_attention_mask.data()),
    reinterpret_cast<uint32_t*>(d_tensor_metadata.data()));

  return tokenizer_result{nrows_tensor_tokenIDS,
                          std::make_unique<rmm::device_buffer>(std::move(d_tensor_token_ids)),
                          std::make_unique<rmm::device_buffer>(std::move(d_attention_mask)),
                          std::make_unique<rmm::device_buffer>(std::move(d_tensor_metadata))};
}

}  // namespace detail

tokenizer_result subword_tokenize(cudf::strings_column_view const& strings,
                                  std::string const& vocabulary_hashed_filename,
                                  uint32_t max_sequence_length,
                                  uint32_t stride,
                                  bool do_lower,
                                  bool do_truncate,
                                  uint32_t max_num_strings,
                                  uint32_t max_num_chars,
                                  uint32_t max_rows_tensor,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::subword_tokenize(strings,
                                  vocabulary_hashed_filename,
                                  max_sequence_length,
                                  stride,
                                  do_lower,
                                  do_truncate,
                                  max_num_strings,
                                  max_num_chars,
                                  max_rows_tensor,
                                  0,
                                  mr);
}

}  // namespace nvtext
