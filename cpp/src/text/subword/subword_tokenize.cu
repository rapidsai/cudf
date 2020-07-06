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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>
#include <nvtext/detail/load_hash_file.hpp>
#include <nvtext/subword_tokenize.hpp>
#include <text/subword/detail/wordpiece_tokenizer.hpp>

#include <thrust/for_each.h>
#include <thrust/transform_scan.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Convert tokens and row2tensor map to final tensor data.
 *
 * @param[in] token_ids Tokens from tokenizer
 * @param[in] offsets Offsets to each string's output row of tokens
 * @param[in] row2tensor String to tensor token counts
 * @param[in] row2row_within_tensor Token counts within sub-rows of the output
 * @param[in] max_sequence_length Maximum number of tokens in a row
 * @param[in] stride Number of tokens in sub-rows
 * @param[in] do_truncate True if tokens should not spill into sub-rows in the output
 * @param[out] final_tensor Output vector of token-ids
 * @param[out] attn_mask Identifies valid token id entries
 * @param[out] metadata Additional data per row
 */
__global__ void kernel_compute_tensor_metadata(
  // input
  uint32_t const* token_ids,
  uint32_t const* offsets,
  uint32_t const* row2tensor,
  uint32_t const* row2row_within_tensor,
  uint32_t max_sequence_length,
  uint32_t stride,
  bool do_truncate,
  // output
  uint32_t* final_tensor,
  uint32_t* attn_mask,
  uint32_t* metadata)
{
  uint32_t const absolute_row_id         = blockIdx.x;
  uint32_t const tensor_id               = row2tensor[absolute_row_id];
  uint32_t const row_within_tensor       = row2row_within_tensor[absolute_row_id];
  uint32_t const offset_token_ids_tensor = offsets[tensor_id];
  uint32_t const n_tokens_tensor         = offsets[tensor_id + 1] - offset_token_ids_tensor;
  bool const last_row_of_tensor =
    (absolute_row_id == gridDim.x - 1) || (row2tensor[absolute_row_id + 1] != tensor_id);

  uint32_t const row_offset_token_ids =
    offset_token_ids_tensor +
    (row_within_tensor ? (max_sequence_length + (stride * (row_within_tensor - 1))) : 0);

  auto const output_idx = absolute_row_id * max_sequence_length + threadIdx.x;

  if (row_within_tensor == 0) {
    if (threadIdx.x < n_tokens_tensor) {
      // copy token ids
      final_tensor[output_idx] = token_ids[row_offset_token_ids + threadIdx.x];
      attn_mask[output_idx]    = 1;
    } else {
      // pad with 0
      final_tensor[output_idx] = 0;
      attn_mask[output_idx]    = 0;
    }
  } else {
    uint32_t const n_replicates = max_sequence_length - stride;
    if ((row_offset_token_ids - n_replicates + threadIdx.x) <
        (offset_token_ids_tensor + n_tokens_tensor)) {
      // replicate elements or copy new tokens
      final_tensor[output_idx] = token_ids[row_offset_token_ids - n_replicates + threadIdx.x];
      attn_mask[output_idx]    = 1;
    } else {
      // pad with 0
      final_tensor[output_idx] = 0;
      attn_mask[output_idx]    = 0;
    }
  }

  // write metadata
  if (threadIdx.x == 0) {
    metadata[absolute_row_id * 3] = tensor_id;
    if (row_within_tensor == 0)
      metadata[absolute_row_id * 3 + 1] = 0;
    else
      metadata[absolute_row_id * 3 + 1] = (max_sequence_length - stride) / 2;
    if (last_row_of_tensor) {
      if (n_tokens_tensor < max_sequence_length)
        metadata[absolute_row_id * 3 + 2] = n_tokens_tensor - 1;
      else {
        if (!do_truncate)
          metadata[absolute_row_id * 3 + 2] =
            (max_sequence_length - stride) + (n_tokens_tensor - max_sequence_length) % stride - 1;
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
                                  hashed_vocabulary const& vocab_table,
                                  uint32_t max_sequence_length,
                                  uint32_t stride,
                                  bool do_lower_case,
                                  bool do_truncate,
                                  uint32_t max_num_strings,
                                  uint32_t max_num_chars,
                                  uint32_t max_rows_tensor,
                                  cudaStream_t stream,
                                  rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = strings.size();
  auto const offsets       = strings.offsets();
  auto const d_offsets     = offsets.data<uint32_t>() + strings.offset();
  auto const offset        = cudf::detail::get_value<int32_t>(offsets, strings.offset(), stream);
  auto const chars_bytes =
    cudf::detail::get_value<int32_t>(offsets, strings.offset() + strings_count, stream) - offset;
  auto const d_chars = strings.chars().data<char>() + offset;

  // Create tokenizer
  wordpiece_tokenizer tokenizer(vocab_table,
                                max_num_strings,
                                max_num_chars,
                                max_rows_tensor,
                                max_sequence_length,
                                stride,
                                do_truncate,
                                do_lower_case,
                                stream);
  // Run tokenizer
  auto const tokens = tokenizer.tokenize(d_chars, d_offsets, strings_count, stream);
  // assign output components
  uint32_t const* device_token_ids = tokens.first;
  uint32_t const* device_offsets   = tokens.second;

  // Format output from tokenizer
  // each string can create 1 or more tensor entries
  // compute the string-per-tensor offsets values by scanning over the number of tokens for each
  // string
  rmm::device_uvector<uint32_t> offsets_per_tensor(strings_count + 1, stream);
  auto d_offsets_per_tensor = offsets_per_tensor.data();
  auto const execpol        = rmm::exec_policy(stream);
  thrust::transform_exclusive_scan(
    execpol->on(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(strings_count + 1),
    offsets_per_tensor.begin(),
    [device_offsets, do_truncate, max_sequence_length, stride] __device__(cudf::size_type idx) {
      uint32_t num_tokens = device_offsets[idx + 1] - device_offsets[idx];
      if (do_truncate || num_tokens <= max_sequence_length) return uint32_t{1};
      return 1 + ((num_tokens - max_sequence_length + stride - 1) / stride);
    },
    uint32_t{0},
    thrust::plus<uint32_t>());
  // last element is the total number of tokens
  uint32_t const nrows_tensor_token_ids = offsets_per_tensor.element(strings_count, stream);

  // compute global_row to tensor, and global_row to within_tensor_row correspondence
  rmm::device_uvector<uint32_t> row2tensor(nrows_tensor_token_ids, stream);
  auto d_row2tensor = row2tensor.data();
  rmm::device_uvector<uint32_t> row2row_within_tensor(nrows_tensor_token_ids, stream);
  auto d_row2row_within_tensor = row2row_within_tensor.data();
  thrust::for_each_n(
    execpol->on(stream),
    thrust::make_counting_iterator<uint32_t>(0),
    strings_count,
    [d_offsets_per_tensor, d_row2tensor, d_row2row_within_tensor] __device__(auto idx) {
      uint32_t offset = d_offsets_per_tensor[idx];
      uint32_t nrows  = d_offsets_per_tensor[idx + 1] - offset;
      for (uint32_t jdx = 0; jdx < nrows; ++jdx) {
        d_row2tensor[jdx + offset]            = idx;
        d_row2row_within_tensor[jdx + offset] = jdx;
      }
    });

  // create output data columns
  auto tensor_token_ids = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32},
                                                    nrows_tensor_token_ids * max_sequence_length,
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
  auto tensor_attention_mask =
    cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32},
                              nrows_tensor_token_ids * max_sequence_length,
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto tensor_metadata = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32},
                                                   nrows_tensor_token_ids * 3,
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);

  // compute final-tensor, mask, and metadata
  kernel_compute_tensor_metadata<<<nrows_tensor_token_ids, max_sequence_length, 0, stream>>>(
    device_token_ids,
    device_offsets,
    d_row2tensor,
    d_row2row_within_tensor,
    max_sequence_length,
    stride,
    do_truncate,
    tensor_token_ids->mutable_view().data<uint32_t>(),
    tensor_attention_mask->mutable_view().data<uint32_t>(),
    tensor_metadata->mutable_view().data<uint32_t>());

  return tokenizer_result{nrows_tensor_token_ids,
                          max_sequence_length,
                          std::move(tensor_token_ids),
                          std::move(tensor_attention_mask),
                          std::move(tensor_metadata)};
}

}  // namespace detail

tokenizer_result subword_tokenize(cudf::strings_column_view const& strings,
                                  std::string const& filename_hashed_vocabulary,
                                  uint32_t max_sequence_length,
                                  uint32_t stride,
                                  bool do_lower_case,
                                  bool do_truncate,
                                  uint32_t max_num_strings,
                                  uint32_t max_num_chars,
                                  uint32_t max_rows_tensor,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  hashed_vocabulary vocab_table = load_vocabulary_file(filename_hashed_vocabulary, mr);
  return detail::subword_tokenize(strings,
                                  vocab_table,
                                  max_sequence_length,
                                  stride,
                                  do_lower_case,
                                  do_truncate,
                                  max_num_strings,
                                  max_num_chars,
                                  max_rows_tensor,
                                  0,
                                  mr);
}

tokenizer_result subword_tokenize(cudf::strings_column_view const& strings,
                                  hashed_vocabulary const& vocabulary_table,
                                  uint32_t max_sequence_length,
                                  uint32_t stride,
                                  bool do_lower_case,
                                  bool do_truncate,
                                  uint32_t max_num_strings,
                                  uint32_t max_num_chars,
                                  uint32_t max_rows_tensor,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::subword_tokenize(strings,
                                  vocabulary_table,
                                  max_sequence_length,
                                  stride,
                                  do_lower_case,
                                  do_truncate,
                                  max_num_strings,
                                  max_num_chars,
                                  max_rows_tensor,
                                  0,
                                  mr);
}

}  // namespace nvtext
