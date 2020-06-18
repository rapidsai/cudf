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
#include <nvtext/subword_tokenize.hpp>

//#include <nvToolsExt.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "tokenizers.hpp"

namespace nvtext {
namespace detail {

std::unique_ptr<TokenizerResult> subword_tokenize(cudf::strings_column_view const& sentences,
                                                  std::string const& filename_hashed_vocabulary,
                                                  uint32_t max_sequence_length,
                                                  uint32_t stride,
                                                  bool do_lower,
                                                  bool do_truncate,
                                                  uint32_t max_num_sentences,
                                                  uint32_t max_num_chars,
                                                  uint32_t max_rows_tensor,
                                                  cudaStream_t stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  // Create tokenizer
  // nvtxRangePushA("create Tokenizer");
  GpuFullTokenizer tokenizer(filename_hashed_vocabulary,
                             max_num_sentences,
                             max_num_chars,
                             max_rows_tensor,
                             max_sequence_length,
                             stride,
                             do_truncate,
                             do_lower);
  // nvtxRangePop();

  auto strings_count = sentences.size();
  auto offsets       = sentences.offsets();
  auto d_offsets     = offsets.data<uint32_t>() + sentences.offset();
  // auto offset        = cudf::detail::get_value<int32_t>(offsets, sentences.offset(), stream);
  auto offset = thrust::device_pointer_cast(d_offsets)[0];
  // auto chars_bytes = cudf::detail::get_value<int32_t>(offsets, sentences.offset() +
  // strings_count, stream) - offset;
  auto chars_bytes = thrust::device_pointer_cast(d_offsets)[strings_count] - offset;
  auto d_chars     = sentences.chars().data<char>() + offset;

  // Run GPU tokenizer
  // nvtxRangePushA("Tokenize");
  // tokenizer.tokenize(device_sentences, offsets, offset_size);
  tokenizer.tokenize(d_chars, d_offsets, strings_count);
  // nvtxRangePop();

  // Get output from tokenizer
  auto result          = std::make_unique<TokenizerResult>();
  result->nrows_tensor = tokenizer.get_nrows_tensor_tokenIDS();
  // TODO use the 'mr' parameter to allocate these
  cudaMalloc((void**)&result->device_tensor_tokenIDS,
             result->nrows_tensor * max_sequence_length * sizeof(uint32_t));
  cudaMalloc((void**)&result->device_attention_mask,
             result->nrows_tensor * max_sequence_length * sizeof(uint32_t));
  cudaMalloc((void**)&result->device_tensor_metadata, result->nrows_tensor * 3 * sizeof(uint32_t));
  cudaMemcpy(result->device_tensor_tokenIDS,
             tokenizer.get_tensor_tokenIDS(),
             result->nrows_tensor * max_sequence_length * sizeof(uint32_t),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(result->device_attention_mask,
             tokenizer.get_attention_mask(),
             result->nrows_tensor * max_sequence_length * sizeof(uint32_t),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(result->device_tensor_metadata,
             tokenizer.get_tensor_metadata(),
             result->nrows_tensor * 3 * sizeof(uint32_t),
             cudaMemcpyDeviceToDevice);
  return result;
}

}  // namespace detail

std::unique_ptr<TokenizerResult> subword_tokenize(cudf::strings_column_view const& sentences,
                                                  std::string const& vocabulary_hashed_filename,
                                                  uint32_t max_sequence_length,
                                                  uint32_t stride,
                                                  bool do_lower,
                                                  bool do_truncate,
                                                  uint32_t max_num_sentences,
                                                  uint32_t max_num_chars,
                                                  uint32_t max_rows_tensor,
                                                  rmm::mr::device_memory_resource* mr)
{
  return detail::subword_tokenize(sentences,
                                  vocabulary_hashed_filename,
                                  max_sequence_length,
                                  stride,
                                  do_lower,
                                  do_truncate,
                                  max_num_sentences,
                                  max_num_chars,
                                  max_rows_tensor,
                                  0,
                                  mr);
}

}  // namespace nvtext
