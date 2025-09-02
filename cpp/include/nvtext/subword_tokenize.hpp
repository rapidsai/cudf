/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT nvtext {

/**
 * @addtogroup nvtext_tokenize
 * @{
 * @file
 */

/**
 * @brief Result object for the subword_tokenize functions.
 */
struct tokenizer_result {
  /**
   * @brief The number of rows for the output token-ids.
   */
  uint32_t nrows_tensor{};
  /**
   * @brief The number of token-ids in each row.
   */
  uint32_t sequence_length{};
  /**
   * @brief A vector of token-ids for each row.
   *
   * The data is a flat matrix (nrows_tensor x sequence_length) of token-ids.
   * This column is of type UINT32 with no null entries.
   */
  std::unique_ptr<cudf::column> tensor_token_ids;
  /**
   * @brief This mask identifies which tensor-token-ids are valid.
   *
   * This column is of type UINT32 with no null entries.
   */
  std::unique_ptr<cudf::column> tensor_attention_mask;
  /**
   * @brief The metadata for each tensor row.
   *
   * There are three elements per tensor row [row-id, start_pos, stop_pos])
   * This column is of type UINT32 with no null entries.
   */
  std::unique_ptr<cudf::column> tensor_metadata;
};

/**
 * @brief Converts the output of the wordpiece tokenizer to tensor data
 * and metadata format
 *
 * For `do_truncate==true`:
 * ```
 * size of tensor_token_ids = max_sequence_length * input.size()
 * size of tensor_attention_mask = max_sequence_length * input.size()
 * size of tensor_metadata = 3 * input.size()
 * ```
 *
 * For `do_truncate==false` the number of output rows depends on the
 * number of tokens resolved and the `stride` value which may repeat tokens
 * in subsequent overflow rows.
 *
 * @throw cudf::logic_error if `stride > max_sequence_length`
 * @throw std::overflow_error if `max_sequence_length * max_rows_tensor`
 *        exceeds the column size limit
 *
 * @param input The input tokens from wordpiece tokenizer or similar algorithm
 * @param max_sequence_length Limit of the number of token-ids per row in the output
 * @param stride Each output row will replicate the `max_sequence_length - stride`
 *        token-ids from the previous row, unless it is the first row.
 * @param do_truncate If true, the tokenizer will discard all the token-ids after
 *        `max_sequence_length` for each input string. If false, it will use a new row
 *        in the output token-ids to continue generating the output.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resource to allocate any returned objects.
 * @return token-ids, attention-mask, and metadata
 */
tokenizer_result tokenized_to_tensor(
  cudf::lists_column_view const& input,
  cudf::size_type max_sequence_length,
  cudf::size_type stride,
  bool do_truncate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
