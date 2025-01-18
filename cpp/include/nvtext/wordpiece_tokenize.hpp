/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
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
 * @brief Vocabulary object to be used with nvtext::wordpiece_tokenizer
 *
 * Use nvtext::load_wordpiece_vocabulary to create this object.
 */
struct wordpiece_vocabulary {
  /**
   * @brief Vocabulary object constructor
   *
   * Token ids are the row indices within the vocabulary column.
   * Each vocabulary entry is expected to be unique otherwise the behavior is undefined.
   *
   * @throw cudf::logic_error if `vocabulary` contains nulls or is empty
   *
   * @param input Strings for the vocabulary
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  wordpiece_vocabulary(cudf::strings_column_view const& input,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  ~wordpiece_vocabulary();

  struct wordpiece_vocabulary_impl;
  wordpiece_vocabulary_impl* _impl{};
};

/**
 * @brief Create a tokenize_vocabulary object from a strings column
 *
 * Token ids are the row indices within the vocabulary column.
 * Each vocabulary entry is expected to be unique otherwise the behavior is undefined.
 *
 * @throw cudf::logic_error if `vocabulary` contains nulls or is empty
 *
 * @param input Strings for the vocabulary
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Object to be used with nvtext::tokenize_with_vocabulary
 */
std::unique_ptr<wordpiece_vocabulary> load_wordpiece_vocabulary(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the token ids for the input string by using a word-piece
 * tokenizer with the given vocabulary limited by a specified maximum number
 * of words per row
 *
 * ```
 * [['I', 'have', 'a', 'new', 'GP', '##U', '!', '[PAD]']]
 * ```
 * Any null row entry results in a corresponding null entry in the output
 *
 * @param input Strings column to tokenize
 * @param vocabulary Used to lookup tokens within `input`
 * @param max_words_per_row Maximum number of words to tokenize for each row;
 *        Default 0 includes all words when tokenizing.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column of token ids
 */
std::unique_ptr<cudf::column> wordpiece_tokenize(
  cudf::strings_column_view const& input,
  wordpiece_vocabulary const& vocabulary,
  cudf::size_type max_words_per_row = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of tokenize group
}  // namespace CUDF_EXPORT nvtext
