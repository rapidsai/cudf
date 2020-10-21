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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
/**
 * @copydoc cudf::dictionary::add_keys(dictionary_column_view const&,column_view
 * const&,mm::mr::device_memory_resource*)
 *
 * @param dictionary_column Existing dictionary column.
 * @param new_keys New keys to incorporate into the dictionary_column
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column.
 */
std::unique_ptr<column> add_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& new_keys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::dictionary::remove_keys(dictionary_column_view const&,column_view
 * const&,mm::mr::device_memory_resource*)
 *
 * @param dictionary_column Existing dictionary column.
 * @param keys_to_remove The keys to remove from the dictionary_column
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& keys_to_remove,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::dictionary::remove_unused_keys(dictionary_column_view
 * const&,mm::mr::device_memory_resource*)
 *
 * @param dictionary_column Existing dictionary column.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_unused_keys(
  dictionary_column_view const& dictionary_column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::dictionary::remove_unused_keys(dictionary_column_view
 * const&,mm::mr::device_memory_resource*)
 *
 * @param dictionary_column Existing dictionary column.
 * @param keys New keys to use for the output column. Must not contain nulls.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column.
 */
std::unique_ptr<column> set_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& keys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Create new dictionaries that have keys merged from the input dictionaries.
 *
 * This will concatenate the keys for each dictionary and then call `set_keys` on each.
 * The result is a vector of new dictionaries with a common set of keys.
 *
 * @param input Dictionary columns to match keys.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column.
 */
std::vector<std::unique_ptr<column>> match_dictionaries(
  std::vector<dictionary_column_view> input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
