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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar.hpp>

namespace cudf {
namespace dictionary {
namespace detail {

/**
 * @copydoc cudf::dictionary::get_index(dictionary_column_view const&,scalar
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<scalar> get_index(
  dictionary_column_view const& dictionary,
  scalar const& key,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Get the index for a key if it were added to the given dictionary.
 *
 * The actual index is returned if the `key` is already part of the dictionary's key set.
 *
 * @code{.pseudo}
 * d1 = {["a","c","d"],[2,0,1,0]}
 * idx = get_insert_index(d1,"b")
 * idx is 1
 * @endcode{.pseudo}
 *
 * @throw cudf::logic_error if `key.type() != dictionary.keys().type()`
 *
 * @param dictionary The dictionary to search for the key.
 * @param key The value to search for in the dictionary keyset.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Numeric scalar index value of the key within the dictionary
 */
std::unique_ptr<scalar> get_insert_index(
  dictionary_column_view const& dictionary,
  scalar const& key,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
