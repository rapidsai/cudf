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
/**
 * @addtogroup dictionary_search
 * @{
 * @file
 */

/**
 * @brief Return the index value for a given key.
 *
 * If the key does not exist in the dictionary the returned scalar will have `is_valid()==false`
 *
 * @throw cudf::logic_error if `key.type() != dictionary.keys().type()`
 *
 * @param dictionary The dictionary to search for the key.
 * @param key The value to search for in the dictionary keyset.
 * @return Numeric scalar index value of the key within the dictionary
 */
std::unique_ptr<scalar> get_index(
  dictionary_column_view const& dictionary,
  scalar const& key,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace dictionary
}  // namespace cudf
