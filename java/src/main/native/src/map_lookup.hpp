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

namespace cudf {

namespace jni {

  /**
   * @brief Looks up a "map" column by specified key, and returns a column of string values.
   * 
   * The map-column is represented as follows:
   * 
   *  list_view<struct_view< string_view, string_view > >. 
   *                         <---KEY--->  <--VALUE-->
   * 
   * The string_view struct members are the key and value, respectively.
   * For each row in the input list column, the value corresponding to the first match
   * of the specified lookup_key is returned. If the key is not found, a null is returned.
   * 
   * @param map_column The input "map" column to be searched. Must be of
   *                   type list_view<struct_view<string_view, string_view>>.
   * @param lookup_key The search key, whose value is to be returned for each list row
   * @param has_nulls  Whether the input column might contain null list-rows, or null keys.
   * @param mr         The device memory resource to be used for allocations
   * @param stream     The CUDA stream
   * @return           A string_view column with the value from the first match in each list.
   *                   A null row is returned for any row where the lookup_key is not found.
   * @throw cudf::logic_error If the input column is not of type 
   *                          list_view<struct_view<string_view, string_view>>
   */
  std::unique_ptr<column> map_lookup(
    column_view const& map_column,
    string_scalar lookup_key,
    bool has_nulls                      = true,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
    cudaStream_t stream                 = 0);

} // namespace jni;

} // namespace cudf;