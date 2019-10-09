/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>

namespace cudf 
{
namespace strings
{

/**---------------------------------------------------------------------------*
 * @brief Row-wise concatenates the given list of strings columns.
 *
 * ```
 * s1 = ['aa', null, '', 'aa']
 * s2 = ['', 'bb', 'bb', null]
 * r = concatenate(s1,s2)
 * r is ['aa', null, 'bb', null]
 * ```
 *
 * @param strings_columns List of string columns to concatenate.
 * @param separator Null-terminated CPU string that should appear between each instance.
 *        Default is empty string.
 * @param narep Null-terminated CPU string that should be used in place of any null strings found.
 *        Default of null means any null operand produces a null result.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New column with concatenated results
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> concatenate( std::vector<strings_column_view>& strings_columns,
                                           const char* separator="",
                                           const char* narep=nullptr,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Concatenates all strings in the column into one new string.
 *
 * ```
 * s = ['aa', null, '', 'zz' ]
 * r = join_strings(s,':','_')
 * r is ['aa:_::zz']
 * ```
 *
 * @param strings Strings for this operation.
 * @param separator Null-terminated CPU string that should appear between each string.
 * @param narep Null-terminated CPU string that should represent any null strings found.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New column containing one string.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> join_strings( strings_column_view strings,
                                            const char* separator="",
                                            const char* narep=nullptr,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );


} // namespace strings
} // namespace cudf
