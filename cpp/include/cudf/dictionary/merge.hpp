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

namespace cudf
{
namespace dictionary
{

/**
 * @brief Create a new dictionary column by merging the keys and indices
 * from two existing dictionary columns.
 *
 * The indices of the resulting column are created to appear as though the
 * second dictionary_column indices is appended to the first.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = {["b","e"],[1,0,1,1,0]}
 * d3 = merge(d1,d2)
 * d3 is now {["a","b","c","d","e"],[3,0,2,0,4,1,4,4,1]}
 * ```
 *
 * @throw cudf_logic_error if the keys types do not match for both dictionary columns.
 *
 * @param dictionary_column1 Existing dictionary column.
 * @param dictionary_column2 2nd existing dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> merge_dictionaries( dictionary_column_view const& dictionary_column1,
                                            dictionary_column_view const& dictionary_column2,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace dictionary
} // namespace cudf
