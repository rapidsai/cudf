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
#include <cudf/table/table_view.hpp>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Returns a single column by vertically concatenating the given vector of
 * strings columns.
 *
 * The caller is required to fill in the validity mask in the output column.
 *
 * ```
 * s1 = ['aa', 'bb', 'cc']
 * s2 = ['dd', 'ee']
 * r = concatenate_vertically([s1,s2])
 * r is now ['aa', 'bb', 'cc', 'dd', 'ee']
 * ```
 *
 * @param strings_columns List of string columns to concatenate.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use for any kernels in this function.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate( std::vector<strings_column_view> const& strings_columns,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0 );

} // namespace detail
} // namespace strings
} // namespace cudf
