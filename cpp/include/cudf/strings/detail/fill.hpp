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
 * @brief Returns a strings column replacing a range of rows
 * with the specified string.
 *
 * If the value parameter is invalid, the specified rows are filled with
 * null entries.
 *
 * @throw cudf::logic_error if [begin,end) is outside the range of the input column.
 *
 * @param strings Strings column to fill.
 * @param begin First row index to include the new string.
 * @param end Last row index (exclusive).
 * @param value String to use when filling the range.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use for any kernels in this function.
 * @return New strings column.
 */
std::unique_ptr<column> fill( strings_column_view const& strings,
                              size_type begin, size_type end,
                              string_scalar const& value,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream = 0 );

} // namespace detail
} // namespace strings
} // namespace cudf
