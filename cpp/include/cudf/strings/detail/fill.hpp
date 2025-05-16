/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace strings::detail {
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
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> fill(strings_column_view const& strings,
                             size_type begin,
                             size_type end,
                             string_scalar const& value,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace CUDF_EXPORT cudf
