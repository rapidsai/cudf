/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Return validity of a row
 *
 * Retrieves the validity (NULL or non-NULL) of the specified row from device memory.
 *
 * @note Synchronizes `stream`.
 *
 * @throw cudf::logic_error if `element_index < 0 or >= col_view.size()`
 *
 * @param col_view The column to retrieve the validity from.
 * @param element_index The index of the row to retrieve.
 * @param stream The stream to use for copying the validity to the host.
 * @return Host boolean that indicates the validity of the row.
 */

bool is_element_valid_sync(column_view const& col_view,
                           size_type element_index,
                           rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
