/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace CUDF_EXPORT cudf {
namespace structs::detail {

/**
 * @brief Returns a single column by concatenating the given vector of structs columns.
 *
 * @code{.pseudo}
 * s1 = [ col0 : {0, 1}
 *        col1 : {2, 3, 4, 5, 6}
 *        col2 : {"abc", "def", "ghi"} ]
 *
 * s2 = [ col0 : {7, 8}
 *        col1 : {-4, -5, -6}
 *        col2 : {"uvw", "xyz"} ]
 *
 * r = concatenate({s1, s2})
 *
 * r is now [ col0: {0, 1, 7, 8}
 *            col1: {2, 3, 4, 5, 6, -4, -5, -6}
 *            col2: {"abc", "def", "ghi", "uvw", "xyz"} ]
 * @endcode
 *
 * @param columns Vector of structs columns to concatenate.
 * @param stream  CUDA stream used for device memory operations and kernel launches.
 * @param mr      Device memory resource used to allocate the returned column's device memory.
 * @return        New column with concatenated results.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace structs::detail
}  // namespace CUDF_EXPORT cudf
