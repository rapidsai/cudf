/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <nanoarrow/nanoarrow.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Utility to handle STRING, LARGE_STRINGS, and STRING_VIEW types
 *
 * @param schema Arrow schema includes the column type
 * @param input Column data, nulls, offset
 * @param mask Mask to apply to the output column
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for all device memory allocations
 */
std::unique_ptr<column> string_column_from_arrow_host(ArrowSchemaView* schema,
                                                      ArrowArray const* input,
                                                      std::unique_ptr<rmm::device_buffer>&& mask,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

/**
 * @brief Convert ArrowArray to cudf column utility
 *
 * This function is simply a convenience wrapper around the dispatch functor with
 * some extra handling to avoid having to reproduce it for all of the nested types.
 * It also allows us to centralize the location where the recursive calls happen
 * so that we only need to forward declare this one function, rather than multiple
 * functions which handle the overloads for nested types (list, struct, etc.)
 *
 * @param schema Arrow schema includes the column type
 * @param input Column data, nulls, offset
 * @param type The cudf column type to map input to
 * @param skip_mask True if the mask is handled by the caller
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for all device memory allocations
 */
std::unique_ptr<column> get_column_copy(ArrowSchemaView* schema,
                                        ArrowArray const* input,
                                        data_type type,
                                        bool skip_mask,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
