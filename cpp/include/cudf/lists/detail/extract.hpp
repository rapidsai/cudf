/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/lists/extract.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view, size_type,
 * rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             size_type const index,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view, column_view const&,
 * rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             column_view const& indices,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace CUDF_EXPORT cudf
