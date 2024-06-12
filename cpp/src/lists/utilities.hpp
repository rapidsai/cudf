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

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf::lists::detail {

/**
 * @brief Generate list labels for elements in the child column of the input lists column.
 *
 * @param input The input lists column
 * @param n_elements The number of elements in the child column of the input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return A column containing list labels corresponding to each element in the child column
 */
std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        size_type n_elements,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @brief Reconstruct an offsets column from the input list labels column.
 *
 * @param labels The list labels corresponding to each list element
 * @param n_lists The number of lists to build the offsets column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return The output offsets column
 */
std::unique_ptr<column> reconstruct_offsets(column_view const& labels,
                                            size_type n_lists,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Generate 0-based list offsets from the offsets of the input lists column.
 *
 * @param input The input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return The output offsets column with values start from 0
 */
std::unique_ptr<column> get_normalized_offsets(lists_column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

}  // namespace cudf::lists::detail
