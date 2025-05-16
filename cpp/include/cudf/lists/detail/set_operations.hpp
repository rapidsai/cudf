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

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace lists::detail {

/**
 * @copydoc cudf::list::have_overlap
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> have_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::list::intersect_distinct
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> intersect_distinct(lists_column_view const& lhs,
                                           lists_column_view const& rhs,
                                           null_equality nulls_equal,
                                           nan_equality nans_equal,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::list::union_distinct
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> union_distinct(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::list::difference_distinct
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> difference_distinct(lists_column_view const& lhs,
                                            lists_column_view const& rhs,
                                            null_equality nulls_equal,
                                            nan_equality nans_equal,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/** @} */  // end of group
}  // namespace lists::detail
}  // namespace CUDF_EXPORT cudf
