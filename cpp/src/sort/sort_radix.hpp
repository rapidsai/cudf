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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Check if radix sort is available for the given column
 *
 * @param column The column to check
 * @return true if radix sort is available, false otherwise
 */
bool is_radix_sortable(column_view const& column);

/**
 * @brief Sort a column using radix sort
 *
 * This should only be used for fixed-width columns with no nulls.
 *
 * @param input The column to sort
 * @param ascending The sort order
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 * @return The sorted column
 */
std::unique_ptr<column> sort_radix(column_view const& input,
                                   bool ascending,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @brief Sort a column using radix sort
 *
 * This should only be used for fixed-width columns with no nulls.
 *
 * @param input The column to sort
 * @param indices The indices to return
 * @param ascending The sort order
 * @param stream The CUDA stream to use
 */
void sorted_order_radix(column_view const& input,
                        mutable_column_view& indices,
                        bool ascending,
                        rmm::cuda_stream_view stream);
}  // namespace detail
}  // namespace cudf
