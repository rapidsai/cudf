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


namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Returns a new strings column created from a subset of
 * of the strings column. The subset of strings selected is between
 * start (inclusive) and end (exclusive) with incrememnts of step.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * s2 = slice( s1, 2 )
 * s2 is ["c", "d", "e", "f"]
 * s3 = slice( s1, 1, 2 )
 * s3 is ["b", "d", "f"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param start Index to first string to select in the column (inclusive).
 * @param end Index to last string to select in the column (exclusive).
 *            Default -1 indicates the last element.
 * @param step Increment value between indices.
 *             Default step is 1.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New strings column of size (end-start)/step.
 */
std::unique_ptr<cudf::column> slice( strings_column_view const& strings,
                                     size_type start, size_type end=-1,
                                     size_type step=1,
                                     cudaStream_t stream=0,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map )
 * s2 is ["a", "c"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param gather_map The indices with which to select strings for the new column.
 *        Values must be within [0,size()) range.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New strings column of size indices.size()
 */
std::unique_ptr<cudf::column> gather( strings_column_view const& strings,
                                      cudf::column_view gather_map,
                                      cudaStream_t stream=0,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );


} // namespace detail
} // namespace strings
} // namespace cudf
