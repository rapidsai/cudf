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

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> slice( strings_column_view strings,
                                     size_type start, size_type end=-1,
                                     size_type step=1,
                                     cudaStream_t stream=0,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> gather( strings_column_view strings,
                                      cudf::column_view gather_map,
                                      cudaStream_t stream=0,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**
 * @brief Creates a new strings column as if an in-place scatter from the source
 * `strings` column was performed on that column.
 *
 * The size of the `values` must match the size of the `scatter_map`.
 * The values in `scatter_map` must be in the range [0,strings.size()).
 *
 * If the same index appears more than once in `scatter_map` the result is undefined.
 *
 * This operation basically pre-fills the output column with elements from `strings`. 
 * Then, for each value `i` in range `[0,values.size())`, the `values[i]` element is
 * assigned to `output[scatter_map[i]]`.
 *
 * The output column will have null entries at the `scatter_map` indices if the
 * `values` element at the corresponding entry is null. Also, any values from the
 * original `strings` that are null and not included in the `scatter_map` will
 * remain null.
 *
 * ```
 * s1 = ["a", "b", "c", "d"]
 * s2 = ["e", "f"]
 * map = [1, 3]
 * s3 = scatter( s1, s2, m1 )
 * s3 is ["a", "e", "c", "f"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param values The instance for which to retrieve the strings
 *        specified in map column.
 * @param scatter_map The 0-based index values to retrieve from the
 *        strings parameter. Number of values must equal the number
 *        of elements in values pararameter: `values.size()`.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       strings_column_view values,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
/**
 * @brief Creates a new strings column as if an in-place scatter from the source
 * `strings` column was performed using the same string `value` for each index
 * in `scatter_map`.
 *
 * If the same index appears more than once in `scatter_map` the result is undefined.
 *
 * The values in `scatter_map` must be in the range [0,strings.size()).
 * 
 * This operation basically pre-fills the output column with elements from `strings`. 
 * Then, for each value `i` in range `[0,values.size())`, the `value` element is
 * assigned to `output[scatter_map[i]]`.
 *
 * The output column will have null entries at the `scatter_map` indices if the
 * `value` parameter is null. Also, any values from the original `strings` that
 * are null and not included in the `scatter_map` will remain null.
 *
 * ```
 * s1 = ["a", "b", "c", "d"]
 * map = [1, 3]
 * s2 = scatter( s1, "e", m1 )
 * s2 is ["a", "e", "c", "e"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param value Null-terminated encoded string in host memory to use with
 *        the scatter_map. Pass nullptr to specify a null entry be created.
 * @param scatter_map The 0-based index values to place the given string.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       const char* value,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

} // namespace detail
} // namespace strings
} // namespace cudf
