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

#include <cudf/filling.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime.h>

#include <memory>

namespace cudf {
namespace experimental {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Internal API to fill a range of elements in-place in a column with a
 * scalar value.
 * 
 * Fills N elements of @p destination starting at @p begin with @p value, where
 * N = (@p end - @p begin).
 *
 * Overwrites the range of elements in @p destination indicated by the indices
 * [@p begin, @p end) with @p value. Use the out-of-place fill function
 * returning std::unique_ptr<column> for use cases requiring memory
 * reallocation.
 *
 * @throws `cudf::logic_error` if memory reallocation is required (e.g. for
 * variable width types).
 * @throws `cudf::logic_error` for invalid range (if @p begin < 0,
 * @p begin > @p end, @p begin >= @p destination.size(), or
 * @p end > @p destination.size()).
 * @throws `cudf::logic_error` if @p destination and @p value have different
 * types.
 * @throws `cudf::logic_error` if @p value is invalid but @p destination is not
 * nullable.
 *
 * @param destination The preallocated column to fill into
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @param stream CUDA stream to run this function
 * @return void
 *---------------------------------------------------------------------------**/
void fill(mutable_column_view& destination, size_type begin, size_type end,
          scalar const& value, cudaStream_t stream = 0);

/**---------------------------------------------------------------------------*
 * @brief Internal API to fill a range of elements in a column out-of-place with
 a scalar value.
 * 
 * Creates a new column as-if an in-place fill was performed into @p input;
 * i.e. it is as if a copy of @p input was created first and then the elements
 * indicated by the indices [@p begin, @p end) were overwritten by @p value.
 *
 * @throws `cudf::logic_error` for invalid range (if @p begin < 0,
 * @p begin > @p end, @p begin >= @p destination.size(), or
 * @p end > @p destination.size()).
 * @throws `cudf::logic_error` if @p destination and @p value have different
 * types.
 *
 * @param input The input column used to create a new column. The new column
 * is created by replacing the values of @p input in the specified range with
 * @p value.
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @param mr Memory resource to allocate the result output column
 * @param stream CUDA stream to run this function
 * @return std::unique_ptr<column> The result output column
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> fill(
    column_view const& input, size_type begin, size_type end,
    scalar const& value,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
                            
}  // namespace detail
}  // namespace experimental
}  // namespace cudf
