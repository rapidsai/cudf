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

#if 1
#include <cudf/filling_untracked.hpp>
#else
#include <cudf/filling.hpp>
#endif
#include <cudf/types.hpp>
#include <rmm/mr/device_memory_resource.hpp>

// for gdf_scalar, unnecessary once we switch to cudf::scalar
#include <cudf/types.h>

#include <cuda_runtime.h>

#include <memory>

namespace cudf {
namespace experimental {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Internal API to fill a range of elements in a column with a scalar
 * value.
 *
 * Fills N elements of @p destination starting at @p begin with @p value, where
 * N = (@p end - @p begin).
 *
 * The datatypes of @p destination and @p value must be the same. This function
 * assumes that no memory reallocation is necessary for @p destination. This
 * function updates in-place and throws an exception if memory reallocation is
 * necessary (e.g. for variable width types). Use the out-of-place fill function
 * returning std::unique_ptr<column> for use cases requiring memory
 * reallocation.
 *
 * @param destination The preallocated column to fill into
 * @param begin The starting index of the fill range
 * @param end The index one past the end of the fill range
 * @param value The scalar value to fill
 * @return void
 *---------------------------------------------------------------------------**/
void fill(mutable_column_view& destination, size_type begin, size_type end,
          gdf_scalar const& value);

/**---------------------------------------------------------------------------*
 * @brief Internal API to fill a range of elements in a column with a scalar
 * value.
 *
 * This fill function updates out-of-place creating a new column object to
 * return. The returned column holds @p value for N elements from @p begin,
 * where N = (@p end - @p begin). The returned column stores same values to
 * @p input outside the fill range.
 *
 * The datatypes of @p input and @p value must be the same.
 *
 * @param input The input column used to create a new column. The new column
 * is created by replacing the values of @p input in the specified range with
 * @p value.
 * @param begin The starting index of the fill range
 * @param end The index one past the end of the fill range
 * @param value The scalar value to fill
 * @param stream CUDA stream to run this function
 * @param mr Memory resource to allocate the result output column
 * @return std::unique_ptr<column> The result output column
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> fill(
    column_view const& input, size_type begin, size_type end,
    gdf_scalar const& value, cudaStream_t stream,
    rmm::mr::device_memory_resource* mr);
                            
}  // namespace detail
}  // namespace experimental
}  // namespace cudf
