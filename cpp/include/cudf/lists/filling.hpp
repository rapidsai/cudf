/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_filling
 * @{
 * @file
 * @brief Column APIs for individual list sequence
 */

/**
 * @brief Create a lists column in which each row contains a sequence of values specified by a tuple
 * of (`start`, `size`) parameters.
 *
 * Create a lists column in which each row is a sequence of values starting from a `start` value,
 * incrementing by one, and its cardinality is specified by a `size` value. The `start` and `size`
 * values used to generate each list is taken from the corresponding row of the input @p starts and
 * @p sizes columns.
 *
 *  - @p sizes must be a column of integer types.
 *  - All the input columns must not have nulls.
 *  - If any row of the @p sizes column contains negative value, the output is undefined.
 *
 * @code{.pseudo}
 * starts = [0, 1, 2, 3, 4]
 * sizes  = [0, 2, 2, 1, 3]
 *
 * output = [ [], [1, 2], [2, 3], [3], [4, 5, 6] ]
 * @endcode
 *
 * @throws cudf::logic_error if @p sizes column is not of integer types.
 * @throws cudf::logic_error if any input column has nulls.
 * @throws cudf::logic_error if @p starts and @p sizes columns do not have the same size.
 * @throws std::overflow_error if the output column would exceed the column size limit.
 *
 * @param starts First values in the result sequences.
 * @param sizes Numbers of values in the result sequences.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The result column containing generated sequences.
 */
std::unique_ptr<column> sequences(
  column_view const& starts,
  column_view const& sizes,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create a lists column in which each row contains a sequence of values specified by a tuple
 * of (`start`, `step`, `size`) parameters.
 *
 * Create a lists column in which each row is a sequence of values starting from a `start` value,
 * incrementing by a `step` value, and its cardinality is specified by a `size` value. The values
 * `start`, `step`, and `size` used to generate each list is taken from the corresponding row of the
 * input @p starts, @p steps, and @p sizes columns.
 *
 *  - @p sizes must be a column of integer types.
 *  - @p starts and @p steps columns must have the same type.
 *  - All the input columns must not have nulls.
 *  - If any row of the @p sizes column contains negative value, the output is undefined.
 *
 * @code{.pseudo}
 * starts = [0, 1, 2, 3, 4]
 * steps  = [2, 1, 1, 1, -3]
 * sizes  = [0, 2, 2, 1, 3]
 *
 * output = [ [], [1, 2], [2, 3], [3], [4, 1, -2] ]
 * @endcode
 *
 * @throws cudf::logic_error if @p sizes column is not of integer types.
 * @throws cudf::logic_error if any input column has nulls.
 * @throws cudf::logic_error if @p starts and @p steps columns have different types.
 * @throws cudf::logic_error if @p starts, @p steps, and @p sizes columns do not have the same size.
 * @throws std::overflow_error if the output column would exceed the column size limit.
 *
 * @param starts First values in the result sequences.
 * @param steps Increment values for the result sequences.
 * @param sizes Numbers of values in the result sequences.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The result column containing generated sequences.
 */
std::unique_ptr<column> sequences(
  column_view const& starts,
  column_view const& steps,
  column_view const& sizes,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace lists
}  // namespace cudf
