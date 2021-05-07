/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/aggregation/aggregation.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Computes the exclusive scan of a column.
 *
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * The identity value for the column type as per the aggregation type
 * is used for the value of the first element in the output column.
 *
 * @throws cudf::logic_error if column data_type is not an arithmetic type.
 *
 * @param input The input column view for the scan
 * @param agg unique_ptr to aggregation operator applied by the scan
 * @param null_handling Exclude null values when computing the result if
 *                      null_policy::EXCLUDE. Include nulls if null_policy::INCLUDE.
 *                      Any operation with a null results in a null.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Column with scan results
 */
std::unique_ptr<column> scan_exclusive(column_view const& input,
                                       std::unique_ptr<aggregation> const& agg,
                                       null_policy null_handling,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes the inclusive scan of a column.
 *
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * String columns are allowed with aggregation types Min and Max.
 *
 * @throws cudf::logic_error if column data_type is not an arithmetic type
 *                           or string type but the `agg` is not Min or Max
 *
 * @param input The input column view for the scan
 * @param agg unique_ptr to aggregation operator applied by the scan
 * @param null_handling Exclude null values when computing the result if
 *                      null_policy::EXCLUDE. Include nulls if null_policy::INCLUDE.
 *                      Any operation with a null results in a null.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Column with scan results
 */
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       std::unique_ptr<aggregation> const& agg,
                                       null_policy null_handling,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace cudf
