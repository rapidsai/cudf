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

#ifndef _AGGREGATION_REQUESTS_HPP_
#define _AGGREGATION_REQUESTS_HPP_

#include <cudf/cudf.h>
#include <cudf/groupby.hpp>

#include <utility>
#include <vector>

namespace cudf {
namespace groupby {
namespace hash {

// Forward declaration
using cudaStream_t = struct CUstream_st*;

/**---------------------------------------------------------------------------*
 * @file aggregations_requests.hpp
 * @brief An "aggregation request" is the pairing of a `gdf_column*` and an
 * aggregation operation to be performed on that column.
 *
 * A "compound" aggregation request is a requested aggregation operation that
 * can only be satisfied by first computing 1 or more "simple" aggregation
 * requests, and then transforming the result of the simple aggregation request
 * into the requested compound aggregation.
 *
 * For example, `MEAN` is a "compound" aggregation. The request to compute an MEAN
 * on a column can be satisfied via the `COUNT` and `SUM` "simple" aggregation
 * operations.
 *---------------------------------------------------------------------------**/
using AggRequestType = std::pair<gdf_column*, operators>;

/**---------------------------------------------------------------------------*
 * @brief An "aggregation counter" is the pairing of a `AggRequestType` and a
 * counter of how many times said operation is needed.
 *---------------------------------------------------------------------------**/
using SimpleAggRequestCounter = std::pair<AggRequestType, gdf_size_type>;

/**---------------------------------------------------------------------------*
 * @brief Converts a set of "compound" aggregation requests into a set of
 *"simple" aggregation requests that can be used to satisfy the compound
 *request.
 *
 * @param compound_requests The set of compound aggregation requests
 * @return The set of corresponding simple aggregation requests that can be used
 * to satisfy the compound requests
 *---------------------------------------------------------------------------**/
std::vector<SimpleAggRequestCounter> compound_to_simple(
    std::vector<AggRequestType> const& compound_requests);

/**---------------------------------------------------------------------------*
 * @brief Computes the `MEAN` aggregation of a column by doing element-wise
 * division of the corresponding `SUM` and `COUNT` aggregation columns.
 *
 * @param sum The result of a `SUM` aggregation request
 * @param count The result of a `COUNT` aggregation request
 * @param stream Stream on which to perform operation
 * @return gdf_column* New column containing the result of elementwise division
 * of the sum and count columns
 *---------------------------------------------------------------------------**/
gdf_column* compute_average(gdf_column sum, gdf_column count,
                            cudaStream_t stream);

/**---------------------------------------------------------------------------*
 * @brief Computes the results of a set of aggregation requests from a set of
 * computed simple requests.
 *
 * Given a set of pre-computed results for simple aggregation requests, computes
 * the results of a set of (potentially compound) requests. If the simple
 * aggregation request neccessary to compute the original request is not
 * present, a `cudf::logic_error` exception is thrown.
 *
 * @param original_requests[in] The original set of potentially compound
 * aggregation requests
 * @param simple_requests[in] Set of simple requests generated from the original
 * requests
 * @param simple_outputs[in] Set of output aggregation columns corresponding to
 *the simple requests
 * @param stream[in] CUDA stream on which to execute
 * @return table Set of columns satisfying each of the original requests
 *---------------------------------------------------------------------------**/
table compute_original_requests(
    std::vector<AggRequestType> const& original_requests,
    std::vector<SimpleAggRequestCounter> const& simple_requests, table simple_outputs,
    cudaStream_t stream);

}  // namespace hash
}  // namespace groupby
}  // namespace cudf

#endif
