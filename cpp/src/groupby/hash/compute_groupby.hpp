/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>

namespace cudf::groupby::detail::hash {
/**
 * @brief Computes groupby using hash table.
 *
 * First, we create a hash table that stores the indices of unique rows in
 * `keys`. The upper limit on the number of values in this map is the number
 * of rows in `keys`.
 *
 * To store the results of aggregations, we create temporary sparse columns
 * which have the same size as input value columns. Using the hash map, we
 * determine the location within the sparse column to write the result of the
 * aggregation into.
 *
 * The sparse column results of all aggregations are stored into the cache
 * `sparse_results`. This enables the use of previously calculated results in
 * other aggregations.
 *
 * All the aggregations which can be computed in a single pass are computed
 * first, in a combined kernel. Then using these results, aggregations that
 * require multiple passes, will be computed.
 *
 * Finally, using the hash map, we generate a vector of indices of populated
 * values in sparse result columns. Then, for each aggregation originally
 * requested in `requests`, we gather sparse results into a column of dense
 * results using the aforementioned index vector. Dense results are stored into
 * the in/out parameter `cache`.
 *
 * @tparam Equal Device row comparator type
 * @tparam Hash Device row hasher type
 *
 * @param keys Table whose rows act as the groupby keys
 * @param requests The set of columns to aggregate and the aggregations to perform
 * @param skip_rows_with_nulls Flag indicating whether to ignore nulls or not
 * @param d_row_equal Device row comparator
 * @param d_row_hash Device row hasher
 * @param cache Dense aggregation results
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 * @return Table of unique keys
 */
template <typename Equal, typename Hash>
std::unique_ptr<cudf::table> compute_groupby(table_view const& keys,
                                             host_span<aggregation_request const> requests,
                                             bool skip_rows_with_nulls,
                                             Equal const& d_row_equal,
                                             Hash const& d_row_hash,
                                             cudf::detail::result_cache* cache,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
