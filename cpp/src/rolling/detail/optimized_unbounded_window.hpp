/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace rmm::mr {
class device_memory_resource;
}

namespace cudf {

class rolling_aggregation;
class table_view;

namespace detail {
/**
 * @brief Checks if it is possible to optimize fully UNBOUNDED window function.
 *
 * @return true if the window aggregation can optimized, i.e. if it is unbounded-preceding,
 * unbounded-following, if it has a supported aggregation type, and if min_periods is 1.
 * @return false if the window aggregation cannot be optimized.
 */
bool can_optimize_unbounded_window(bool unbounded_preceding,
                                   bool unbounded_following,
                                   size_type min_periods,
                                   rolling_aggregation const& agg);

/**
 * @brief Optimized bypass for fully UNBOUNDED window functions.
 *
 * @return the result column from running the unbounded window aggregation,
 * via the optimized aggregation/reduction path.
 */
std::unique_ptr<column> optimized_unbounded_window(table_view const& group_keys,
                                                   column_view const& input,
                                                   rolling_aggregation const& aggr,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr);
}  // namespace detail
}  // namespace cudf
