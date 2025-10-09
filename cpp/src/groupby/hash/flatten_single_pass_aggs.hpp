/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/groupby.hpp>

#include <memory>
#include <tuple>
#include <vector>

namespace cudf::groupby::detail::hash {

/**
 * @brief Flatten any appropriate compound aggregations into single pass aggs
 *
 * For example, a MEAN aggregation will be flattened into a SUM and a COUNT_VALID aggregation.
 *
 * @param requests The aggregation requests
 * @param stream The CUDA stream
 * @return A tuple containing the flattened table view, the aggregation kinds, and the aggregations
 */
std::tuple<table_view,
           cudf::detail::host_vector<aggregation::Kind>,
           std::vector<std::unique_ptr<aggregation>>>
flatten_single_pass_aggs(host_span<aggregation_request const> requests,
                         rmm::cuda_stream_view stream);

/**
 * @brief Get simple aggregations from groupby aggregation
 *
 * @param agg The groupby aggregation
 * @param values_type The data type for the aggregation
 * @return A vector of aggregation kinds
 */
std::vector<aggregation::Kind> get_simple_aggregations(groupby_aggregation const& agg,
                                                       data_type values_type);
}  // namespace cudf::groupby::detail::hash
