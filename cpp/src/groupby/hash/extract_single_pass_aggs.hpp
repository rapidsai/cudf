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
 * @brief Extract all single-pass aggregations.
 *
 * @return A tuple containing:
 *         - A table_view containing the input values columns for the single-pass aggregations,
 *         - A vector of aggregation kinds corresponding to each of these values columns,
 *         - A vector of aggregation objects corresponding to each of these values columns, and
 *         - A boolean value indicating if there are any multi-pass aggregations.
 */
std::tuple<table_view,
           cudf::detail::host_vector<aggregation::Kind>,
           std::vector<std::unique_ptr<aggregation>>,
           bool>
extract_single_pass_aggs(host_span<aggregation_request const> requests,
                         rmm::cuda_stream_view stream);

}  // namespace cudf::groupby::detail::hash
