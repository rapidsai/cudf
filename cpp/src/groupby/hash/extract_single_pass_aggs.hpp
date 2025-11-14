/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 * @brief Extract all compound aggregations into single pass aggs
 *
 * For example, a MEAN aggregation will be flattened into a SUM and a COUNT_VALID aggregation.
 *
 * @param requests The aggregation requests
 * @param stream The CUDA stream
 *
 * @return A tuple containing:
 *         - A table_view containing the input values columns for the single-pass aggregations,
 *         - A vector of aggregation kinds corresponding to each of these values columns,
 *         - A vector of aggregation objects corresponding to each of these values columns, and
 *         - A boolean value indicating if there are any multi-pass (compound) aggregations.
 */
std::tuple<table_view,
           cudf::detail::host_vector<aggregation::Kind>,
           std::vector<std::unique_ptr<aggregation>>,
           bool>
extract_single_pass_aggs(host_span<aggregation_request const> requests,
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
