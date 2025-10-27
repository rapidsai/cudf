/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

/**
 * @brief Process the cities and temperatures
 *
 * Perform the given aggregations using the cities as the keys and the
 * temperatures as values.
 *
 * @param cities The city names
 * @param temperatures The temperature values
 * @param aggregations Which groupby aggregations to perform
 * @param stream CUDA stream to use for launching kernels
 * @return aggregated results
 */
std::unique_ptr<cudf::table> compute_results(
  cudf::column_view const& cities,
  cudf::column_view const& temperatures,
  std::vector<std::unique_ptr<cudf::groupby_aggregation>>&& aggregations,
  rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Produce the final aggregations from sub-aggregate results
 *
 * @param agg_data Sub-aggregations to summarize
 * @param stream CUDA stream to use for launching kernels
 * @return final results
 */
std::unique_ptr<cudf::table> compute_final_aggregates(
  std::vector<std::unique_ptr<cudf::table>>& agg_data,
  rmm::cuda_stream_view stream = cudf::get_default_stream());
