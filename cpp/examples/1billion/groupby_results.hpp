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

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

/**
 */
std::unique_ptr<cudf::table> compute_results(
  cudf::column_view const& cities,
  cudf::column_view const& temperatures,
  std::vector<std::unique_ptr<cudf::groupby_aggregation>>&& aggregations,
  rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 */
std::unique_ptr<cudf::table> compute_final_aggregates(
  std::vector<std::unique_ptr<cudf::table>>& agg_data,
  rmm::cuda_stream_view stream = cudf::get_default_stream());
