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
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtx3/nvToolsExt.h>

#include <vector>

/**
 */
std::unique_ptr<cudf::table> compute_results(cudf::column_view const& cities,
                                             cudf::column_view const& temperatures)
{
  auto groupby_obj      = cudf::groupby::groupby(cudf::table_view({cities}));
  auto aggregation_reqs = std::vector<cudf::groupby::aggregation_request>{};
  auto& req             = aggregation_reqs.emplace_back();
  req.values            = temperatures;
  req.aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  req.aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  req.aggregations.emplace_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  req.aggregations.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  req.aggregations.emplace_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());

  nvtxRangePushA("aggregate");
  auto result = groupby_obj.aggregate(aggregation_reqs);
  cudaStreamSynchronize(0);
  nvtxRangePop();

  auto rtn = result.first->release();
  rtn.emplace_back(std::move(result.second.front().results[0]));
  rtn.emplace_back(std::move(result.second.front().results[1]));
  rtn.emplace_back(std::move(result.second.front().results[2]));
  rtn.emplace_back(std::move(result.second.front().results[3]));
  rtn.emplace_back(std::move(result.second.front().results[4]));

  return std::make_unique<cudf::table>(std::move(rtn));
}
