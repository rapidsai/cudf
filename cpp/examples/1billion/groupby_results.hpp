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
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

/**
 */
std::unique_ptr<cudf::table> compute_results(
  cudf::column_view const& cities,
  cudf::column_view const& temperatures,
  std::vector<std::unique_ptr<cudf::groupby_aggregation>>&& aggregations,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  auto groupby_obj      = cudf::groupby::groupby(cudf::table_view({cities}));
  auto aggregation_reqs = std::vector<cudf::groupby::aggregation_request>{};
  auto& req             = aggregation_reqs.emplace_back();
  req.values            = temperatures;
  req.aggregations      = std::move(aggregations);

  auto result = groupby_obj.aggregate(aggregation_reqs);

  auto rtn = result.first->release();
  for (auto& r : result.second.front().results) {
    rtn.emplace_back(std::move(r));
  }

  return std::make_unique<cudf::table>(std::move(rtn));
}

/**
 */
std::unique_ptr<cudf::table> compute_final_aggregates(
  std::vector<std::unique_ptr<cudf::table>>& agg_data,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  // first combine all the results into tables (vectors of columns)
  std::vector<cudf::column_view> min_cols, max_cols, sum_cols, count_cols;
  for (auto& tbl : agg_data) {
    auto const tv = tbl->view();
    min_cols.push_back(tv.column(1));
    max_cols.push_back(tv.column(2));
    sum_cols.push_back(tv.column(3));
    count_cols.push_back(tv.column(4));
  }

  // Create single columns out of the aggregate table results.
  // This relies on every key appearing in every chunk segment.
  // All the values for each key become contiguous within the output column.
  auto mins   = cudf::interleave_columns(cudf::table_view{min_cols});
  auto maxes  = cudf::interleave_columns(cudf::table_view{max_cols});
  auto sums   = cudf::interleave_columns(cudf::table_view{sum_cols});
  auto counts = cudf::interleave_columns(cudf::table_view{count_cols});

  // build the offsets for segmented reduce
  auto const num_keys = agg_data.front()->num_rows();
  auto seg_offsets =
    cudf::sequence(static_cast<cudf::size_type>(num_keys) + 1,
                   cudf::numeric_scalar<cudf::size_type>(0, true, stream),
                   cudf::numeric_scalar<cudf::size_type>(agg_data.size(), true, stream),
                   stream);
  auto offsets_span = cudf::device_span<cudf::size_type const>(seg_offsets->view());

  // compute the min/max for each key by doing a segmented reduce
  auto min_agg = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  mins         = cudf::segmented_reduce(
    mins->view(), offsets_span, *min_agg, mins->type(), cudf::null_policy::EXCLUDE, stream);
  auto max_agg = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  maxes        = cudf::segmented_reduce(
    maxes->view(), offsets_span, *max_agg, maxes->type(), cudf::null_policy::EXCLUDE, stream);

  // compute the sum and total counts in the same way
  auto sum_agg = cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
  sums         = cudf::segmented_reduce(
    sums->view(), offsets_span, *sum_agg, sums->type(), cudf::null_policy::EXCLUDE, stream);
  counts = cudf::segmented_reduce(
    counts->view(), offsets_span, *sum_agg, counts->type(), cudf::null_policy::EXCLUDE, stream);

  // compute the means using binary-operation to divide the individual rows sum/count
  auto means = cudf::binary_operation(
    sums->view(), counts->view(), cudf::binary_operator::DIV, sums->type(), stream);

  std::vector<std::unique_ptr<cudf::column>> results;
  results.emplace_back(std::move(mins));
  results.emplace_back(std::move(maxes));
  results.emplace_back(std::move(means));
  return std::make_unique<cudf::table>(std::move(results));
}
