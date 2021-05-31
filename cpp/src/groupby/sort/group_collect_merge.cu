/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

// TODO: Reorganize
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/combine.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby {
namespace detail {
std::pair<std::unique_ptr<table>, std::unique_ptr<column>> group_collect_merge(
  host_span<table_view const> agg_keys,
  host_span<column_view const> agg_results,
  null_policy null_handling,
  std::vector<order> column_order,
  std::vector<null_order> null_precedence,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr)
{
  auto const default_mr = rmm::mr::get_current_device_resource();

  //
  // Warning: Below is just a workaround.
  // This is a more efficient merging API that is WIP by Dave Baranec, and will be adopted here
  // after finished.
  //

  // Vertically merge all keys tables and result columns.
  auto const keys    = concatenate(agg_keys);
  auto const results = concatenate(agg_results);

  // Perform groupby on the merged keys and results
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = results->view();
  requests[0].aggregations.emplace_back(make_collect_list_aggregation());

  auto gb_obj = groupby(keys->view(), null_handling, sorted::NO, column_order, null_precedence);
  auto result = gb_obj.aggregate(requests, mr);
  auto merged_lists =
    lists::concatenate_list_elements(result.second.front().results.front()->view());
  return {std::move(result.first), std::move(merged_lists)};
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
