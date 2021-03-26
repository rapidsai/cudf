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

#include <groupby/common/utils.hpp>
#include <groupby/sort/functors.hpp>
#include <groupby/sort/group_scan.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Functor to dispatch aggregation with
 *
 * This functor is to be used with `aggregation_dispatcher` to compute the
 * appropriate aggregation. If the values on which to run the aggregation are
 * unchanged, then this functor should be re-used. This is because it stores
 * memoised sorted and/or grouped values and re-using will save on computation
 * of these values.
 */
struct scan_result_functor final : store_result_functor {
  using store_result_functor::store_result_functor;
  template <aggregation::Kind k>
  void operator()(aggregation const& agg)
  {
    CUDF_FAIL("Unsupported groupby scan aggregation");
  }

 private:
  column_view get_grouped_values()
  {
    // TODO (dm): After implementing single pass multi-agg, explore making a
    //            cache of all grouped value columns rather than one at a time
    if (grouped_values)
      return grouped_values->view();
    else
      return (grouped_values = helper.grouped_values(values))->view();
  };
};

template <>
void scan_result_functor::operator()<aggregation::SUM>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    detail::sum_scan(get_grouped_values(), helper.num_groups(), helper.group_labels(), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::MIN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    detail::min_scan(get_grouped_values(), helper.num_groups(), helper.group_labels(), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::MAX>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    detail::max_scan(get_grouped_values(), helper.num_groups(), helper.group_labels(), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::COUNT_ALL>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(col_idx, agg, detail::count_scan(helper.group_labels(), stream, mr));
}
}  // namespace detail

// Sort-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::sort_scan(
  host_span<aggregation_request const> requests,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // We're going to start by creating a cache of results so that aggs that
  // depend on other aggs will not have to be recalculated. e.g. mean depends on
  // sum and count. std depends on mean and count
  cudf::detail::result_cache cache(requests.size());

  for (size_t i = 0; i < requests.size(); i++) {
    auto store_functor =
      detail::scan_result_functor(i, requests[i].values, helper(), cache, stream, mr);
    for (auto const& aggregation : requests[i].aggregations) {
      // TODO (dm): single pass compute all supported reductions
      cudf::detail::aggregation_dispatcher(aggregation->kind, store_functor, *aggregation);
    }
  }

  auto results = detail::extract_results(requests, cache);

  return std::make_pair(helper().sorted_keys(stream, mr), std::move(results));
}
}  // namespace groupby
}  // namespace cudf
