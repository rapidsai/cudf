/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/structs/utilities.hpp>
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
    // early exit if presorted
    if (is_presorted()) { return values; }

    // TODO (dm): After implementing single pass multi-agg, explore making a
    //            cache of all grouped value columns rather than one at a time
    if (grouped_values)
      return grouped_values->view();
    else
      return (grouped_values = helper.grouped_values(values, stream))->view();
  };
};

template <>
void scan_result_functor::operator()<aggregation::SUM>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::sum_scan(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::MIN>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::min_scan(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::MAX>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::max_scan(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::COUNT_ALL>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(values, agg, detail::count_scan(helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::RANK>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;
  CUDF_EXPECTS(helper.is_presorted(),
               "Rank aggregate in groupby scan requires the keys to be presorted");
  auto const order_by = get_grouped_values();
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in grouped rank scan.");

  cache.add_result(
    values,
    agg,
    detail::rank_scan(
      order_by, helper.group_labels(stream), helper.group_offsets(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::DENSE_RANK>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;
  CUDF_EXPECTS(helper.is_presorted(),
               "Dense rank aggregate in groupby scan requires the keys to be presorted");
  auto const order_by = get_grouped_values();
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in grouped dense_rank scan.");

  cache.add_result(
    values,
    agg,
    detail::dense_rank_scan(
      order_by, helper.group_labels(stream), helper.group_offsets(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::PERCENT_RANK>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;
  CUDF_EXPECTS(helper.is_presorted(),
               "Percent rank aggregate in groupby scan requires the keys to be presorted");
  auto const order_by = get_grouped_values();
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in grouped percent_rank scan.");

  cache.add_result(
    values,
    agg,
    detail::percent_rank_scan(
      order_by, helper.group_labels(stream), helper.group_offsets(stream), stream, mr));
}
}  // namespace detail

// Sort-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::sort_scan(
  host_span<scan_request const> requests,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // We're going to start by creating a cache of results so that aggs that
  // depend on other aggs will not have to be recalculated. e.g. mean depends on
  // sum and count. std depends on mean and count
  cudf::detail::result_cache cache(requests.size());

  for (auto const& request : requests) {
    auto store_functor =
      detail::scan_result_functor(request.values, helper(), cache, stream, mr, _keys_are_sorted);
    for (auto const& aggregation : request.aggregations) {
      // TODO (dm): single pass compute all supported reductions
      cudf::detail::aggregation_dispatcher(aggregation->kind, store_functor, *aggregation);
    }
  }

  auto results = detail::extract_results(requests, cache, stream, mr);

  return std::pair(helper().sorted_keys(stream, mr), std::move(results));
}
}  // namespace groupby
}  // namespace cudf
