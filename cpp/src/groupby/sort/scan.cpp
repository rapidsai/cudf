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
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/sorting.hpp>
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
  // TODO review: this denotes key sorted, but the values are not sorted.
  // CUDF_EXPECTS(helper.is_presorted(),
  //              "Rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(values),
               "Unsupported list type in grouped rank scan.");
  auto const& rank_agg         = dynamic_cast<cudf::detail::rank_aggregation const&>(agg);
  auto const& group_labels     = helper.group_labels(stream);
  auto const group_labels_view = column_view(cudf::device_span<const size_type>(group_labels));
  // TODO pct percentage
  auto const gather_map =
    (rank_agg._method == rank_method::FIRST
       ? cudf::detail::stable_sorted_order
       : cudf::detail::sorted_order)(table_view({group_labels_view, get_grouped_values()}),
                                     {order::ASCENDING, rank_agg._column_order},
                                     {null_order::AFTER, rank_agg._null_precedence},
                                     stream,
                                     mr);

  auto rank_scan = [&]() {
    if (rank_agg._method == rank_method::MIN) {
      return detail::min_rank_scan;
    } else if (rank_agg._method == rank_method::MAX) {
      return detail::max_rank_scan;
    } else if (rank_agg._method == rank_method::FIRST) {
      return detail::first_rank_scan;
    } else if (rank_agg._method == rank_method::DENSE) {
      return detail::dense_rank_scan;
    } else if (rank_agg._method == rank_method::AVERAGE) {
      return detail::average_rank_scan;
    } else {
      CUDF_FAIL("Unsupported rank method in groupby scan");
    }
  }();
  auto result = rank_scan(get_grouped_values(),
                          *gather_map,
                          helper.group_labels(stream),
                          helper.group_offsets(stream),
                          stream,
                          mr);
  cache.add_result(
    values,
    agg,
    std::move(cudf::detail::scatter(
                table_view{{*result}}, *gather_map, table_view{{*result}}, false, stream, mr)
                ->release()[0]));
}

template <>
void scan_result_functor::operator()<aggregation::ANSI_SQL_PERCENT_RANK>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;
  // TODO review: this denotes key sorted, but the values are not sorted.
  // CUDF_EXPECTS(helper.is_presorted(),
  //              "Percent rank aggregate in groupby scan requires the keys to be presorted");
  auto rank_min_agg = make_rank_aggregation<groupby_scan_aggregation>(rank_method::MIN);
  operator()<aggregation::RANK>(*rank_min_agg);
  column_view rank_min = cache.get_result(values, *rank_min_agg);

  auto const order_by = get_grouped_values();
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in grouped percent_rank scan.");

  cache.add_result(
    values,
    agg,
    detail::percent_rank_scan(
      rank_min, helper.group_labels(stream), helper.group_offsets(stream), stream, mr));
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

  return std::make_pair(helper().sorted_keys(stream, mr), std::move(results));
}
}  // namespace groupby
}  // namespace cudf
