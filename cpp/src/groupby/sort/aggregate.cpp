/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <groupby/sort/group_reductions.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/detail/drop_list_duplicates.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <unordered_map>
#include <utility>

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
struct aggregate_result_functor final : store_result_functor {
  using store_result_functor::store_result_functor;
  template <aggregation::Kind k>
  void operator()(aggregation const& agg)
  {
    CUDF_FAIL("Unsupported aggregation.");
  }
};

template <>
void aggregate_result_functor::operator()<aggregation::COUNT_VALID>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    get_grouped_values().nullable()
      ? detail::group_count_valid(
          get_grouped_values(), helper.group_labels(stream), helper.num_groups(stream), stream, mr)
      : detail::group_count_all(
          helper.group_offsets(stream), helper.num_groups(stream), stream, mr));
}

template <>
void aggregate_result_functor::operator()<aggregation::COUNT_ALL>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    detail::group_count_all(helper.group_offsets(stream), helper.num_groups(stream), stream, mr));
}

template <>
void aggregate_result_functor::operator()<aggregation::SUM>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    detail::group_sum(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
};

template <>
void aggregate_result_functor::operator()<aggregation::PRODUCT>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    detail::group_product(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
};

template <>
void aggregate_result_functor::operator()<aggregation::ARGMAX>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(col_idx,
                   agg,
                   detail::group_argmax(get_grouped_values(),
                                        helper.num_groups(stream),
                                        helper.group_labels(stream),
                                        helper.key_sort_order(stream),
                                        stream,
                                        mr));
};

template <>
void aggregate_result_functor::operator()<aggregation::ARGMIN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(col_idx,
                   agg,
                   detail::group_argmin(get_grouped_values(),
                                        helper.num_groups(stream),
                                        helper.group_labels(stream),
                                        helper.key_sort_order(stream),
                                        stream,
                                        mr));
};

template <>
void aggregate_result_functor::operator()<aggregation::MIN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto result = [&]() {
    if (cudf::is_fixed_width(values.type())) {
      return detail::group_min(
        get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr);
    } else {
      auto argmin_agg = make_argmin_aggregation();
      operator()<aggregation::ARGMIN>(*argmin_agg);
      column_view argmin_result = cache.get_result(col_idx, *argmin_agg);

      // We make a view of ARGMIN result without a null mask and gather using
      // this mask. The values in data buffer of ARGMIN result corresponding
      // to null values was initialized to ARGMIN_SENTINEL which is an out of
      // bounds index value and causes the gathered value to be null.
      column_view null_removed_map(
        data_type(type_to_id<size_type>()),
        argmin_result.size(),
        static_cast<void const*>(argmin_result.template data<size_type>()));
      auto transformed_result =
        cudf::detail::gather(table_view({values}),
                             null_removed_map,
                             argmin_result.nullable() ? cudf::out_of_bounds_policy::NULLIFY
                                                      : cudf::out_of_bounds_policy::DONT_CHECK,
                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                             stream,
                             mr);
      return std::move(transformed_result->release()[0]);
    }
  }();

  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::MAX>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto result = [&]() {
    if (cudf::is_fixed_width(values.type())) {
      return detail::group_max(
        get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr);
    } else {
      auto argmax_agg = make_argmax_aggregation();
      operator()<aggregation::ARGMAX>(*argmax_agg);
      column_view argmax_result = cache.get_result(col_idx, *argmax_agg);

      // We make a view of ARGMAX result without a null mask and gather using
      // this mask. The values in data buffer of ARGMAX result corresponding
      // to null values was initialized to ARGMAX_SENTINEL which is an out of
      // bounds index value and causes the gathered value to be null.
      column_view null_removed_map(
        data_type(type_to_id<size_type>()),
        argmax_result.size(),
        static_cast<void const*>(argmax_result.template data<size_type>()));
      auto transformed_result =
        cudf::detail::gather(table_view({values}),
                             null_removed_map,
                             argmax_result.nullable() ? cudf::out_of_bounds_policy::NULLIFY
                                                      : cudf::out_of_bounds_policy::DONT_CHECK,
                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                             stream,
                             mr);
      return std::move(transformed_result->release()[0]);
    }
  }();

  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::MEAN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto sum_agg   = make_sum_aggregation();
  auto count_agg = make_count_aggregation();
  operator()<aggregation::SUM>(*sum_agg);
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view sum_result   = cache.get_result(col_idx, *sum_agg);
  column_view count_result = cache.get_result(col_idx, *count_agg);

  // TODO (dm): Special case for timestamp. Add target_type_impl for it.
  //            Blocked until we support operator+ on timestamps
  auto result =
    cudf::detail::binary_operation(sum_result,
                                   count_result,
                                   binary_operator::DIV,
                                   cudf::detail::target_type(values.type(), aggregation::MEAN),
                                   stream,
                                   mr);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::VARIANCE>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto var_agg   = dynamic_cast<cudf::detail::var_aggregation const&>(agg);
  auto mean_agg  = make_mean_aggregation();
  auto count_agg = make_count_aggregation();
  operator()<aggregation::MEAN>(*mean_agg);
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view mean_result = cache.get_result(col_idx, *mean_agg);
  column_view group_sizes = cache.get_result(col_idx, *count_agg);

  auto result = detail::group_var(get_grouped_values(),
                                  mean_result,
                                  group_sizes,
                                  helper.group_labels(stream),
                                  var_agg._ddof,
                                  stream,
                                  mr);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::STD>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto std_agg = dynamic_cast<cudf::detail::std_aggregation const&>(agg);
  auto var_agg = make_variance_aggregation(std_agg._ddof);
  operator()<aggregation::VARIANCE>(*var_agg);
  column_view var_result = cache.get_result(col_idx, *var_agg);

  auto result = cudf::detail::unary_operation(var_result, unary_operator::SQRT, stream, mr);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::QUANTILE>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto count_agg = make_count_aggregation();
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view group_sizes = cache.get_result(col_idx, *count_agg);
  auto quantile_agg       = dynamic_cast<cudf::detail::quantile_aggregation const&>(agg);

  auto result = detail::group_quantiles(get_sorted_values(),
                                        group_sizes,
                                        helper.group_offsets(stream),
                                        helper.num_groups(stream),
                                        quantile_agg._quantiles,
                                        quantile_agg._interpolation,
                                        stream,
                                        mr);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::MEDIAN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto count_agg = make_count_aggregation();
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view group_sizes = cache.get_result(col_idx, *count_agg);

  auto result = detail::group_quantiles(get_sorted_values(),
                                        group_sizes,
                                        helper.group_offsets(stream),
                                        helper.num_groups(stream),
                                        {0.5},
                                        interpolation::LINEAR,
                                        stream,
                                        mr);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::NUNIQUE>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto nunique_agg = dynamic_cast<cudf::detail::nunique_aggregation const&>(agg);

  auto result = detail::group_nunique(get_sorted_values(),
                                      helper.group_labels(stream),
                                      helper.num_groups(stream),
                                      helper.group_offsets(stream),
                                      nunique_agg._null_handling,
                                      stream,
                                      mr);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::NTH_ELEMENT>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto nth_element_agg = dynamic_cast<cudf::detail::nth_element_aggregation const&>(agg);

  auto count_agg = make_count_aggregation(nth_element_agg._null_handling);
  if (count_agg->kind == aggregation::COUNT_VALID) {
    operator()<aggregation::COUNT_VALID>(*count_agg);
  } else if (count_agg->kind == aggregation::COUNT_ALL) {
    operator()<aggregation::COUNT_ALL>(*count_agg);
  } else {
    CUDF_FAIL("Wrong count aggregation kind");
  }
  column_view group_sizes = cache.get_result(col_idx, *count_agg);

  cache.add_result(col_idx,
                   agg,
                   detail::group_nth_element(get_grouped_values(),
                                             group_sizes,
                                             helper.group_labels(stream),
                                             helper.group_offsets(stream),
                                             helper.num_groups(stream),
                                             nth_element_agg._n,
                                             nth_element_agg._null_handling,
                                             stream,
                                             mr));
}

template <>
void aggregate_result_functor::operator()<aggregation::COLLECT_LIST>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) { return; }

  auto const null_handling =
    dynamic_cast<cudf::detail::collect_list_aggregation const&>(agg)._null_handling;
  auto result = detail::group_collect(get_grouped_values(),
                                      helper.group_offsets(stream),
                                      helper.num_groups(stream),
                                      null_handling,
                                      stream,
                                      mr);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void aggregate_result_functor::operator()<aggregation::COLLECT_SET>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) { return; }

  auto const null_handling =
    dynamic_cast<cudf::detail::collect_set_aggregation const&>(agg)._null_handling;
  auto const collect_result = detail::group_collect(get_grouped_values(),
                                                    helper.group_offsets(stream),
                                                    helper.num_groups(stream),
                                                    null_handling,
                                                    stream,
                                                    rmm::mr::get_current_device_resource());
  auto const nulls_equal =
    dynamic_cast<cudf::detail::collect_set_aggregation const&>(agg)._nulls_equal;
  auto const nans_equal =
    dynamic_cast<cudf::detail::collect_set_aggregation const&>(agg)._nans_equal;
  cache.add_result(
    col_idx,
    agg,
    lists::detail::drop_list_duplicates(
      lists_column_view(collect_result->view()), nulls_equal, nans_equal, stream, mr));
};

/**
 * @brief Perform merging for the lists that correspond to the same key value.
 *
 * This aggregation is similar to `COLLECT_LIST` with the following differences:
 *  - It requires the input values to be a non-nullable lists column, and
 *  - The values (lists) corresponding to the same key will not result in a list of lists as output
 *    from `COLLECT_LIST`. Instead, those lists will result in a list generated by merging them
 *    together.
 *
 * In practice, this aggregation is used to merge the partial results of multiple (distributed)
 * groupby `COLLECT_LIST` aggregations into a final `COLLECT_LIST` result. Those distributed
 * aggregations were executed on different values columns partitioned from the original values
 * column, then their results were (vertically) concatenated before given as the values column for
 * this aggregation.
 */
template <>
void aggregate_result_functor::operator()<aggregation::MERGE_LISTS>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) { return; }

  cache.add_result(
    col_idx,
    agg,
    detail::group_merge_lists(
      get_grouped_values(), helper.group_offsets(stream), helper.num_groups(stream), stream, mr));
};

/**
 * @brief Perform merging for the lists corresponding to the same key value, then dropping duplicate
 * list entries.
 *
 * This aggregation is similar to `COLLECT_SET` with the following differences:
 *  - It requires the input values to be a non-nullable lists column, and
 *  - The values (lists) corresponding to the same key will result in a list generated by merging
 *    them together then dropping duplicate entries.
 *
 * In practice, this aggregation is used to merge the partial results of multiple (distributed)
 * groupby `COLLECT_LIST` or `COLLECT_SET` aggregations into a final `COLLECT_SET` result. Those
 * distributed aggregations were executed on different values columns partitioned from the original
 * values column, then their results were (vertically) concatenated before given as the values
 * column for this aggregation.
 *
 * Firstly, this aggregation performs `MERGE_LISTS` to concatenate the input lists (corresponding to
 * the same key) into intermediate lists, then it calls `lists::drop_list_duplicates` on them to
 * remove duplicate list entries. As such, the input (partial results) to this aggregation should be
 * generated by (distributed) `COLLECT_LIST` aggregations, not `COLLECT_SET`, to avoid unnecessarily
 * removing duplicate entries for the partial results.
 *
 * Since duplicate list entries will be removed, the parameters `null_equality` and `nan_equality`
 * are needed for calling to `lists::drop_list_duplicates`.
 */
template <>
void aggregate_result_functor::operator()<aggregation::MERGE_SETS>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) { return; }

  auto const merged_result  = detail::group_merge_lists(get_grouped_values(),
                                                       helper.group_offsets(stream),
                                                       helper.num_groups(stream),
                                                       stream,
                                                       rmm::mr::get_current_device_resource());
  auto const merge_sets_agg = dynamic_cast<cudf::detail::merge_sets_aggregation const&>(agg);
  cache.add_result(col_idx,
                   agg,
                   lists::detail::drop_list_duplicates(lists_column_view(merged_result->view()),
                                                       merge_sets_agg._nulls_equal,
                                                       merge_sets_agg._nans_equal,
                                                       stream,
                                                       mr));
};

}  // namespace detail

// Sort-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::sort_aggregate(
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
      detail::aggregate_result_functor(i, requests[i].values, helper(), cache, stream, mr);
    for (size_t j = 0; j < requests[i].aggregations.size(); j++) {
      // TODO (dm): single pass compute all supported reductions
      cudf::detail::aggregation_dispatcher(
        requests[i].aggregations[j]->kind, store_functor, *requests[i].aggregations[j]);
    }
  }

  auto results = detail::extract_results(requests, cache);

  return std::make_pair(helper().unique_keys(stream, mr), std::move(results));
}
}  // namespace groupby
}  // namespace cudf
