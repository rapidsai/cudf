/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include "group_reductions.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

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
struct store_result_functor {
  store_result_functor(size_type col_idx,
                       column_view const& values,
                       sort::sort_groupby_helper& helper,
                       cudf::detail::result_cache& cache,
                       cudaStream_t stream,
                       rmm::mr::device_memory_resource* mr)
    : col_idx(col_idx), values(values), helper(helper), cache(cache), stream(stream), mr(mr)
  {
  }

  template <aggregation::Kind k>
  void operator()(aggregation const& agg)
  {
  }

 private:
  /**
   * @brief Get the grouped values
   *
   * Computes the grouped values from @p values on first invocation and returns
   * the stored result on subsequent invocation
   */
  column_view get_grouped_values()
  {
    // TODO (dm): After implementing single pass multi-agg, explore making a
    //            cache of all grouped value columns rather than one at a time
    if (grouped_values)
      return grouped_values->view();
    else if (sorted_values)
      // TODO (dm): When we implement scan, it wouldn't be ok to return sorted
      //            values when asked for grouped values. Change this then.
      return sorted_values->view();
    else
      grouped_values = helper.grouped_values(values);
    return grouped_values->view();
  };

  /**
   * @brief Get the grouped and sorted values
   *
   * Computes the grouped and sorted (within each group) values from @p values
   * on first invocation and returns the stored result on subsequent invocation
   */
  column_view get_sorted_values()
  {
    if (not sorted_values) sorted_values = helper.sorted_values(values);
    return sorted_values->view();
  };

 private:
  size_type col_idx;                  ///< Index of column in requests being operated on
  sort::sort_groupby_helper& helper;  ///< Sort helper
  cudf::detail::result_cache& cache;  ///< cache of results to store into
  column_view const& values;          ///< Column of values to group and aggregate

  cudaStream_t stream;                  ///< CUDA stream on which to execute kernels
  rmm::mr::device_memory_resource* mr;  ///< Memory resource to allocate space for results

  std::unique_ptr<column> sorted_values;   ///< Memoised grouped and sorted values
  std::unique_ptr<column> grouped_values;  ///< Memoised grouped values
};

template <>
void store_result_functor::operator()<aggregation::COUNT_VALID>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx,
    agg,
    get_grouped_values().nullable()
      ? detail::group_count_valid(
          get_grouped_values(), helper.group_labels(), helper.num_groups(), mr, stream)
      : detail::group_count_all(helper.group_offsets(), helper.num_groups(), mr, stream));
}

template <>
void store_result_functor::operator()<aggregation::COUNT_ALL>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(
    col_idx, agg, detail::group_count_all(helper.group_offsets(), helper.num_groups(), mr, stream));
}

template <>
void store_result_functor::operator()<aggregation::SUM>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(col_idx,
                   agg,
                   detail::group_sum(
                     get_grouped_values(), helper.num_groups(), helper.group_labels(), mr, stream));
};

template <>
void store_result_functor::operator()<aggregation::ARGMAX>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(col_idx,
                   agg,
                   detail::group_argmax(get_grouped_values(),
                                        helper.num_groups(),
                                        helper.group_labels(),
                                        helper.key_sort_order(),
                                        mr,
                                        stream));
};

template <>
void store_result_functor::operator()<aggregation::ARGMIN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  cache.add_result(col_idx,
                   agg,
                   detail::group_argmin(get_grouped_values(),
                                        helper.num_groups(),
                                        helper.group_labels(),
                                        helper.key_sort_order(),
                                        mr,
                                        stream));
};

template <>
void store_result_functor::operator()<aggregation::MIN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto result = [&]() {
    if (cudf::is_fixed_width(values.type())) {
      return detail::group_min(
        get_grouped_values(), helper.num_groups(), helper.group_labels(), mr, stream);
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
                             argmin_result.nullable() ? cudf::detail::out_of_bounds_policy::IGNORE
                                                      : cudf::detail::out_of_bounds_policy::NULLIFY,
                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                             mr,
                             stream);
      return std::move(transformed_result->release()[0]);
    }
  }();

  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::MAX>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto result = [&]() {
    if (cudf::is_fixed_width(values.type())) {
      return detail::group_max(
        get_grouped_values(), helper.num_groups(), helper.group_labels(), mr, stream);
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
                             argmax_result.nullable() ? cudf::detail::out_of_bounds_policy::IGNORE
                                                      : cudf::detail::out_of_bounds_policy::NULLIFY,
                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                             mr,
                             stream);
      return std::move(transformed_result->release()[0]);
    }
  }();

  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::MEAN>(aggregation const& agg)
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
                                   mr,
                                   stream);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::VARIANCE>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto var_agg   = static_cast<cudf::detail::std_var_aggregation const&>(agg);
  auto mean_agg  = make_mean_aggregation();
  auto count_agg = make_count_aggregation();
  operator()<aggregation::MEAN>(*mean_agg);
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view mean_result = cache.get_result(col_idx, *mean_agg);
  column_view group_sizes = cache.get_result(col_idx, *count_agg);

  auto result = detail::group_var(get_grouped_values(),
                                  mean_result,
                                  group_sizes,
                                  helper.group_labels(),
                                  var_agg._ddof,
                                  mr,
                                  stream);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::STD>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto std_agg = static_cast<cudf::detail::std_var_aggregation const&>(agg);
  auto var_agg = make_variance_aggregation(std_agg._ddof);
  operator()<aggregation::VARIANCE>(*var_agg);
  column_view var_result = cache.get_result(col_idx, *var_agg);

  auto result = cudf::detail::unary_operation(var_result, unary_op::SQRT, mr, stream);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::QUANTILE>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto count_agg = make_count_aggregation();
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view group_sizes = cache.get_result(col_idx, *count_agg);
  auto quantile_agg       = static_cast<cudf::detail::quantile_aggregation const&>(agg);

  auto result = detail::group_quantiles(get_sorted_values(),
                                        group_sizes,
                                        helper.group_offsets(),
                                        helper.num_groups(),
                                        quantile_agg._quantiles,
                                        quantile_agg._interpolation,
                                        mr,
                                        stream);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::MEDIAN>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto count_agg = make_count_aggregation();
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view group_sizes = cache.get_result(col_idx, *count_agg);

  auto result = detail::group_quantiles(get_sorted_values(),
                                        group_sizes,
                                        helper.group_offsets(),
                                        helper.num_groups(),
                                        {0.5},
                                        interpolation::LINEAR,
                                        mr,
                                        stream);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::NUNIQUE>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto nunique_agg = static_cast<cudf::detail::nunique_aggregation const&>(agg);

  auto result = detail::group_nunique(get_sorted_values(),
                                      helper.group_labels(),
                                      helper.num_groups(),
                                      helper.group_offsets(),
                                      nunique_agg._null_handling,
                                      mr,
                                      stream);
  cache.add_result(col_idx, agg, std::move(result));
};

template <>
void store_result_functor::operator()<aggregation::NTH_ELEMENT>(aggregation const& agg)
{
  if (cache.has_result(col_idx, agg)) return;

  auto nth_element_agg = static_cast<cudf::detail::nth_element_aggregation const&>(agg);

  auto count_agg = make_count_aggregation(nth_element_agg._null_handling);
  if (count_agg->kind == aggregation::COUNT_VALID)
    operator()<aggregation::COUNT_VALID>(*count_agg);
  else if (count_agg->kind == aggregation::COUNT_ALL)
    operator()<aggregation::COUNT_ALL>(*count_agg);
  else
    CUDF_FAIL("Wrong count aggregation kind");
  column_view group_sizes = cache.get_result(col_idx, *count_agg);

  cache.add_result(col_idx,
                   agg,
                   detail::group_nth_element(get_grouped_values(),
                                             group_sizes,
                                             helper.group_labels(),
                                             helper.group_offsets(),
                                             helper.num_groups(),
                                             nth_element_agg._n,
                                             nth_element_agg._null_handling,
                                             mr,
                                             stream));
}
}  // namespace detail

// Sort-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::sort_aggregate(
  std::vector<aggregation_request> const& requests,
  cudaStream_t stream,
  rmm::mr::device_memory_resource* mr)
{
  // We're going to start by creating a cache of results so that aggs that
  // depend on other aggs will not have to be recalculated. e.g. mean depends on
  // sum and count. std depends on mean and count
  cudf::detail::result_cache cache(requests.size());

  for (size_t i = 0; i < requests.size(); i++) {
    auto store_functor =
      detail::store_result_functor(i, requests[i].values, helper(), cache, stream, mr);
    for (size_t j = 0; j < requests[i].aggregations.size(); j++) {
      // TODO (dm): single pass compute all supported reductions
      cudf::detail::aggregation_dispatcher(
        requests[i].aggregations[j]->kind, store_functor, *requests[i].aggregations[j]);
    }
  }

  auto results = detail::extract_results(requests, cache);

  return std::make_pair(helper().unique_keys(mr, stream), std::move(results));
}
}  // namespace groupby
}  // namespace cudf
