/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "result_cache.hpp"
#include "group_reductions.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/unary.hpp>

#include <memory>
#include <utility>
#include <unordered_map>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {
namespace sort {

// constexpr bool is_single_pass_reduction(aggregation::Kind k) {
//   return (k == aggregation::SUM) ||
//          (k == aggregation::MIN) ||
//          (k == aggregation::MAX) ||
//          (k == aggregation::COUNT);
// }

// void compute_single_pass_reductions(
//     helper &sort_helper,
//     std::vector<aggregation_request> const& requests,
//     std::vector<aggregation_result> & results,
//     cudaStream_t &stream)
// {
//   for (size_t i = 0; i < requests.size(); i++) {
//     // std::unique_ptr<column> sorted_values;
//     // rmm::device_vector<size_type> group_sizes;
//     // std::tie(sorted_values, group_sizes) =
//     //   sort_helper.sorted_values_and_num_valids(requests[i].values);

//     for (size_t j = 0; j < requests[i].aggregations.size(); j++) {
//       if (is_single_pass_reduction(requests[i].aggregations[j]->kind)) {
//         switch (requests[i].aggregations[j]->kind) {
//         case aggregation::SUM:
//           // doo something
//           break;
        
//         default:
//           break;
//         }
//       }
//     }
//   }
// }

}  // namespace sort

// TODO (dm): Find a better home for this. Probably a result_cache member
std::vector<aggregation_result> extract_results(
    std::vector<aggregation_request> const& requests,
    result_cache& cache)
{
  std::vector<aggregation_result> results(requests.size());

  for (size_t i = 0; i < requests.size(); i++) {
    for (auto &&agg : requests[i].aggregations) {
      results[i].results.emplace_back( cache.release_result(i, agg) );      
    }
  }
  return results;
}
}  // namespace detail

// Sort-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> 
groupby::sort_aggregate(
    std::vector<aggregation_request> const& requests,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr)
{
  // We're going to start by creating a cache of results so that aggs that
  // depend on other aggs will not have to be recalculated. e.g. mean depends on
  // sum and count. std depends on mean and count
  detail::result_cache cache(requests.size());
  
  for (size_t i = 0; i < requests.size(); i++) {
    // TODO (dm): Use key_sorted_order to make permutation iterator and avoid
    //            generating value columns
    std::unique_ptr<column> sorted_values;
    std::unique_ptr<column> grouped_values;

    auto get_grouped_values = [&] () 
    {
      // TODO (dm): After implementing single pass mutli-agg, explore making a
      //            cache of all grouped value columns rather than one at a time
      if (grouped_values)
        return grouped_values->view();
      else if (sorted_values)
        // TODO (dm): When we implement scan, it wouldn't be ok to return sorted
        //            values when asked for grouped values. Change this then.
        return sorted_values->view();
      else
        grouped_values = helper().grouped_values(requests[i].values);
      return grouped_values->view();
    };

    auto get_sorted_values = [&] () 
    {
      if (not sorted_values)
        sorted_values = helper().sorted_values(requests[i].values);
      return sorted_values->view();
    };

    auto store_count = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      cache.add_result(i, agg, 
                       detail::group_count(get_grouped_values(), 
                                helper().group_labels(),
                                helper().num_groups(), mr, stream));
    };

    auto store_sum = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      auto count_agg = make_count_aggregation();
      store_count(count_agg);
      column_view count_result = cache.get_result(i, count_agg);
      cache.add_result(i, agg, 
                      detail::group_sum(get_grouped_values(), count_result, 
                                        helper().group_labels(), stream));
    };

    auto store_min = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      auto count_agg = make_count_aggregation();
      store_count(count_agg);
      column_view count_result = cache.get_result(i, count_agg);
      cache.add_result(i, agg, 
                      detail::group_min(get_grouped_values(), count_result, 
                                        helper().group_labels(), stream));
    };

    auto store_mean = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      auto sum_agg = make_sum_aggregation();
      auto count_agg = make_count_aggregation();
      store_sum(sum_agg);
      store_count(count_agg);
      column_view sum_result = cache.get_result(i, sum_agg);
      column_view count_result = cache.get_result(i, count_agg);
      // TODO (dm): Special case for timestamp. Add target_type_impl for it
      auto result = cudf::experimental::detail::binary_operation(
        sum_result, count_result, binary_operator::DIV, 
        cudf::experimental::detail::target_type(
          requests[i].values.type(), aggregation::MEAN), mr, stream);
      cache.add_result(i, agg, std::move(result));
    };

    auto store_var = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      auto var_agg =
        static_cast<experimental::detail::std_var_aggregation const*>(agg.get());
      auto mean_agg = make_mean_aggregation();
      auto count_agg = make_count_aggregation();
      store_mean(mean_agg);
      store_count(count_agg);
      column_view mean_result = cache.get_result(i, mean_agg);
      column_view group_sizes = cache.get_result(i, count_agg);
      auto result = detail::group_var(get_grouped_values(), mean_result, 
                              group_sizes, helper().group_labels(), var_agg->_ddof,
                              mr, stream);
      cache.add_result(i, agg, std::move(result));
    };
    
    auto store_std = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      auto std_agg =
        static_cast<experimental::detail::std_var_aggregation const*>(agg.get());
      auto var_agg = make_variance_aggregation(std_agg->_ddof);
      store_var(var_agg);
      column_view var_result = cache.get_result(i, var_agg);
      auto result = experimental::detail::unary_operation(
        var_result, experimental::unary_op::SQRT, mr, stream);
      cache.add_result(i, agg, std::move(result));
    };

    auto store_quantile = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      auto count_agg = make_count_aggregation();
      store_count(count_agg);
      column_view group_sizes = cache.get_result(i, count_agg);
      auto quantile_agg =
        static_cast<experimental::detail::quantile_aggregation const*>(agg.get());
      auto result = detail::group_quantiles(
        get_sorted_values(), group_sizes, helper().group_offsets(),
        quantile_agg->_quantiles, quantile_agg->_interpolation, mr, stream);
      cache.add_result(i, agg, std::move(result));
    };

    auto store_median = [&] (std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(i, agg))
        return;
      auto count_agg = make_count_aggregation();
      store_count(count_agg);
      column_view group_sizes = cache.get_result(i, count_agg);
      auto result = detail::group_quantiles(
        get_sorted_values(), group_sizes, helper().group_offsets(),
        {0.5}, interpolation::LINEAR, mr, stream);
      cache.add_result(i, agg, std::move(result));
    };

    for (size_t j = 0; j < requests[i].aggregations.size(); j++) {
      switch (requests[i].aggregations[j]->kind) {
        // TODO (dm): single pass compute all supported reductions
      case aggregation::SUM:
        store_sum(requests[i].aggregations[j]);
        break;
      case aggregation::MIN:
        store_min(requests[i].aggregations[j]);
        break;
      case aggregation::COUNT:
        store_count(requests[i].aggregations[j]);
        break;
      case aggregation::MEAN:
        store_mean(requests[i].aggregations[j]);
        break;
      case aggregation::STD:
        store_std(requests[i].aggregations[j]);
        break;
      case aggregation::VARIANCE:
        store_var(requests[i].aggregations[j]);
        break;
      case aggregation::QUANTILE:
        store_quantile(requests[i].aggregations[j]);
        break;
      case aggregation::MEDIAN:
        store_median(requests[i].aggregations[j]);
        break;
      default:
        break;
      }
    }
  }  
  
  auto results = extract_results(requests, cache);
  
  return std::make_pair(helper().unique_keys(),
                        std::move(results));
}
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
