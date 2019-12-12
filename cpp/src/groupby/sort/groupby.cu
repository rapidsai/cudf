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

#include "sort_helper.hpp"
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


// Sort-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
    table_view const& keys, std::vector<aggregation_request> const& requests,
    bool ignore_null_keys, bool keys_are_sorted,
    std::vector<order> const& column_order,
    std::vector<null_order> const& null_precedence,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr)
{

  // Sort keys using sort_helper
  // TODO (dm): sort helper should be stored in groupby object
  // TODO (dm): convert sort helper's include_nulls to ignore_nulls
  helper sorter(keys, not ignore_null_keys, null_precedence, keys_are_sorted);

  // We're going to start by creating a cache of results so that aggs that
  // depend on other aggs will not have to be recalculated. e.g. mean depends on
  // sum and count. std depends on mean and count
  result_cache cache(requests.size());

  for (size_t i = 0; i < requests.size(); i++) {
    // TODO (dm): Not all aggs require sorted values. Only sort if there is an 
    //            agg that requires sorted result
    // TODO (dm): Use key_sorted_order to make permutation iterator and avoid
    //            generating value columns
    std::unique_ptr<column> sorted_values;
    rmm::device_vector<size_type> group_sizes;
    std::tie(sorted_values, group_sizes) =
      sorter.sorted_values_and_num_valids(requests[i].values);

        auto store_sum = [&] (size_type col_idx,
                              std::unique_ptr<aggregation> const& agg)
        {
          if (cache.has_result(col_idx, agg))
            return;
          cache.add_result(col_idx, agg, 
                          group_sum(sorted_values->view(), 
                                    sorter.group_labels(),
                                    sorter.num_groups(), stream));
        };

        auto store_count = [&] (size_type col_idx,
                                std::unique_ptr<aggregation> const& agg)
        {
          if (cache.has_result(col_idx, agg))
            return;
          auto counts = std::make_unique<column>(
                          data_type(type_to_id<size_type>()),
                          group_sizes.size(),
                          rmm::device_buffer(group_sizes.data().get(),
                            group_sizes.size() * sizeof(size_type)));
      cache.add_result(col_idx, agg, std::move(counts));
    };

    auto store_mean = [&] (size_type col_idx,
                           std::unique_ptr<aggregation> const& agg)
    {
      if (cache.has_result(col_idx, agg))
        return;
      auto sum_agg = make_sum_aggregation();
      auto count_agg = make_count_aggregation();
      store_sum(col_idx, sum_agg);
      store_count(col_idx, count_agg);
      column_view sum_result = cache.get_result(col_idx, sum_agg);
      column_view count_result = cache.get_result(col_idx, count_agg);
      // TODO (dm): Special case for timestamp. Add target_type_impl for it
      auto result = cudf::experimental::detail::binary_operation(
        sum_result, count_result, binary_operator::DIV, 
        cudf::experimental::detail::target_type(
          requests[col_idx].values.type(), aggregation::MEAN), mr, stream);
      cache.add_result(col_idx, agg, std::move(result));
        };

    for (size_t j = 0; j < requests[i].aggregations.size(); j++) {
      switch (requests[i].aggregations[j]->kind) {
        // TODO (dm): single pass compute all supported reductions
      case aggregation::SUM:
        store_sum(i, requests[i].aggregations[j]);
        break;
      case aggregation::COUNT:
        store_count(i, requests[i].aggregations[j]);
        break;
      case aggregation::MEAN:
        store_mean(i, requests[i].aggregations[j]);
        break;
      case aggregation::QUANTILE:
  
        break;
      default:
        break;
      }
    }
  }  
  
  // TODO (dm): construct aggregation_result's by extracting from result_cache

  return std::make_pair(std::make_unique<table>(),
                        std::vector<aggregation_result>{});
}
}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
