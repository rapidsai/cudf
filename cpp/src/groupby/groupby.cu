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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/rolling.hpp>

#include <memory>
#include <utility>

namespace cudf {
namespace experimental {
namespace groupby {

// Constructor
groupby::groupby(table_view const& keys, bool ignore_null_keys,
                 bool keys_are_sorted, std::vector<order> const& column_order,
                 std::vector<null_order> const& null_precedence)
    : _keys{keys},
      _ignore_null_keys{ignore_null_keys},
      _keys_are_sorted{keys_are_sorted},
      _column_order{column_order},
      _null_precedence{null_precedence} {}

// Select hash vs. sort groupby implementation
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
groupby::dispatch_aggregation(std::vector<aggregation_request> const& requests,
                              cudaStream_t stream,
                              rmm::mr::device_memory_resource* mr) {
  // If sort groupby has been called once on this groupby object, then
  // always use sort groupby from now on. Because once keys are sorted, 
  // all the aggs that can be done by hash groupby are efficiently done by
  // sort groupby as well.
  // Only use hash groupby if the keys aren't sorted and all requests can be
  // satisfied with a hash implementation
  if (not _keys_are_sorted and
      not _helper and
      detail::hash::can_use_hash_groupby(_keys, requests)) {
    return detail::hash::groupby(_keys, requests, _ignore_null_keys, stream,
                                 mr);
  } else {
    return sort_aggregate(requests, stream, mr);
  }
}

// Destructor
// Needs to be in source file because sort_groupby_helper was forward declared
groupby::~groupby() = default;

namespace {
/// Make an empty table with appropriate types for requested aggs
template <typename aggregation_request_t, typename F>
auto templated_empty_results(std::vector<aggregation_request_t> const& requests, F get_kind) {
  
  std::vector<aggregation_result> empty_results;

  std::transform(
      requests.begin(), requests.end(), std::back_inserter(empty_results),
      [&get_kind](auto const& request) {
        std::vector<std::unique_ptr<column>> results;

        std::transform(
            request.aggregations.begin(), request.aggregations.end(),
            std::back_inserter(results), [&request, get_kind](auto const& agg) {
              return make_empty_column(experimental::detail::target_type(
                  request.values.type(), get_kind(agg)));
            });

        return aggregation_result{std::move(results)};
      });

  return empty_results;
}

/// Verifies the agg requested on the request's values is valid
void verify_valid_requests(std::vector<aggregation_request> const& requests) {
  CUDF_EXPECTS(
      std::all_of(requests.begin(), requests.end(),
                  [](auto const& request) {
                    return std::all_of(
                        request.aggregations.begin(),
                        request.aggregations.end(),
                        [&request](auto const& agg) {
                          return experimental::detail::is_valid_aggregation(
                              request.values.type(), agg->kind);
                        });
                  }),
      "Invalid type/aggregation combination.");
}

// TODO: Remove when `rolling_window()` switches to use aggregation-enum directly.
rolling_operator to_rolling_operator(aggregation const& agg) {
  switch(agg.kind) {
    case aggregation::SUM     : return cudf::experimental::rolling_operator::SUM;
    case aggregation::MIN     : return cudf::experimental::rolling_operator::MIN;
    case aggregation::MAX     : return cudf::experimental::rolling_operator::MAX;
    case aggregation::MEAN    : return cudf::experimental::rolling_operator::MEAN;
    case aggregation::COUNT   : return cudf::experimental::rolling_operator::COUNT;
    default : throw std::logic_error("Unsupported operator: " + std::to_string(agg.kind));
  }
}

}  // namespace

// Compute aggregation requests
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
groupby::aggregate(std::vector<aggregation_request> const& requests,
                   rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(std::all_of(requests.begin(), requests.end(),
                           [this](auto const& request) {
                             return request.values.size() == _keys.num_rows();
                           }),
               "Size mismatch between request values and groupby keys.");

  verify_valid_requests(requests);

  if (_keys.num_rows() == 0) {
    std::make_pair(empty_like(_keys), 
                   templated_empty_results(requests, 
                                           [](std::unique_ptr<aggregation> const& agg)
                                           {return agg->kind;}));
  }

  return dispatch_aggregation(requests, 0, mr);
}

// Get the sort helper object
detail::sort::sort_groupby_helper& groupby::helper() {
  if (_helper)
    return *_helper;
  _helper = std::make_unique<detail::sort::sort_groupby_helper>(
    _keys, _ignore_null_keys, _keys_are_sorted);
  return *_helper;
};

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::aggregate(
    std::vector<window_aggregation_request> const& requests,
    rmm::mr::device_memory_resource* mr) {

  CUDF_EXPECTS(std::all_of(requests.begin(), requests.end(),
                           [this](auto const& request) {
                             return request.values.size() == _keys.num_rows();
                           }),
               "Size mismatch between request values and groupby keys.");

  CUDF_EXPECTS(this->_keys_are_sorted, 
               "Window-aggregation is currently supported only on pre-sorted key columns.");

  if (_keys.num_rows() == 0) {
    std::make_pair(empty_like(_keys), 
                  templated_empty_results(requests, 
                                          [](std::pair<window_bounds, std::unique_ptr<aggregation>> const& agg) 
                                          {return agg.second->kind;}));
  }

  auto group_offsets = helper().group_offsets();
  group_offsets.push_back(_keys.num_rows()); // Cap the end.

  std::vector<aggregation_result> results;
  std::transform(
    requests.begin(), requests.end(), std::back_inserter(results),
    [&](auto const& window_request) {
      std::vector<std::unique_ptr<column>> per_request_results;
      auto const& values = window_request.values;
      std::transform(
        window_request.aggregations.begin(), window_request.aggregations.end(), 
        std::back_inserter(per_request_results),
        [&](std::pair<window_bounds, std::unique_ptr<aggregation>> const& agg) {
          return cudf::experimental::rolling_window(
            values,
            group_offsets,
            agg.first.preceding,
            agg.first.following,
            agg.first.min_periods,
            to_rolling_operator(*agg.second),
            mr
          );
        }
      );
      return aggregation_result{std::move(per_request_results)};
    }
  );

  return std::make_pair(std::make_unique<table>(_keys), std::move(results));
}

}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
