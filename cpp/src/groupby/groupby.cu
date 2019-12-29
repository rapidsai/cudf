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
#include <cudf/detail/rolling/rolling.cuh>
#include <cudf/groupby.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

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

/**
 * @brief  Applies a fixed-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * static (the same for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window + 1`. Element `i` uses elements
 *   `[i-preceding_window, i+following_window]` to do the window computation, provided that they
 *   fall within the confines of their corresponding groups, as indicated by `group_offsets`.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 * 
 * The returned column for `op == COUNT` always has `INT32` type. All other operators return a 
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @param[in] input_col The input column
 * @param[in] group_offsets A column of indexes, indicating partition/grouping boundaries.
 * @param[in] preceding_window The static rolling window size in the backward direction.
 * @param[in] following_window The static rolling window size in the forward direction.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] op The rolling window aggregation type (SUM, MAX, MIN, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 **/

std::unique_ptr<column> rolling_window(column_view const& input,
                                       rmm::device_vector<cudf::size_type> const& group_offsets,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  if (group_offsets.empty()) {
    // Empty group_offsets list. Treat `input` as a single group. i.e. Ignore grouping.
    return rolling_window(input, preceding_window, following_window, min_periods, aggr, mr);
  }

  // `group_offsets` are interpreted in adjacent pairs, each pair representing the offsets
  // of the first, and one past the last elements in a group.
  //
  // If `group_offsets` is not empty, it must contain at least two offsets:
  //   a. 0, indicating the first element in `input`
  //   b. input.size(), indicating one past the last element in `input`.
  //
  // Thus, for an input of 1000 rows,
  //   0. [] indicates a single group, spanning the entire column.
  //   1  [10] is invalid.
  //   2. [0, 1000] indicates a single group, spanning the entire column (thus, equivalent to no groups.)
  //   3. [0, 500, 1000] indicates two equal-sized groups: [0,500), and [500,1000).

  CUDF_EXPECTS(group_offsets.size() >= 2 && group_offsets[0] == 0 
               && group_offsets[group_offsets.size()-1] == input.size(),
               "Must have at least one group.");

  auto offsets_begin = group_offsets.begin(); // Required, since __device__ lambdas cannot capture by ref,
  auto offsets_end   = group_offsets.end();   //   or capture local variables without listing them.

  auto preceding_calculator = [offsets_begin, offsets_end, preceding_window] __device__ (size_type idx) {
    // `upper_bound()` cannot return `offsets_end`, since it is capped with `input.size()`.
    auto group_end = thrust::upper_bound(thrust::device, offsets_begin, offsets_end, idx);
    auto group_start = group_end - 1; // The previous offset identifies the start of the group.
    return thrust::minimum<size_type>{}(preceding_window, idx - (*group_start));
  };
 
  auto following_calculator = [offsets_begin, offsets_end, following_window] __device__ (size_type idx) {
    // `upper_bound()` cannot return `offsets_end`, since it is capped with `input.size()`.
    auto group_end = thrust::upper_bound(thrust::device, offsets_begin, offsets_end, idx);
    return thrust::minimum<size_type>{}(following_window, (*group_end - 1) - idx);
  };
  
  return cudf::experimental::detail::rolling_window(
    input,
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), following_calculator),
    min_periods, aggr, mr
  );
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

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::windowed_aggregate(
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
          return rolling_window(
            values,
            group_offsets,
            agg.first.preceding,
            agg.first.following,
            agg.first.min_periods,
            agg.second,
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
