/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/sorting.hpp>

template <typename T>
struct GroupbyHistogramTest : public cudf::test::BaseFixture {};

template <typename T>
struct GroupbyMergeHistogramTest : public cudf::test::BaseFixture {};

// Avoid unsigned types, as the tests below have negative values in their input.
using HistogramTestTypes = cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                                              cudf::test::FloatingPointTypes,
                                              cudf::test::FixedPointTypes,
                                              cudf::test::ChronoTypes>;
TYPED_TEST_SUITE(GroupbyHistogramTest, HistogramTestTypes);
TYPED_TEST_SUITE(GroupbyMergeHistogramTest, HistogramTestTypes);

auto groupby_histogram(cudf::column_view const& keys,
                       cudf::column_view const& values,
                       std::unique_ptr<cudf::groupby_aggregation>&& agg)
{
  CUDF_EXPECTS(
    agg->kind == cudf::aggregation::HISTOGRAM || agg->kind == cudf::aggregation::MERGE_HISTOGRAM,
    "Aggregation must be either HISTOGRAM or MERGE_HISTOGRAM.");

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = values;
  requests[0].aggregations.push_back(std::move(agg));

  auto gb_obj              = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const agg_results   = gb_obj.aggregate(requests, cudf::test::get_default_stream());
  auto const agg_histogram = agg_results.second[0].results[0]->view();
  EXPECT_NE(agg_histogram.type().id(), cudf::type_id::LIST);
  EXPECT_EQ(agg_histogram.num_children(), 2);

  auto const key_sort_order = cudf::sorted_order(agg_results.first->view(), {}, {});
  auto sorted_keys =
    std::move(cudf::gather(agg_results.first->view(), *key_sort_order)->release().front());
  auto const sorted_vals = std::move(
    cudf::gather(cudf::table_view({agg_results.second[0].results[0]->view()}), *key_sort_order)
      ->release()
      .front());
  auto sorted_histograms = cudf::lists::sort_lists(cudf::lists_column_view{*sorted_vals},
                                                   cudf::order::ASCENDING,
                                                   cudf::null_order::BEFORE,
                                                   rmm::mr::get_current_device_resource());

  return std::pair{std::move(sorted_keys), std::move(sorted_histograms)};
}
