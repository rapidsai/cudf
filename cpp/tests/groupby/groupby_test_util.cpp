/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include "groupby_test_util.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <random>

void test_single_agg(cudf::column_view const& keys,
                     cudf::column_view const& values,
                     cudf::column_view const& expect_keys,
                     cudf::column_view const& expect_vals,
                     std::unique_ptr<cudf::groupby_aggregation>&& agg,
                     force_use_sort_impl use_sort,
                     cudf::null_policy include_null_keys,
                     cudf::sorted keys_are_sorted,
                     std::vector<cudf::order> const& column_order,
                     std::vector<cudf::null_order> const& null_precedence,
                     cudf::sorted reference_keys_are_sorted)
{
  auto const [sorted_expect_keys, sorted_expect_vals] = [&]() {
    if (reference_keys_are_sorted == cudf::sorted::NO) {
      auto const sort_expect_order =
        cudf::sorted_order(cudf::table_view{{expect_keys}}, column_order, null_precedence);
      auto sorted_expect_keys = cudf::gather(cudf::table_view{{expect_keys}}, *sort_expect_order);
      auto sorted_expect_vals = cudf::gather(cudf::table_view{{expect_vals}}, *sort_expect_order);
      return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
    } else {
      auto sorted_expect_keys = std::make_unique<cudf::table>(cudf::table_view{{expect_keys}});
      auto sorted_expect_vals = std::make_unique<cudf::table>(cudf::table_view{{expect_vals}});
      return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
    }
  }();

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  if (use_sort == force_use_sort_impl::YES) {
    // WAR to force cudf::groupby to use sort implementation
    requests[0].aggregations.push_back(
      cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
  }

  // since the default behavior of cudf::groupby(...) for an empty null_precedence vector is
  // null_order::AFTER whereas for cudf::sorted_order(...) it's null_order::BEFORE
  auto const precedence = null_precedence.empty()
                            ? std::vector<cudf::null_order>(1, cudf::null_order::BEFORE)
                            : null_precedence;

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), include_null_keys, keys_are_sorted, column_order, precedence);

  auto result = gb_obj.aggregate(requests, cudf::test::get_default_stream());

  if (use_sort == force_use_sort_impl::YES && keys_are_sorted == cudf::sorted::NO) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, result.first->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_expect_vals->get_column(0),
                                        *result.second[0].results[0]);

  } else {
    auto const sort_order  = cudf::sorted_order(result.first->view(), column_order, precedence);
    auto const sorted_keys = cudf::gather(result.first->view(), *sort_order);
    auto const sorted_vals =
      cudf::gather(cudf::table_view({result.second[0].results[0]->view()}), *sort_order);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, *sorted_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_expect_vals->get_column(0),
                                        sorted_vals->get_column(0));
  }
}

void test_sum_agg(cudf::column_view const& keys,
                  cudf::column_view const& values,
                  cudf::column_view const& expected_keys,
                  cudf::column_view const& expected_values)
{
  auto const do_test = [&](auto const use_sort_option) {
    test_single_agg(keys,
                    values,
                    expected_keys,
                    expected_values,
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    use_sort_option,
                    cudf::null_policy::INCLUDE);
  };
  do_test(force_use_sort_impl::YES);
  do_test(force_use_sort_impl::NO);
}

void test_single_scan(cudf::column_view const& keys,
                      cudf::column_view const& values,
                      cudf::column_view const& expect_keys,
                      cudf::column_view const& expect_vals,
                      std::unique_ptr<cudf::groupby_scan_aggregation>&& agg,
                      cudf::null_policy include_null_keys,
                      cudf::sorted keys_are_sorted,
                      std::vector<cudf::order> const& column_order,
                      std::vector<cudf::null_order> const& null_precedence)
{
  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back(cudf::groupby::scan_request());
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), include_null_keys, keys_are_sorted, column_order, null_precedence);

  // cudf::groupby scan uses sort implementation
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({expect_keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals, *result.second[0].results[0]);
}
