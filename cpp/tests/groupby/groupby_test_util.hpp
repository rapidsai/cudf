/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#pragma once

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <random>

namespace cudf {
namespace test {
enum class force_use_sort_impl : bool { NO, YES };

inline void test_groups(column_view const& keys,
                        column_view const& expect_grouped_keys,
                        std::vector<size_type> const& expect_group_offsets,
                        column_view const& values                = {},
                        column_view const& expect_grouped_values = {})
{
  groupby::groupby gb(table_view({keys}));
  groupby::groupby::groups gb_groups;

  if (values.size()) {
    gb_groups = gb.get_groups(table_view({values}));
  } else {
    gb_groups = gb.get_groups();
  }
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_grouped_keys}), gb_groups.keys->view());

  auto got_offsets = gb_groups.offsets;
  EXPECT_EQ(expect_group_offsets.size(), got_offsets.size());
  for (auto i = 0u; i != expect_group_offsets.size(); ++i) {
    EXPECT_EQ(expect_group_offsets[i], got_offsets[i]);
  }

  if (values.size()) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_grouped_values}), gb_groups.values->view());
  }
}

inline void test_single_agg(column_view const& keys,
                            column_view const& values,
                            column_view const& expect_keys,
                            column_view const& expect_vals,
                            std::unique_ptr<groupby_aggregation>&& agg,
                            force_use_sort_impl use_sort           = force_use_sort_impl::NO,
                            null_policy include_null_keys          = null_policy::EXCLUDE,
                            sorted keys_are_sorted                 = sorted::NO,
                            std::vector<order> const& column_order = {},
                            std::vector<null_order> const& null_precedence = {},
                            sorted reference_keys_are_sorted               = sorted::NO)
{
  auto const [sorted_expect_keys, sorted_expect_vals] = [&]() {
    if (reference_keys_are_sorted == sorted::NO) {
      auto const sort_expect_order =
        sorted_order(table_view{{expect_keys}}, column_order, null_precedence);
      auto sorted_expect_keys = gather(table_view{{expect_keys}}, *sort_expect_order);
      auto sorted_expect_vals = gather(table_view{{expect_vals}}, *sort_expect_order);
      return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
    } else {
      auto sorted_expect_keys = std::make_unique<table>(table_view{{expect_keys}});
      auto sorted_expect_vals = std::make_unique<table>(table_view{{expect_vals}});
      return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
    }
  }();

  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  if (use_sort == force_use_sort_impl::YES) {
    // WAR to force groupby to use sort implementation
    requests[0].aggregations.push_back(make_nth_element_aggregation<groupby_aggregation>(0));
  }

  // since the default behavior of groupby(...) for an empty null_precedence vector is
  // null_order::AFTER whereas for sorted_order(...) it's null_order::BEFORE
  auto const precedence =
    null_precedence.empty() ? std::vector<null_order>(1, null_order::BEFORE) : null_precedence;

  groupby::groupby gb_obj(
    table_view({keys}), include_null_keys, keys_are_sorted, column_order, precedence);

  auto result = gb_obj.aggregate(requests);

  if (use_sort == force_use_sort_impl::YES && keys_are_sorted == sorted::NO) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, result.first->view());
    cudf::test::detail::expect_columns_equivalent(sorted_expect_vals->get_column(0),
                                                  *result.second[0].results[0],
                                                  debug_output_level::ALL_ERRORS);

  } else {
    auto const sort_order  = sorted_order(result.first->view(), column_order, precedence);
    auto const sorted_keys = gather(result.first->view(), *sort_order);
    auto const sorted_vals = gather(table_view({result.second[0].results[0]->view()}), *sort_order);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, *sorted_keys);
    cudf::test::detail::expect_columns_equivalent(sorted_expect_vals->get_column(0),
                                                  sorted_vals->get_column(0),
                                                  debug_output_level::ALL_ERRORS);
  }
}

inline void test_sum_agg(column_view const& keys,
                         column_view const& values,
                         column_view const& expected_keys,
                         column_view const& expected_values)
{
  auto const do_test = [&](auto const use_sort_option) {
    test_single_agg(keys,
                    values,
                    expected_keys,
                    expected_values,
                    cudf::make_sum_aggregation<groupby_aggregation>(),
                    use_sort_option,
                    null_policy::INCLUDE);
  };
  do_test(force_use_sort_impl::YES);
  do_test(force_use_sort_impl::NO);
}

inline void test_single_scan(column_view const& keys,
                             column_view const& values,
                             column_view const& expect_keys,
                             column_view const& expect_vals,
                             std::unique_ptr<groupby_scan_aggregation>&& agg,
                             null_policy include_null_keys                  = null_policy::EXCLUDE,
                             sorted keys_are_sorted                         = sorted::NO,
                             std::vector<order> const& column_order         = {},
                             std::vector<null_order> const& null_precedence = {})
{
  std::vector<groupby::scan_request> requests;
  requests.emplace_back(groupby::scan_request());
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  groupby::groupby gb_obj(
    table_view({keys}), include_null_keys, keys_are_sorted, column_order, null_precedence);

  // groupby scan uses sort implementation
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_keys}), result.first->view());
  cudf::test::detail::expect_columns_equivalent(
    expect_vals, *result.second[0].results[0], debug_output_level::ALL_ERRORS);
}

}  // namespace test
}  // namespace cudf
