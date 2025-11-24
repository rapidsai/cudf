/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby_test_util.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

namespace cudf ::test {

void test_single_agg(char const* file,
                     int line,
                     column_view const& keys,
                     column_view const& values,
                     column_view const& expect_keys,
                     column_view const& expect_vals,
                     std::unique_ptr<groupby_aggregation>&& agg,
                     force_use_sort_impl use_sort,
                     null_policy include_null_keys,
                     sorted keys_are_sorted,
                     std::vector<order> const& column_order,
                     std::vector<null_order> const& null_precedence,
                     sorted reference_keys_are_sorted)
{
  SCOPED_TRACE("Original failure location: " + std::string{file} + ":" + std::to_string(line));

  auto const [sorted_expect_keys, sorted_expect_vals] = [&]() {
    if (reference_keys_are_sorted == sorted::NO) {
      auto const sort_expect_order =
        sorted_order(table_view{{expect_keys}}, column_order, null_precedence);
      auto sorted_expect_keys = gather(table_view{{expect_keys}}, *sort_expect_order);
      auto sorted_expect_vals = gather(table_view{{expect_vals}}, *sort_expect_order);
      return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
    }
    auto sorted_expect_keys = std::make_unique<table>(table_view{{expect_keys}});
    auto sorted_expect_vals = std::make_unique<table>(table_view{{expect_vals}});
    return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
  }();

  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back();
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

  auto result = gb_obj.aggregate(requests, test::get_default_stream());

  if (use_sort == force_use_sort_impl::YES && keys_are_sorted == sorted::NO) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, result.first->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_expect_vals->get_column(0),
                                        *result.second[0].results[0]);

  } else {
    auto const sort_order  = sorted_order(result.first->view(), column_order, precedence);
    auto const sorted_keys = gather(result.first->view(), *sort_order);
    auto const sorted_vals = gather(table_view({result.second[0].results[0]->view()}), *sort_order);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, *sorted_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_expect_vals->get_column(0),
                                        sorted_vals->get_column(0));
  }
}

void test_sum_agg(char const* file,
                  int line,
                  column_view const& keys,
                  column_view const& values,
                  column_view const& expected_keys,
                  column_view const& expected_values)
{
  auto const do_test = [&](auto const use_sort_option) {
    test_single_agg(file,
                    line,
                    keys,
                    values,
                    expected_keys,
                    expected_values,
                    make_sum_aggregation<groupby_aggregation>(),
                    use_sort_option,
                    null_policy::INCLUDE);
  };
  do_test(force_use_sort_impl::YES);
  do_test(force_use_sort_impl::NO);
}

void test_single_scan(char const* file,
                      int line,
                      column_view const& keys,
                      column_view const& values,
                      column_view const& expect_keys,
                      column_view const& expect_vals,
                      std::unique_ptr<groupby_scan_aggregation>&& agg,
                      null_policy include_null_keys,
                      sorted keys_are_sorted,
                      std::vector<order> const& column_order,
                      std::vector<null_order> const& null_precedence)
{
  SCOPED_TRACE("Original failure location: " + std::string{file} + ":" + std::to_string(line));

  std::vector<groupby::scan_request> requests;
  requests.emplace_back();
  requests[0].values = values;
  requests[0].aggregations.push_back(std::move(agg));

  groupby::groupby gb_obj(
    table_view({keys}), include_null_keys, keys_are_sorted, column_order, null_precedence);

  // groupby scan uses sort implementation
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals, *result.second[0].results[0]);
}

}  // namespace cudf::test