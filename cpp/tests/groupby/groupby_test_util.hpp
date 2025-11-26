/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>

enum class force_use_sort_impl : bool { NO, YES };

namespace cudf::test {

void test_single_agg(char const* file,
                     int line,
                     column_view const& keys,
                     column_view const& values,
                     column_view const& expect_keys,
                     column_view const& expect_vals,
                     std::unique_ptr<groupby_aggregation>&& agg,
                     force_use_sort_impl use_sort                   = force_use_sort_impl::NO,
                     null_policy include_null_keys                  = null_policy::EXCLUDE,
                     sorted keys_are_sorted                         = sorted::NO,
                     std::vector<order> const& column_order         = {},
                     std::vector<null_order> const& null_precedence = {},
                     sorted reference_keys_are_sorted               = sorted::NO);
void test_sum_agg(char const* file,
                  int line,
                  column_view const& keys,
                  column_view const& values,
                  column_view const& expected_keys,
                  column_view const& expected_values);

void test_single_scan(char const* file,
                      int line,
                      column_view const& keys,
                      column_view const& values,
                      column_view const& expect_keys,
                      column_view const& expect_vals,
                      std::unique_ptr<groupby_scan_aggregation>&& agg,
                      null_policy include_null_keys                  = null_policy::EXCLUDE,
                      sorted keys_are_sorted                         = sorted::NO,
                      std::vector<order> const& column_order         = {},
                      std::vector<null_order> const& null_precedence = {});

}  // namespace cudf::test

#define CUDF_TEST_SINGLE_AGG(...)  cudf::test::test_single_agg(__FILE__, __LINE__, __VA_ARGS__)
#define CUDF_TEST_SUM_AGG(...)     cudf::test::test_sum_agg(__FILE__, __LINE__, __VA_ARGS__)
#define CUDF_TEST_SINGLE_SCAN(...) cudf::test::test_single_scan(__FILE__, __LINE__, __VA_ARGS__)
