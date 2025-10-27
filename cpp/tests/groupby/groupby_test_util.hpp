/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>

enum class force_use_sort_impl : bool { NO, YES };

void test_single_agg(cudf::column_view const& keys,
                     cudf::column_view const& values,
                     cudf::column_view const& expect_keys,
                     cudf::column_view const& expect_vals,
                     std::unique_ptr<cudf::groupby_aggregation>&& agg,
                     force_use_sort_impl use_sort                 = force_use_sort_impl::NO,
                     cudf::null_policy include_null_keys          = cudf::null_policy::EXCLUDE,
                     cudf::sorted keys_are_sorted                 = cudf::sorted::NO,
                     std::vector<cudf::order> const& column_order = {},
                     std::vector<cudf::null_order> const& null_precedence = {},
                     cudf::sorted reference_keys_are_sorted               = cudf::sorted::NO);

void test_sum_agg(cudf::column_view const& keys,
                  cudf::column_view const& values,
                  cudf::column_view const& expected_keys,
                  cudf::column_view const& expected_values);

void test_single_scan(cudf::column_view const& keys,
                      cudf::column_view const& values,
                      cudf::column_view const& expect_keys,
                      cudf::column_view const& expect_vals,
                      std::unique_ptr<cudf::groupby_scan_aggregation>&& agg,
                      cudf::null_policy include_null_keys          = cudf::null_policy::EXCLUDE,
                      cudf::sorted keys_are_sorted                 = cudf::sorted::NO,
                      std::vector<cudf::order> const& column_order = {},
                      std::vector<cudf::null_order> const& null_precedence = {});
