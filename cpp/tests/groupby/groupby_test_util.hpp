/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
