/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "scan_tests.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/reduction.hpp>

template <typename T>
struct TypedEwmScanTest : BaseScanTest<T> {
  inline void test_ungrouped_ewma_scan(cudf::column_view const& input,
                                       cudf::column_view const& expect_vals,
                                       cudf::scan_aggregation const& agg,
                                       cudf::null_policy null_handling)
  {
    auto col_out = cudf::scan(input, agg, cudf::scan_type::INCLUSIVE, null_handling);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals, col_out->view());
  }
};

TYPED_TEST_SUITE(TypedEwmScanTest, cudf::test::FloatingPointTypes);

TYPED_TEST(TypedEwmScanTest, Ewm)
{
  auto const v = make_vector<TypeParam>({1.0, 2.0, 3.0, 4.0, 5.0});
  auto col     = this->make_column(v);

  auto const expected_ewma_vals_adjust = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1.0, 1.75, 2.61538461538461497469, 3.54999999999999982236, 4.52066115702479365268}};

  auto const expected_ewma_vals_noadjust =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{1.0,
                                                       1.66666666666666651864,
                                                       2.55555555555555535818,
                                                       3.51851851851851815667,
                                                       4.50617283950617242283}};

  this->test_ungrouped_ewma_scan(
    *col,
    expected_ewma_vals_adjust,
    *cudf::make_ewma_aggregation<cudf::scan_aggregation>(0.5, cudf::ewm_history::INFINITE),
    cudf::null_policy::INCLUDE);
  this->test_ungrouped_ewma_scan(
    *col,
    expected_ewma_vals_noadjust,
    *cudf::make_ewma_aggregation<cudf::scan_aggregation>(0.5, cudf::ewm_history::FINITE),
    cudf::null_policy::INCLUDE);
}

TYPED_TEST(TypedEwmScanTest, EwmWithNulls)
{
  auto const v = make_vector<TypeParam>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 0, 1, 0, 0, 1, 1});
  auto col     = this->make_column(v, b);

  auto const expected_ewma_vals_adjust =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{1.0,
                                                       1.0,
                                                       2.79999999999999982236,
                                                       2.79999999999999982236,
                                                       2.79999999999999982236,
                                                       5.87351778656126466416,
                                                       6.70977596741344139986}};

  auto const expected_ewma_vals_noadjust =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{1.0,
                                                       1.0,
                                                       2.71428571428571441260,
                                                       2.71428571428571441260,
                                                       2.71428571428571441260,
                                                       5.82706766917293172980,
                                                       6.60902255639097724327}};

  this->test_ungrouped_ewma_scan(
    *col,
    expected_ewma_vals_adjust,
    *cudf::make_ewma_aggregation<cudf::scan_aggregation>(0.5, cudf::ewm_history::INFINITE),
    cudf::null_policy::INCLUDE);
  this->test_ungrouped_ewma_scan(
    *col,
    expected_ewma_vals_noadjust,
    *cudf::make_ewma_aggregation<cudf::scan_aggregation>(0.5, cudf::ewm_history::FINITE),
    cudf::null_policy::INCLUDE);
}
