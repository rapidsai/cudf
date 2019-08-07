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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <cudf/quantiles.hpp>

#include <gtest/gtest.h>

struct group_quantile : public GdfTest {};

TEST_F(group_quantile, SingleColumn)
{
    auto keys = cudf::test::column_wrapper<int32_t>        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto vals = cudf::test::column_wrapper<float>          { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                                       //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    auto expect_keys = cudf::test::column_wrapper<int32_t> { 1,       2,          3      };
                                                       //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    auto expect_vals = cudf::test::column_wrapper<double>  {    3,        4.5,       7   };
    
    cudf::table key_table;
    gdf_column val_col;
    std::tie(key_table, val_col) = cudf::group_quantiles({keys}, vals, 0.5);

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    auto result_vals = cudf::test::column_wrapper<double>(val_col);
    ASSERT_EQ(result_vals, expect_vals) << "Expected: " << expect_vals.to_str()
                                        << "  Actual: " << result_vals.to_str();
}