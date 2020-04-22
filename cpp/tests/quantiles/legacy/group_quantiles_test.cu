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

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <cudf/legacy/quantiles.hpp>

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
    cudf::table val_table;
    std::tie(key_table, val_table) = cudf::group_quantiles({keys}, {vals}, {0.5});

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    auto result_vals = cudf::test::column_wrapper<double>(*(val_table.get_column(0)));
    ASSERT_EQ(result_vals, expect_vals) << "Expected: " << expect_vals.to_str()
                                        << "  Actual: " << result_vals.to_str();
}

TEST_F(group_quantile, SingleColumnMultiQuant)
{
    auto keys = cudf::test::column_wrapper<int32_t>        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto vals = cudf::test::column_wrapper<float>          { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                                       //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    auto expect_keys = cudf::test::column_wrapper<int32_t> { 1,       2,          3      };
                                                       //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    auto expect_vals = cudf::test::column_wrapper<double>  {  1.5,4.5, 3.25, 6,    4.5,7.5};
    
    cudf::table key_table;
    cudf::table val_table;
    std::tie(key_table, val_table) = cudf::group_quantiles({keys}, {vals}, {0.25, 0.75});

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    auto result_vals = cudf::test::column_wrapper<double>(*(val_table.get_column(0)));
    ASSERT_EQ(result_vals, expect_vals) << "Expected: " << expect_vals.to_str()
                                        << "  Actual: " << result_vals.to_str();
}

TEST_F(group_quantile, MultiColumn)
{
    auto keys = cudf::test::column_wrapper<int32_t>        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto vals0 = cudf::test::column_wrapper<float>         { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto vals1 = cudf::test::column_wrapper<float>         { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

                                                       //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    auto expect_keys = cudf::test::column_wrapper<int32_t> { 1,       2,          3      };
                                                       //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    auto expect_vals0 = cudf::test::column_wrapper<double> {    3,        4.5,       7   };
                                                       //  { 3, 6, 9, 0, 4, 5, 8, 1, 2, 7}
    auto expect_vals1 = cudf::test::column_wrapper<double> {    6,        4.5,       2   };
    
    cudf::table key_table;
    cudf::table val_table;
    std::tie(key_table, val_table) = cudf::group_quantiles({keys}, {vals0, vals1}, {0.5});

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    auto result_vals0 = cudf::test::column_wrapper<double>(*(val_table.get_column(0)));
    ASSERT_EQ(result_vals0, expect_vals0) << "Expected: " << expect_vals0.to_str()
                                          << "  Actual: " << result_vals0.to_str();
    auto result_vals1 = cudf::test::column_wrapper<double>(*(val_table.get_column(1)));
    ASSERT_EQ(result_vals1, expect_vals1) << "Expected: " << expect_vals1.to_str()
                                          << "  Actual: " << result_vals1.to_str();
}

TEST_F(group_quantile, SingleColumnNullable)
{
    std::vector<int32_t> keys_data  { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    std::vector<bool>    keys_valid { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};
    std::vector<double>  vals_data  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<bool>    vals_valid { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1};

                                                       //  { 1, 1, 2, 2, 2, 3, 3}
    auto expect_keys_data = std::vector<int32_t>           { 1,    2,       3   };
                                                       //  { 3, 6, 1, 4, 9, 2, 8}
    auto expect_vals = cudf::test::column_wrapper<double>  {  4.5,    4,      5 };
    
    auto keys = cudf::test::column_wrapper<int32_t> ( keys_data,
        [&]( cudf::size_type row ) { return keys_valid[row]; }
    );
    auto vals = cudf::test::column_wrapper<double> ( vals_data,
        [&]( cudf::size_type row ) { return vals_valid[row]; }
    );

    auto expect_keys = cudf::test::column_wrapper<int32_t> ( expect_keys_data,
        []( cudf::size_type i ) { return true; } 
    );

    cudf::table key_table;
    cudf::table val_table;
    std::tie(key_table, val_table) = cudf::group_quantiles({keys}, {vals}, {0.5});

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    auto result_vals = cudf::test::column_wrapper<double>(*(val_table.get_column(0)));
    ASSERT_EQ(result_vals, expect_vals) << "Expected: " << expect_vals.to_str()
                                        << "  Actual: " << result_vals.to_str();
}
