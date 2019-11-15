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

#include <tests/groupby/common/legacy/groupby_test.hpp>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <cudf/legacy/reduction.hpp>

#include <gtest/gtest.h>

struct GroupSTDTest : public GdfTest {
    auto all_valid() {
        auto all_valid = [] (cudf::size_type) { return true; };
        return all_valid;
    }
};

TEST_F(GroupSTDTest, SingleColumn)
{
    auto keys = cudf::test::column_wrapper<int32_t>        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto vals = cudf::test::column_wrapper<float>          { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                                       //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    auto expect_keys = cudf::test::column_wrapper<int32_t> { 1,       2,          3      };
                                                       //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    auto expect_vals = cudf::test::column_wrapper<double> ({    3,   sqrt(131./12),sqrt(31./3)},
        all_valid());
    
    cudf::table key_table;
    cudf::table val_table;
    std::tie(key_table, val_table) = cudf::group_std({keys}, {vals});

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    cudf::test::detail::expect_values_are_equal({val_table.get_column(0)}, {expect_vals.get()});
}

TEST_F(GroupSTDTest, MultiColumn)
{
    auto keys = cudf::test::column_wrapper<int32_t>        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto vals0 = cudf::test::column_wrapper<float>         { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto vals1 = cudf::test::column_wrapper<float>         { 1, 4, 6, 3, 4, 8, 0, 6, 6, 2};

                                                       //  { 1, 1, 1,   2, 2, 2, 2,    3, 3, 3}
    auto expect_keys = cudf::test::column_wrapper<int32_t> { 1,         2,             3      };
                                                       //  { 0, 3, 6,   1, 4, 5, 9,    2, 7, 8}
    auto expect_vals0 = cudf::test::column_wrapper<double>({    3,     sqrt(131./12),  sqrt(31./3)}, 
        all_valid());
                                                       //  { 0, 1, 3,   2, 4, 4, 8,    6, 6, 6}
    auto expect_vals1 = cudf::test::column_wrapper<double>({sqrt(7./3), sqrt(19./3),   0      },
        all_valid());
    
    cudf::table key_table;
    cudf::table val_table;
    std::tie(key_table, val_table) = cudf::group_std({keys}, {vals0, vals1});

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    cudf::test::detail::expect_tables_are_equal(val_table, {expect_vals0.get(), expect_vals1.get()});
}

TEST_F(GroupSTDTest, SingleColumnNullable)
{
    std::vector<int32_t> keys_data  { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4};
    std::vector<bool>    keys_valid { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1};
    std::vector<double>  vals_data  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};
    std::vector<bool>    vals_valid { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};

                                                       //  { 1, 1,     2, 2, 2,   3, 3,      4}
    auto expect_keys_data = std::vector<int32_t>           { 1,        2,         3,         4};
                                                       //  { 3, 6,     1, 4, 9,   2, 8,      -}
    auto expect_vals_data = std::vector<double>            {3/sqrt(2), 7/sqrt(3), 3*sqrt(2), 0};
    auto expect_vals_valid = std::vector<bool>             {1,           1,         1,       0};
    
    auto keys = cudf::test::column_wrapper<int32_t> ( keys_data,
        [&]( cudf::size_type row ) { return keys_valid[row]; }
    );
    auto vals = cudf::test::column_wrapper<double> ( vals_data,
        [&]( cudf::size_type row ) { return vals_valid[row]; }
    );

    auto expect_keys = cudf::test::column_wrapper<int32_t> ( expect_keys_data,
        []( cudf::size_type i ) { return true; } 
    );
    auto expect_vals = cudf::test::column_wrapper<double> ( expect_vals_data,
        [&]( cudf::size_type i ) { return expect_vals_valid[i]; } 
    );

    cudf::table key_table;
    cudf::table val_table;
    std::tie(key_table, val_table) = cudf::group_std({keys}, {vals});

    auto result_keys = cudf::test::column_wrapper<int32_t>(*(key_table.get_column(0)));
    ASSERT_EQ(result_keys, expect_keys) << "Expected: " << expect_keys.to_str()
                                        << "  Actual: " << result_keys.to_str();

    cudf::test::detail::expect_values_are_equal({val_table.get_column(0)}, {expect_vals.get()});
}
