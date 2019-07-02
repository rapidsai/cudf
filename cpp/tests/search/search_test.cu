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
#include <cudf/search.hpp>

using cudf::test::column_wrapper;

class SearchTest : public GdfTest {};

TEST_F(SearchTest, non_null_column__find_first)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 10, 20, 30, 40, 50 };
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  0,  0,  0,  1,  2,  3,  3,  4,  4,  5 };

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::search_ordered(
            *(column.get()),
            *(values.get()),
            true)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, non_null_column__find_last)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 10, 20, 30, 40, 50 };
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  0,  0,  1,  1,  3,  3,  4,  4,  5,  5 };

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::search_ordered(
            *(column.get()),
            *(values.get()),
            false)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_smallest)
{
    using element_type = int64_t;

    std::vector<element_type>   column_data     { 10, 60, 10, 20, 30, 40, 50 };
    std::vector<gdf_valid_type> column_valids   {  0,  0,  1,  1,  1,  1,  1 };
    std::vector<element_type>   values_data     {  8,  8, 10, 11, 30, 32, 40, 47, 50, 90 };
    std::vector<element_type>   value_valids    {  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };

    auto expect = column_wrapper<gdf_index_type>{  2,  2,  3,  3,  5,  5,  6,  6,  7,  7 };
    
    auto column = column_wrapper<element_type> ( column_data.size(),
        [&]( gdf_index_type row ) { return column_data[row]; },
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data.size(),
        [&]( gdf_index_type row ) { return values_data[row]; },
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::search_ordered(
            *(column.get()),
            *(values.get()),
            false,  // find_first
            false)  // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_smallest)
{
    using element_type = int64_t;

    std::vector<element_type>   column_data     { 10, 60, 10, 20, 30, 40, 50 };
    std::vector<gdf_valid_type> column_valids   {  0,  0,  1,  1,  1,  1,  1 };
    std::vector<element_type>   values_data     {  8,  8, 10, 11, 30, 32, 40, 47, 50, 90 };
    std::vector<element_type>   value_valids    {  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };

    auto expect = column_wrapper<gdf_index_type>{  0,  2,  2,  3,  4,  5,  5,  6,  6,  7 };
    
    auto column = column_wrapper<element_type> ( column_data.size(),
        [&]( gdf_index_type row ) { return column_data[row]; },
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data.size(),
        [&]( gdf_index_type row ) { return values_data[row]; },
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::search_ordered(
            *(column.get()),
            *(values.get()),
            true,   // find_first
            false)  // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_largest)
{
    using element_type = int64_t;

    std::vector<element_type>   column_data     { 10, 20, 30, 40, 50, 10, 60 };
    std::vector<gdf_valid_type> column_valids   {  1,  1,  1,  1,  1,  0,  0 };
    std::vector<element_type>   values_data     {  8, 10, 11, 30, 32, 40, 47, 50, 90,  8 };
    std::vector<element_type>   value_valids    {  1,  1,  1,  1,  1,  1,  1,  1,  1,  0 };

    auto expect = column_wrapper<gdf_index_type>{  0,  1,  1,  3,  3,  4,  4,  5,  5,  7 };
    
    auto column = column_wrapper<element_type> ( column_data.size(),
        [&]( gdf_index_type row ) { return column_data[row]; },
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data.size(),
        [&]( gdf_index_type row ) { return values_data[row]; },
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::search_ordered(
            *(column.get()),
            *(values.get()),
            false, // find_first
            true)  // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_largest)
{
    using element_type = int64_t;

    std::vector<element_type>   column_data     { 10, 20, 30, 40, 50, 10, 60 };
    std::vector<gdf_valid_type> column_valids   {  1,  1,  1,  1,  1,  0,  0 };
    std::vector<element_type>   values_data     {  8, 10, 11, 30, 32, 40, 47, 50, 90,  8 };
    std::vector<element_type>   value_valids    {  1,  1,  1,  1,  1,  1,  1,  1,  1,  0 };

    auto expect = column_wrapper<gdf_index_type>{  0,  0,  1,  2,  3,  3,  4,  4,  5,  5 };
    
    auto column = column_wrapper<element_type> ( column_data.size(),
        [&]( gdf_index_type row ) { return column_data[row]; },
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data.size(),
        [&]( gdf_index_type row ) { return values_data[row]; },
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::search_ordered(
            *(column.get()),
            *(values.get()),
            true,  // find_first
            true)  // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

