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

TEST_F(SearchTest, non_null_column_multiple_needles_find_first)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 10, 20, 30, 40, 50 };
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  0,  0,  0,  1,  2,  3,  3,  4,  4,  5 };

    gdf_column result_column;

    EXPECT_NO_THROW(
        result_column = cudf::search_sorted(
            *(column.get()),
            *(values.get()),
            true)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, non_null_column_multiple_needles_find_last)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 10, 20, 30, 40, 50 };
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  0,  0,  1,  1,  3,  3,  4,  4,  5,  5 };

    gdf_column result_column;

    EXPECT_NO_THROW(
        result_column = cudf::search_sorted(
            *(column.get()),
            *(values.get()),
            false)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}
