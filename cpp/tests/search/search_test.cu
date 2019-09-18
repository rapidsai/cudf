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
#include <tests/utilities/scalar_wrapper.cuh>
#include <cudf/search.hpp>

using cudf::test::column_wrapper;
using cudf::test::scalar_wrapper;

class SearchTest : public GdfTest {};

TEST_F(SearchTest, empty_table)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  {};
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 };

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            {column.get()},
            {values.get()},
            {false}
        )
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, empty_values)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 10, 20, 30, 40, 50 };
    auto values = column_wrapper<element_type>  {};
    auto expect = column_wrapper<gdf_index_type>{};

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            {column.get()},
            {values.get()},
            {false}
        )
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, non_null_column__find_first)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 10, 20, 30, 40, 50 };
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  0,  0,  0,  1,  2,  3,  3,  4,  4,  5 };

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            {column.get()},
            {values.get()},
            {false}
        )
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
        result_column = cudf::upper_bound(
            {column.get()},
            {values.get()},
            {false}
        )
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, non_null_column_desc__find_first)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 50, 40, 30, 20, 10 };
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  5,  5,  4,  4,  2,  2,  1,  1,  0,  0 };

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            {column.get()},
            {values.get()},
            {true})   // descending
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, non_null_column_desc__find_last)
{
    using element_type = int64_t;

    auto column = column_wrapper<element_type>  { 50, 40, 30, 20, 10 };
    auto values = column_wrapper<element_type>  {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    auto expect = column_wrapper<gdf_index_type>{  5,  5,  5,  4,  3,  2,  2,  1,  1,  0 };

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::upper_bound(
            {column.get()},
            {values.get()},
            {true})   // descending
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
    
    auto column = column_wrapper<element_type> ( column_data,
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data,
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::upper_bound(
            {column.get()},
            {values.get()},
            {false},   // descending
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
    
    auto column = column_wrapper<element_type> ( column_data,
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data,
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            {column.get()},
            {values.get()},
            {false},   // descending
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
    
    auto column = column_wrapper<element_type> ( column_data,
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data,
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::upper_bound(
            {column.get()},
            {values.get()},
            {false},  // descending
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
    
    auto column = column_wrapper<element_type> ( column_data,
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );
    auto values = column_wrapper<element_type> ( values_data,
        [&]( gdf_index_type row ) { return value_valids[row]; }
    );

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            {column.get()},
            {values.get()},
            {false},   // descending
            true)  // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);

    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table__find_first)
{
    auto column_0 = column_wrapper<int32_t> {  10,  20,  20,  20,  20,  20,  50 };
    auto column_1 = column_wrapper<float>   { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
    auto column_2 = column_wrapper<int8_t>  {  90,  77,  78,  61,  62,  63,  41 };

    auto values_0 = column_wrapper<int32_t> { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    auto values_1 = column_wrapper<float>   { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    auto values_2 = column_wrapper<int8_t>  { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };

    auto expect = column_wrapper<gdf_index_type>
                                            { 0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  2,  1,  3,  3,  3,  6,  4,  6,  6,  6,  7 };

    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>(3, false);

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            input_table,
            values_table,
            desc_flags)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table__find_last)
{
    auto column_0 = column_wrapper<int32_t> {  10,  20,  20,  20,  20,  20,  50 };
    auto column_1 = column_wrapper<float>   { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
    auto column_2 = column_wrapper<int8_t>  {  90,  77,  78,  61,  62,  63,  41 };

    auto values_0 = column_wrapper<int32_t> { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    auto values_1 = column_wrapper<float>   { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    auto values_2 = column_wrapper<int8_t>  { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };

    auto expect = column_wrapper<gdf_index_type>
                                            { 0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  1,  2,  3,  1,  3,  3,  3,  6,  5,  6,  6,  7,  7 };

    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>(3, false);

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::upper_bound(
            input_table,
            values_table,
            desc_flags)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table_partial_desc__find_first)
{
    auto column_0 = column_wrapper<int32_t> {  50,  20,  20,  20,  20,  20,  10 };
    auto column_1 = column_wrapper<float>   {  .7,  .5,  .5,  .7,  .7,  .7, 5.0 };
    auto column_2 = column_wrapper<int8_t>  {  41,  78,  77,  63,  62,  61,  90 };

    auto values_0 = column_wrapper<int32_t> { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    auto values_1 = column_wrapper<float>   { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    auto values_2 = column_wrapper<int8_t>  { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };

    auto expect = column_wrapper<gdf_index_type>
                                            { 7,  7,  7,  7,  6,  7,  6,  6,  7,  7,  7,  7,  6,  1,  3,  2,  1,  3,  3,  3,  3,  3,  4,  3,  1,  0,  0 };
    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>{true, false, true};

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            input_table,
            values_table,
            desc_flags)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table_partial_desc__find_last)
{
    auto column_0 = column_wrapper<int32_t> {  50,  20,  20,  20,  20,  20,  10 };
    auto column_1 = column_wrapper<float>   {  .7,  .5,  .5,  .7,  .7,  .7, 5.0 };
    auto column_2 = column_wrapper<int8_t>  {  41,  78,  77,  63,  62,  61,  90 };

    auto values_0 = column_wrapper<int32_t> { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    auto values_1 = column_wrapper<float>   { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    auto values_2 = column_wrapper<int8_t>  { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };

    auto expect = column_wrapper<gdf_index_type>
                                            { 7,  7,  7,  7,  6,  7,  7,  6,  7,  7,  7,  7,  6,  1,  3,  3,  2,  3,  3,  3,  3,  3,  5,  3,  1,  1,  0 };

    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>{true, false, true};

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::upper_bound(
            input_table,
            values_table,
            desc_flags)
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table__find_first__nulls_as_smallest)
{
    std::vector<int32_t> column_0_data  {  30,  10,  10,  20,  20,  20,  20,  20,  20,  20,  50 };
    std::vector<bool>    column_0_valid {   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1 };
    std::vector<float>   column_1_data  {  .5, 6.0, 5.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7 };
    std::vector<bool>    column_1_valid {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 };
    std::vector<int8_t>  column_2_data  {  50,  95,  90,  79,  76,  77,  78,  61,  62,  63,  41 };
    std::vector<bool>    column_2_valid {   1,   1,   1,   0,   0,   1,   1,   1,   1,   1,   1 };

    std::vector<int32_t> values_0_data  { 10, 40, 20 };
    std::vector<bool>    values_0_valid {  1,  0,  1 };
    std::vector<float>   values_1_data  {  6, .5, .5 };
    std::vector<bool>    values_1_valid {  0,  1,  1 };
    std::vector<int8_t>  values_2_data  { 95, 50, 77 };
    std::vector<bool>    values_2_valid {  1,  1,  0 };

    auto expect = column_wrapper<gdf_index_type>
                                        {  1,  0,  3 };

    auto column_0 = column_wrapper<int32_t> ( column_0_data,
        [&]( gdf_index_type row ) { return column_0_valid[row]; }
    );
    auto column_1 = column_wrapper<float> ( column_1_data,
        [&]( gdf_index_type row ) { return column_1_valid[row]; }
    );
    auto column_2 = column_wrapper<int8_t> ( column_2_data,
        [&]( gdf_index_type row ) { return column_2_valid[row]; }
    );

    auto values_0 = column_wrapper<int32_t> ( values_0_data,
        [&]( gdf_index_type row ) { return values_0_valid[row]; }
    );
    auto values_1 = column_wrapper<float> ( values_1_data,
        [&]( gdf_index_type row ) { return values_1_valid[row]; }
    );
    auto values_2 = column_wrapper<int8_t> ( values_2_data,
        [&]( gdf_index_type row ) { return values_2_valid[row]; }
    );

    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>(3, false);

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            input_table,
            values_table,
            desc_flags,
            false)  // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table__find_last__nulls_as_smallest)
{
    std::vector<int32_t> column_0_data  {  30,  10,  10,  20,  20,  20,  20,  20,  20,  20,  50 };
    std::vector<bool>    column_0_valid {   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1 };
    std::vector<float>   column_1_data  {  .5, 6.0, 5.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7 };
    std::vector<bool>    column_1_valid {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 };
    std::vector<int8_t>  column_2_data  {  50,  95,  90,  79,  76,  77,  78,  61,  62,  63,  41 };
    std::vector<bool>    column_2_valid {   1,   1,   1,   0,   0,   1,   1,   1,   1,   1,   1 };

    std::vector<int32_t> values_0_data  { 10, 40, 20 };
    std::vector<bool>    values_0_valid {  1,  0,  1 };
    std::vector<float>   values_1_data  {  6, .5, .5 };
    std::vector<bool>    values_1_valid {  0,  1,  1 };
    std::vector<int8_t>  values_2_data  { 96, 50, 77 };
    std::vector<bool>    values_2_valid {  1,  1,  0 };

    auto expect = column_wrapper<gdf_index_type>
                                        {  2,  1,  5 };

    auto column_0 = column_wrapper<int32_t> ( column_0_data,
        [&]( gdf_index_type row ) { return column_0_valid[row]; }
    );
    auto column_1 = column_wrapper<float> ( column_1_data,
        [&]( gdf_index_type row ) { return column_1_valid[row]; }
    );
    auto column_2 = column_wrapper<int8_t> ( column_2_data,
        [&]( gdf_index_type row ) { return column_2_valid[row]; }
    );

    auto values_0 = column_wrapper<int32_t> ( values_0_data,
        [&]( gdf_index_type row ) { return values_0_valid[row]; }
    );
    auto values_1 = column_wrapper<float> ( values_1_data,
        [&]( gdf_index_type row ) { return values_1_valid[row]; }
    );
    auto values_2 = column_wrapper<int8_t> ( values_2_data,
        [&]( gdf_index_type row ) { return values_2_valid[row]; }
    );

    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>(3, false);

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::upper_bound(
            input_table,
            values_table,
            desc_flags,
            false)   // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table__find_first__nulls_as_largest)
{
    std::vector<int32_t> column_0_data  {  10,  10,  20,  20,  20,  20,  20,  20,  20,  50,  30 };
    std::vector<bool>    column_0_valid {   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0 };
    std::vector<float>   column_1_data  { 5.0, 6.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7,  .5 };
    std::vector<bool>    column_1_valid {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 };
    std::vector<int8_t>  column_2_data  {  90,  95,  77,  78,  79,  76,  61,  62,  63,  41,  50 };
    std::vector<bool>    column_2_valid {   1,   1,   1,   1,   0,   0,   1,   1,   1,   1,   1 };

    std::vector<int32_t> values_0_data  { 10, 40, 20 };
    std::vector<bool>    values_0_valid {  1,  0,  1 };
    std::vector<float>   values_1_data  {  6, .5, .5 };
    std::vector<bool>    values_1_valid {  0,  1,  1 };
    std::vector<int8_t>  values_2_data  { 95, 50, 77 };
    std::vector<bool>    values_2_valid {  1,  1,  0 };

    auto expect = column_wrapper<gdf_index_type>
                                        {  1, 10,  4 };

    auto column_0 = column_wrapper<int32_t> ( column_0_data,
        [&]( gdf_index_type row ) { return column_0_valid[row]; }
    );
    auto column_1 = column_wrapper<float> ( column_1_data,
        [&]( gdf_index_type row ) { return column_1_valid[row]; }
    );
    auto column_2 = column_wrapper<int8_t> ( column_2_data,
        [&]( gdf_index_type row ) { return column_2_valid[row]; }
    );

    auto values_0 = column_wrapper<int32_t> ( values_0_data,
        [&]( gdf_index_type row ) { return values_0_valid[row]; }
    );
    auto values_1 = column_wrapper<float> ( values_1_data,
        [&]( gdf_index_type row ) { return values_1_valid[row]; }
    );
    auto values_2 = column_wrapper<int8_t> ( values_2_data,
        [&]( gdf_index_type row ) { return values_2_valid[row]; }
    );

    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>(3, false);

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::lower_bound(
            input_table,
            values_table,
            desc_flags,
            true)   // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, table__find_last__nulls_as_largest)
{
    std::vector<int32_t> column_0_data  {  10,  10,  20,  20,  20,  20,  20,  20,  20,  50,  30 };
    std::vector<bool>    column_0_valid {   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0 };
    std::vector<float>   column_1_data  { 5.0, 6.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7,  .5 };
    std::vector<bool>    column_1_valid {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 };
    std::vector<int8_t>  column_2_data  {  90,  95,  77,  78,  79,  76,  61,  62,  63,  41,  50 };
    std::vector<bool>    column_2_valid {   1,   1,   1,   1,   0,   0,   1,   1,   1,   1,   1 };

    std::vector<int32_t> values_0_data  { 10, 40, 20 };
    std::vector<bool>    values_0_valid {  1,  0,  1 };
    std::vector<float>   values_1_data  {  6, .5, .5 };
    std::vector<bool>    values_1_valid {  0,  1,  1 };
    std::vector<int8_t>  values_2_data  { 96, 50, 77 };
    std::vector<bool>    values_2_valid {  1,  1,  0 };

    auto expect = column_wrapper<gdf_index_type>
                                        {  2, 11,  6 };

    auto column_0 = column_wrapper<int32_t> ( column_0_data,
        [&]( gdf_index_type row ) { return column_0_valid[row]; }
    );
    auto column_1 = column_wrapper<float> ( column_1_data,
        [&]( gdf_index_type row ) { return column_1_valid[row]; }
    );
    auto column_2 = column_wrapper<int8_t> ( column_2_data,
        [&]( gdf_index_type row ) { return column_2_valid[row]; }
    );

    auto values_0 = column_wrapper<int32_t> ( values_0_data,
        [&]( gdf_index_type row ) { return values_0_valid[row]; }
    );
    auto values_1 = column_wrapper<float> ( values_1_data,
        [&]( gdf_index_type row ) { return values_1_valid[row]; }
    );
    auto values_2 = column_wrapper<int8_t> ( values_2_data,
        [&]( gdf_index_type row ) { return values_2_valid[row]; }
    );

    std::vector<gdf_column*> columns{column_0.get(), column_1.get(), column_2.get()};
    std::vector<gdf_column*> values {values_0.get(), values_1.get(), values_2.get()};

    auto input_table  = cudf::table(columns);
    auto values_table = cudf::table(values);
    auto desc_flags   = std::vector<bool>(3, false);

    gdf_column result_column{};

    EXPECT_NO_THROW(
        result_column = cudf::upper_bound(
            input_table,
            values_table,
            desc_flags,
            true)    // nulls_as_largest
    );

    auto result = column_wrapper<gdf_index_type>(result_column);
    
    ASSERT_EQ(result, expect) << "  Actual:" << result.to_str()
                              << "Expected:" << expect.to_str();
}

TEST_F(SearchTest, contains_true)
{
    using element_type = int64_t;
    bool expect = true;
    bool  result = false;

    auto column = column_wrapper<element_type> {0, 1, 17, 19, 23, 29, 71};
    auto value = scalar_wrapper<element_type>{23};

    result = cudf::contains(
        column.get()[0],
        value.get()[0]
        );

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_false)
{
    using element_type = int64_t;
    bool expect = false;
    bool  result = false;

    auto column = column_wrapper<element_type> {0, 1, 17, 19, 23, 29, 71};
    auto value = scalar_wrapper<element_type> {24};

    result = cudf::contains(
        column.get()[0],
        value.get()[0]);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_value)
{
    using element_type = int64_t;
    bool expect = false;
    bool  result = false;

    auto column = column_wrapper<element_type> {0, 1, 17, 19, 23, 29, 71};
    auto value = scalar_wrapper<element_type> (23, false);

    result = cudf::contains(
        column.get()[0],
        value.get()[0]);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_column)
{
    using element_type = int64_t;
    bool expect = false;
    bool  result = false;

    auto column = column_wrapper<element_type> {};
    auto value = scalar_wrapper<element_type> {24};

    result = cudf::contains(
        column.get()[0],
        value.get()[0]);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_true)
{
    using element_type = int64_t;
    bool result = false;
    bool expect = true;

    std::vector<element_type>   column_data     { 0, 1, 17, 19, 23, 29, 71};
    std::vector<gdf_valid_type> column_valids   { 0,  0,  1,  1,  1,  1,  1 };
    auto value = scalar_wrapper<element_type> {23};

    auto column = column_wrapper<element_type> ( column_data,
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );

        result = cudf::contains(
            column.get()[0],
            value.get()[0]);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_false)
{
    using element_type = int64_t;
    bool result = false;
    bool expect = false;

    std::vector<element_type>   column_data     { 0, 1, 17, 19, 23, 29, 71};
    std::vector<gdf_valid_type> column_valids   { 0, 0, 1, 1, 0, 1, 1};
    auto value = scalar_wrapper<element_type> {23};

    auto column = column_wrapper<element_type> ( column_data,
        [&]( gdf_index_type row ) { return column_valids[row]; }
    );

        result = cudf::contains(
            column.get()[0],
            value.get()[0]);

    ASSERT_EQ(result, expect);
}
