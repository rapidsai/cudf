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

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/scalar_wrapper.cuh>

// TODO:  temporary
#include <tests/utilities/column_utilities.hpp>

#include "cudf/search.hpp"

using cudf::test::fixed_width_column_wrapper;
using cudf::test::scalar_wrapper;
// TODO::  temporary
using cudf::test::expect_columns_equal;
using cudf::test::column_values_equal;
using cudf::test::column_view_to_str;

using cudf::column_view;

class SearchTest : public GdfTest {};

using index_type = int32_t;

TEST_F(SearchTest, empty_table)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>  column {};
    fixed_width_column_wrapper<element_type>  values {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    fixed_width_column_wrapper<index_type>    expect {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 };

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
        result = cudf::experimental::lower_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::BEFORE)
                    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, empty_values)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>  column { 10, 20, 30, 40, 50 };
    fixed_width_column_wrapper<element_type>  values {};
    fixed_width_column_wrapper<index_type>    expect {};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::BEFORE)
                    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, non_null_column__find_first)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>  column { 10, 20, 30, 40, 50 };
    fixed_width_column_wrapper<element_type>  values {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    fixed_width_column_wrapper<index_type>    expect {  0,  0,  0,  1,  2,  3,  3,  4,  4,  5 };

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::BEFORE)
                    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, non_null_column__find_last)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>  column { 10, 20, 30, 40, 50 };
    fixed_width_column_wrapper<element_type>  values {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    fixed_width_column_wrapper<index_type>    expect {  0,  0,  1,  1,  3,  3,  4,  4,  5,  5 };

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::BEFORE)
                    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, non_null_column_desc__find_first)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>  column { 50, 40, 30, 20, 10 };
    fixed_width_column_wrapper<element_type>  values {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    fixed_width_column_wrapper<index_type>    expect {  5,  5,  4,  4,  2,  2,  1,  1,  0,  0 };

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::DESCENDING},
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, non_null_column_desc__find_last)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>  column { 50, 40, 30, 20, 10 };
    fixed_width_column_wrapper<element_type>  values {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    fixed_width_column_wrapper<index_type>    expect {  5,  5,  5,  4,  3,  2,  2,  1,  1,  0 };

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::DESCENDING},
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_smallest)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>   column { { 10, 60, 10, 20, 30, 40, 50 },
                                                        {  0,  0,  1,  1,  1,  1,  1 } };
    fixed_width_column_wrapper<element_type>   values { {  8,  8, 10, 11, 30, 32, 40, 47, 50, 90 },
                                                        {  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 } };

    fixed_width_column_wrapper<index_type>     expect {  2,  2,  3,  3,  5,  5,  6,  6,  7,  7 };
    
    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::BEFORE)
                    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_smallest)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>   column { { 10, 60, 10, 20, 30, 40, 50 },
                                                        {  0,  0,  1,  1,  1,  1,  1 } };
    fixed_width_column_wrapper<element_type>   values { {  8,  8, 10, 11, 30, 32, 40, 47, 50, 90 },
                                                        {  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 } };

    fixed_width_column_wrapper<index_type>     expect {  0,  2,  2,  3,  4,  5,  5,  6,  6,  7 };
    
    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_largest)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>   column { { 10, 20, 30, 40, 50, 10, 60 },
                                                        {  1,  1,  1,  1,  1,  0,  0 } };
    fixed_width_column_wrapper<element_type>   values { {  8, 10, 11, 30, 32, 40, 47, 50, 90,  8 },
                                                        {  1,  1,  1,  1,  1,  1,  1,  1,  1,  0 } };
    fixed_width_column_wrapper<index_type>     expect {  0,  1,  1,  3,  3,  4,  4,  5,  5,  7 };
    
    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::AFTER)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_largest)
{
    using element_type = int64_t;

    fixed_width_column_wrapper<element_type>   column { { 10, 20, 30, 40, 50, 10, 60 },
                                                        {  1,  1,  1,  1,  1,  0,  0 } };
    fixed_width_column_wrapper<element_type>   values { {  8, 10, 11, 30, 32, 40, 47, 50, 90,  8 },
                                                        {  1,  1,  1,  1,  1,  1,  1,  1,  1,  0 } };
    fixed_width_column_wrapper<index_type>     expect {  0,  0,  1,  2,  3,  3,  4,  4,  5,  5 };
    
    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             {cudf::table_view{{column}}},
                                                             {cudf::table_view{{values}}},
                                                             {cudf::order::ASCENDING},
                                                             cudf::null_order::AFTER)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table__find_first)
{
    fixed_width_column_wrapper<int32_t> column_0 {  10,  20,  20,  20,  20,  20,  50 };
    fixed_width_column_wrapper<float>   column_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
    fixed_width_column_wrapper<int8_t>  column_2 {  90,  77,  78,  61,  62,  63,  41 };

    fixed_width_column_wrapper<int32_t> values_0 { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    fixed_width_column_wrapper<float>   values_1 { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    fixed_width_column_wrapper<int8_t>  values_2 { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };

    fixed_width_column_wrapper<index_type> expect { 0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  2,  1,  3,  3,  3,  6,  4,  6,  6,  6,  7 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table__find_last)
{
    fixed_width_column_wrapper<int32_t> column_0 {  10,  20,  20,  20,  20,  20,  50 };
    fixed_width_column_wrapper<float>   column_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
    fixed_width_column_wrapper<int8_t>  column_2 {  90,  77,  78,  61,  62,  63,  41 };

    fixed_width_column_wrapper<int32_t> values_0 { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    fixed_width_column_wrapper<float>   values_1 { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    fixed_width_column_wrapper<int8_t>  values_2 { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };


    fixed_width_column_wrapper<index_type> expect { 0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  1,  2,  3,  1,  3,  3,  3,  6,  5,  6,  6,  7,  7 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};


    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table_partial_desc__find_first)
{
    fixed_width_column_wrapper<int32_t>    column_0 {  50,  20,  20,  20,  20,  20,  10 };
    fixed_width_column_wrapper<float>      column_1 {  .7,  .5,  .5,  .7,  .7,  .7, 5.0 };
    fixed_width_column_wrapper<int8_t>     column_2 {  41,  78,  77,  63,  62,  61,  90 };

    fixed_width_column_wrapper<int32_t>    values_0 { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    fixed_width_column_wrapper<float>      values_1 { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    fixed_width_column_wrapper<int8_t>     values_2 { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };

    fixed_width_column_wrapper<index_type> expect   { 7,  7,  7,  7,  6,  7,  6,  6,  7,  7,  7,  7,  6,  1,  3,  2,  1,  3,  3,  3,  3,  3,  4,  3,  1,  0,  0 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table_partial_desc__find_last)
{
    fixed_width_column_wrapper<int32_t>    column_0 {  50,  20,  20,  20,  20,  20,  10 };
    fixed_width_column_wrapper<float>      column_1 {  .7,  .5,  .5,  .7,  .7,  .7, 5.0 };
    fixed_width_column_wrapper<int8_t>     column_2 {  41,  78,  77,  63,  62,  61,  90 };

    fixed_width_column_wrapper<int32_t>    values_0 { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    fixed_width_column_wrapper<float>      values_1 { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    fixed_width_column_wrapper<int8_t>     values_2 { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };

    fixed_width_column_wrapper<index_type> expect   { 7,  7,  7,  7,  6,  7,  7,  6,  7,  7,  7,  7,  6,  1,  3,  3,  2,  3,  3,  3,  3,  3,  5,  3,  1,  1,  0 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table__find_first__nulls_as_smallest)
{
    fixed_width_column_wrapper<int32_t>        column_0 { {  30,  10,  10,  20,  20,  20,  20,  20,  20,  20,  50 },
                                                          {   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1 } };
    fixed_width_column_wrapper<float>          column_1 { {  .5, 6.0, 5.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7 },
                                                          {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 } };
    fixed_width_column_wrapper<int8_t>         column_2 { {  50,  95,  90,  79,  76,  77,  78,  61,  62,  63,  41 },
                                                          {   1,   1,   1,   0,   0,   1,   1,   1,   1,   1,   1 } };

    fixed_width_column_wrapper<int32_t>        values_0 { { 10, 40, 20 },
                                                          {  1,  0,  1 } };
    fixed_width_column_wrapper<float>          values_1 { {  6, .5, .5 },
                                                          {  0,  1,  1 } };
    fixed_width_column_wrapper<int8_t>         values_2 { { 95, 50, 77 },
                                                          {  1,  1,  0 } };

    fixed_width_column_wrapper<index_type>     expect   {  1,  0,  3 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table__find_last__nulls_as_smallest)
{
    fixed_width_column_wrapper<int32_t>        column_0 { {  30,  10,  10,  20,  20,  20,  20,  20,  20,  20,  50 },
                                                          {   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1 } };
    fixed_width_column_wrapper<float>          column_1 { {  .5, 6.0, 5.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7 },
                                                          {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 } };
    fixed_width_column_wrapper<int8_t>         column_2 { {  50,  95,  90,  79,  76,  77,  78,  61,  62,  63,  41 },
                                                          {   1,   1,   1,   0,   0,   1,   1,   1,   1,   1,   1 } };

    fixed_width_column_wrapper<int32_t>        values_0 { { 10, 40, 20 },
                                                          {  1,  0,  1 } };
    fixed_width_column_wrapper<float>          values_1 { {  6, .5, .5 },
                                                          {  0,  1,  1 } };
    fixed_width_column_wrapper<int8_t>         values_2 { { 95, 50, 77 },
                                                          {  1,  1,  0 } };

    fixed_width_column_wrapper<index_type>     expect   {  2,  1,  5 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::BEFORE)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table__find_first__nulls_as_largest)
{
    fixed_width_column_wrapper<int32_t>        column_0 { {  10,  10,  20,  20,  20,  20,  20,  20,  20,  50,  30 },
                                                          {   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0 } };
    fixed_width_column_wrapper<float>          column_1 { { 5.0, 6.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7,  .5 },
                                                          {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 } };
    fixed_width_column_wrapper<int8_t>         column_2 { {  90,  95,  77,  78,  79,  76,  61,  62,  63,  41,  50 },
                                                          {   1,   1,   1,   1,   0,   0,   1,   1,   1,   1,   1 } };

    fixed_width_column_wrapper<int32_t>        values_0 { { 10, 40, 20 },
                                                          {  1,  0,  1 } };
    fixed_width_column_wrapper<float>          values_1 { {  6, .5, .5 },
                                                          {  0,  1,  1 } };
    fixed_width_column_wrapper<int8_t>         values_2 { { 95, 50, 77 },
                                                          {  1,  1,  0 } };

    fixed_width_column_wrapper<index_type>     expect   {  1, 10,  4 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::lower_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::AFTER)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, table__find_last__nulls_as_largest)
{
    fixed_width_column_wrapper<int32_t>        column_0 { {  10,  10,  20,  20,  20,  20,  20,  20,  20,  50,  30 },
                                                          {   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0 } };
    fixed_width_column_wrapper<float>          column_1 { { 5.0, 6.0,  .5,  .5,  .5,  .5,  .7,  .7,  .7,  .7,  .5 },
                                                          {   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1 } };
    fixed_width_column_wrapper<int8_t>         column_2 { {  90,  95,  77,  78,  79,  76,  61,  62,  63,  41,  50 },
                                                          {   1,   1,   1,   1,   0,   0,   1,   1,   1,   1,   1 } };

    fixed_width_column_wrapper<int32_t>        values_0 { { 10, 40, 20 },
                                                          {  1,  0,  1 } };
    fixed_width_column_wrapper<float>          values_1 { {  6, .5, .5 },
                                                          {  0,  1,  1 } };
    fixed_width_column_wrapper<int8_t>         values_2 { { 95, 50, 77 },
                                                          {  1,  1,  0 } };

    fixed_width_column_wrapper<index_type>     expect   {  2, 11,  6 };

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_0.release());
    columns.push_back(column_1.release());
    columns.push_back(column_2.release());

    std::vector<std::unique_ptr<cudf::column>> values;
    values.push_back(values_0.release());
    values.push_back(values_1.release());
    values.push_back(values_2.release());

    cudf::experimental::table input_table(std::move(columns));
    cudf::experimental::table values_table(std::move(values));

    std::vector<cudf::order> order_flags{{cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};

    std::unique_ptr<cudf::column> result{};

    EXPECT_NO_THROW(
                    result = cudf::experimental::upper_bound(
                                                             input_table,
                                                             values_table,
                                                             order_flags,
                                                             cudf::null_order::AFTER)
    );

    expect_columns_equal(*result, expect);
}

TEST_F(SearchTest, contains_true)
{
    using element_type = int64_t;
    bool  expect = true;
    bool  result = false;

    fixed_width_column_wrapper<element_type>    column {0, 1, 17, 19, 23, 29, 71};
    numeric_scalar<element_type>                scalar {23}

    result = cudf::experimental::contains(column, scalar);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_false)
{
    using element_type = int64_t;
    bool  expect = false;
    bool  result = false;

    fixed_width_column_wrapper<element_type>    column {0, 1, 17, 19, 23, 29, 71};
    numeric_scalar<element_type>                scalar {24}

    result = cudf::experimental::contains(column, scalar);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_value)
{
    using element_type = int64_t;
    bool  expect = false;
    bool  result = false;

    fixed_width_column_wrapper<element_type>    column {0, 1, 17, 19, 23, 29, 71};
    numeric_scalar<element_type>                scalar {23, false}

    result = cudf::experimental::contains(column, scalar);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_column)
{
    using element_type = int64_t;
    bool  expect = false;
    bool  result = false;

    fixed_width_column_wrapper<element_type>    column {}
    numeric_scalar<element_type>                scalar {24}

    result = cudf::experimental::contains(column, scalar);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_true)
{
    using element_type = int64_t;
    bool result = false;
    bool expect = true;

    fixed_width_column_wrapper<element_type>    column { { 0, 1, 17, 19, 23, 29, 71},
                                                         { 0,  0,  1,  1,  1,  1,  1 } };
    numeric_scalar<element_type>                scalar {23}

    result = cudf::experimental::contains(column, scalar);

    ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_false)
{
    using element_type = int64_t;
    bool result = false;
    bool expect = false;

    fixed_width_column_wrapper<element_type>    column { { 0, 1, 17, 19, 23, 29, 71},
                                                         { 0,  0,  1,  1,  0,  1,  1 } };
    numeric_scalar<element_type>                scalar {23}

    result = cudf::experimental::contains(column, scalar);

    ASSERT_EQ(result, expect);
}
