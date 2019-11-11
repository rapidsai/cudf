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

#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <algorithm>
#include <cmath> 
#include <ctgmath>

template <typename T>
struct UniqueCountCommon : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(UniqueCountCommon, cudf::test::NumericTypes);

TYPED_TEST(UniqueCountCommon, NoNull)
{
    using T = TypeParam;

    std::vector<T> input = {1, 3, 3, 4, 31, 1, 8, 2, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end()};

    cudf::size_type expected = std::set<double>( input.begin(), input.end() ).size();
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, false, false));
}

struct UniqueCount : public cudf::test::BaseFixture {};

TEST_F(UniqueCount, WithNull)
{
    using T = int32_t;

    // Considering 70 as null
    std::vector<T> input =               {1, 3, 3, 70, 31, 1, 8, 2, 0, 70, 1, 70, 10, 40, 31, 42, 0, 42, 8, 5, 70};
    std::vector<cudf::size_type> valid = {1, 1, 1,  0,  1, 1, 1, 1, 1,  0, 1,  0,  1,  1,  1,  1, 1,  1, 1, 1,  0};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

    cudf::size_type expected = std::set<double>( input.begin(), input.end() ).size();
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, false, false));
}


TEST_F(UniqueCount, IgnoringNull)
{
    using T = int32_t;

    // Considering 70 and 3 as null
    std::vector<T> input =               {1, 3, 3, 70, 31, 1, 8, 2, 0, 70, 1, 70, 10, 40, 31, 42, 0, 42, 8, 5, 70};
    std::vector<cudf::size_type> valid = {1, 0, 0,  0,  1, 1, 1, 1, 1,  0, 1,  0,  1,  1,  1,  1, 1,  1, 1, 1,  0};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

    cudf::size_type expected = std::set<T>( input.begin(), input.end() ).size();
    // Removing 2 from expected to remove count for 70 and 3
    EXPECT_EQ(expected - 2, cudf::experimental::unique_count(input_col, true, false));
}

TEST_F(UniqueCount, WithNansAndNull)
{
    using T = float;

    std::vector<T> input =               {1, 3, NAN, 70, 31, 1, 8, 2, 0, 70, 1, 70, 10, 40, 31, NAN, 0, NAN, 8, 5, 70};
    std::vector<cudf::size_type> valid = {1, 0, 0,    0,  1, 1, 1, 1, 1,  0, 1,  0,  1,  1,  1,  1,  1,  1,  1, 1,  0};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

    cudf::size_type expected = std::set<T>( input.begin(), input.end() ).size();
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, false, false));
}

TEST_F(UniqueCount, WithNansOnly)
{
    using T = float;

    std::vector<T> input =               {1, 3, NAN, 70, 31};
    std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

    cudf::size_type expected = 5;
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, false, false));
}

TEST_F(UniqueCount, NansAsNullWithNoNull)
{
    using T = float;

    std::vector<T> input =               {1, 3, NAN, 70, 31};
    std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

    cudf::size_type expected = 5;
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, false, true));
}

TEST_F(UniqueCount, NansAsNullWithNull)
{
    using T = float;

    std::vector<T> input =               {1, 3, NAN, 70, 31};
    std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

    cudf::size_type expected = 4;
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, false, true));
}

TEST_F(UniqueCount, NansAsNullWithIgnoreNull)
{
    using T = float;

    std::vector<T> input =               {1, 3, NAN, 70, 31};
    std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

    cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

    cudf::size_type expected = 3;
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, true, true));
}

TEST_F(UniqueCount, EmptyColumn)
{
    using T = float;

    cudf::test::fixed_width_column_wrapper<T> input_col{std::initializer_list<T>{}};

    cudf::size_type expected = 0;
    EXPECT_EQ(expected, cudf::experimental::unique_count(input_col, true, true));
}





