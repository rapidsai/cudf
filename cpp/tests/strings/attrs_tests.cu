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

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/attributes.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.cuh>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringAttributesTest : public cudf::test::BaseFixture {};

TEST_F(StringAttributesTest, StringDataCounts)
{
    std::vector<const char*> h_strings{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
    auto strings_column = cudf::test::create_strings_column(h_strings);
    auto strings_view = cudf::strings_column_view(strings_column->view());

    std::vector<cudf::bitmask_type> h_nulls{ 123 };
    {
        std::vector<int32_t> h_expected{ 3, 2, 0, 0, 2, 3, 3 };
        auto d_expected = cudf::test::create_numeric_column(h_expected, h_nulls, 1);

        auto results = cudf::strings::characters_counts(strings_view);
        cudf::test::expect_columns_equal(*results, *d_expected);
    }
    {
        std::vector<int32_t> h_expected{ 3, 2, 0, 0, 2, 3, 6 };
        auto d_expected = cudf::test::create_numeric_column(h_expected, h_nulls, 1);

        auto results = cudf::strings::bytes_counts(strings_view);
        cudf::test::expect_columns_equal(*results, *d_expected);
    }
}

TEST_F(StringAttributesTest, CodePoints)
{
    std::vector<const char*> h_strings{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
    auto strings_column = cudf::test::create_strings_column(h_strings);
    auto strings_view = cudf::strings_column_view(strings_column->view());

    {
        std::vector<int32_t> h_expected{ 101, 101, 101, 98, 98, 97, 97, 98, 98, 98, 50089, 50089, 50089 };
        std::vector<cudf::bitmask_type> h_nulls{ 0xFF };
        auto d_expected = cudf::test::create_numeric_column(h_expected, h_nulls, 0);

        auto results = cudf::strings::code_points(strings_view);
        cudf::test::expect_columns_equal(*results, *d_expected);
    }
}

TEST_F(StringAttributesTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    cudf::column_view expected_column( cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);

    auto results = cudf::strings::bytes_counts(strings_view);
    cudf::test::expect_columns_equal(results->view(), expected_column);
    results = cudf::strings::characters_counts(strings_view);
    cudf::test::expect_columns_equal(results->view(), expected_column);
    results = cudf::strings::code_points(strings_view);
    cudf::test::expect_columns_equal(results->view(), expected_column);
}
