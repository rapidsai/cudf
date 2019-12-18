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
#include <cudf/table/table.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/split/split.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/strings/utilities.h>

#include <vector>
#include <gmock/gmock.h>


struct StringsSplitTest : public cudf::test::BaseFixture {};

TEST_F(StringsSplitTest, Split)
{
    std::vector<const char*> h_strings{ "Héllo thesé", nullptr, "are some", "tést String", "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::strings_column_view strings_view( strings );

    std::vector<const char*> h_expected1{ "Héllo", nullptr, "are", "tést", "" };
    cudf::test::strings_column_wrapper expected1( h_expected1.begin(), h_expected1.end(),
        thrust::make_transform_iterator( h_expected1.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_expected2{ "thesé", nullptr, "some", "String", nullptr };
    cudf::test::strings_column_wrapper expected2( h_expected2.begin(), h_expected2.end(),
        thrust::make_transform_iterator( h_expected2.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<std::unique_ptr<cudf::column>> expected_columns;
    expected_columns.push_back(expected1.release());
    expected_columns.push_back(expected2.release());
    auto expected = std::make_unique<cudf::experimental::table>(std::move(expected_columns));

    auto results = cudf::strings::split(strings_view, cudf::string_scalar(" "));
    EXPECT_TRUE( results->num_columns()==2 );
    cudf::test::expect_tables_equal(*results,*expected);
}

TEST_F(StringsSplitTest, SplitWhitespace)
{
    std::vector<const char*> h_strings{ "Héllo thesé", nullptr, "are\tsome", "tést\nString", "  " };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::strings_column_view strings_view( strings );
    std::vector<const char*> h_expected1{ "Héllo", nullptr, "are", "tést", nullptr };
    cudf::test::strings_column_wrapper expected1( h_expected1.begin(), h_expected1.end(),
        thrust::make_transform_iterator( h_expected1.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_expected2{ "thesé", nullptr, "some", "String", nullptr };
    cudf::test::strings_column_wrapper expected2( h_expected2.begin(), h_expected2.end(),
        thrust::make_transform_iterator( h_expected2.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<std::unique_ptr<cudf::column>> expected_columns;
    expected_columns.push_back(expected1.release());
    expected_columns.push_back(expected2.release());
    auto expected = std::make_unique<cudf::experimental::table>(std::move(expected_columns));

    auto results = cudf::strings::split(strings_view);
    EXPECT_TRUE( results->num_columns()==2 );
    cudf::test::expect_tables_equal(*results,*expected);
}

TEST_F(StringsSplitTest, RSplit)
{
    std::vector<const char*> h_strings{ "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b ", " a  bbb   c" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::strings_column_view strings_view( strings );
    std::vector<const char*> h_expected1{ "héllo", nullptr, "a", "a", "", "ab", "", " a b ", " a  bbb   c" };
    cudf::test::strings_column_wrapper expected1( h_expected1.begin(), h_expected1.end(),
        thrust::make_transform_iterator( h_expected1.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_expected2{ nullptr, nullptr, "bc", "", "ab", "cd", nullptr, nullptr, nullptr };
    cudf::test::strings_column_wrapper expected2( h_expected2.begin(), h_expected2.end(),
        thrust::make_transform_iterator( h_expected2.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_expected3{ nullptr, nullptr, "déf", "bc", "cd", "", nullptr, nullptr, nullptr };
    cudf::test::strings_column_wrapper expected3( h_expected3.begin(), h_expected3.end(),
        thrust::make_transform_iterator( h_expected3.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<std::unique_ptr<cudf::column>> expected_columns;
    expected_columns.push_back(expected1.release());
    expected_columns.push_back(expected2.release());
    expected_columns.push_back(expected3.release());
    auto expected = std::make_unique<cudf::experimental::table>(std::move(expected_columns));

    auto results = cudf::strings::rsplit(strings_view, cudf::string_scalar("_"));
    EXPECT_TRUE( results->num_columns()==3 );
    cudf::test::expect_tables_equal(*results,*expected);
}

TEST_F(StringsSplitTest, RSplitWhitespace)
{
    std::vector<const char*> h_strings{ "héllo", nullptr, "a_bc_déf", "", " a\tb ", " a\r bbb   c" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::strings_column_view strings_view( strings );
    std::vector<const char*> h_expected1{ "héllo", nullptr, "a_bc_déf", nullptr, "a", "a" };
    cudf::test::strings_column_wrapper expected1( h_expected1.begin(), h_expected1.end(),
        thrust::make_transform_iterator( h_expected1.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_expected2{ nullptr, nullptr, nullptr, nullptr, "b", "bbb" };
    cudf::test::strings_column_wrapper expected2( h_expected2.begin(), h_expected2.end(),
        thrust::make_transform_iterator( h_expected2.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_expected3{ nullptr, nullptr, nullptr, nullptr, nullptr, "c" };
    cudf::test::strings_column_wrapper expected3( h_expected3.begin(), h_expected3.end(),
        thrust::make_transform_iterator( h_expected3.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<std::unique_ptr<cudf::column>> expected_columns;
    expected_columns.push_back(expected1.release());
    expected_columns.push_back(expected2.release());
    expected_columns.push_back(expected3.release());
    auto expected = std::make_unique<cudf::experimental::table>(std::move(expected_columns));

    auto results = cudf::strings::rsplit(strings_view);
    EXPECT_TRUE( results->num_columns()==3 );
    cudf::test::expect_tables_equal(*results,*expected);
}

TEST_F(StringsSplitTest, SplitZeroSizeStringsColumns)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::split(zero_size_strings_column);
    EXPECT_TRUE( results->num_columns()==1 );
    cudf::test::expect_strings_empty(results->get_column(0));
    results = cudf::strings::rsplit(zero_size_strings_column);
    EXPECT_TRUE( results->num_columns()==1 );
    cudf::test::expect_strings_empty(results->get_column(0));
}

// This test specifically for https://github.com/rapidsai/custrings/issues/119
TEST_F(StringsSplitTest, AllNullsCase)
{
    std::vector<const char*> h_strings{ nullptr, nullptr, nullptr };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto results = cudf::strings::split(cudf::strings_column_view(strings));
    EXPECT_TRUE( results->num_columns()==1 );
    auto column = results->get_column(0);
    EXPECT_TRUE( column.size()==3 );
    EXPECT_TRUE( column.has_nulls() );
    EXPECT_TRUE( column.null_count()==column.size() );
}

TEST_F(StringsSplitTest, InvalidParameter)
{
    std::vector<const char*> h_strings{ "string left intentionally blank" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    auto strings_view = cudf::strings_column_view(strings);
    EXPECT_THROW( cudf::strings::split(strings_view,cudf::string_scalar("",false)), cudf::logic_error);
    EXPECT_THROW( cudf::strings::rsplit(strings_view,cudf::string_scalar("",false)), cudf::logic_error);
}
