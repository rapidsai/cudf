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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/convert.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsConvertTest : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, ToInteger)
{
    std::vector<const char*> h_strings{ "eee", "1234", nullptr, "", "-9832", "93.24", "765é", "-1.78e+5" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<int32_t> h_expected{ 0, 1234, 0, 0, -9832, 93, 765, -1 };

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::to_integers(strings_view);

    cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, FromInteger)
{
    std::vector<int32_t> h_integers{ 100, 987654321, 0, 0, -12761, 0, 5, -4 };
    std::vector<const char*> h_expected{ "100", "987654321", nullptr, "0", "-12761", "0", "5", "-4" };

    cudf::test::fixed_width_column_wrapper<int32_t> integers( h_integers.begin(), h_integers.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    auto results = cudf::strings::from_integers(integers);

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::from_integers(zero_size_column);
    cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeIntegersColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::to_integers(zero_size_column);
    EXPECT_EQ(0,results->size());
}
