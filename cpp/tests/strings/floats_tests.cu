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
#include <cudf/strings/convert/convert_floats.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsConvertTest : public cudf::test::BaseFixture {};



TEST_F(StringsConvertTest, ToFloats)
{
    std::vector<const char*> h_strings{ "1234", nullptr,
            "-876", "543.2", "-0.12", ".25", "-.002",
            "", "NaN", "abc123", "123abc", "456e", "-1.78e+5",
            "-122.33644782123456789", "12e+309" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    float nanval = std::numeric_limits<float>::quiet_NaN();
    float infval = std::numeric_limits<float>::infinity();
    std::vector<float> h_expected{ 1234.0, 0,
            -876.0, 543.2, -0.12, 0.25, -0.002,
            0, nanval, 0, 123.0, 456.0, -178000.0,
            -122.3364486694336, infval };

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::to_floats(strings_view, cudf::data_type{cudf::FLOAT32} );

    cudf::test::fixed_width_column_wrapper<float> expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, FromFloats)
{
    std::vector<float> h_floats{ 100, 654321.25, -12761.125, 0, 5, -4, std::numeric_limits<float>::quiet_NaN() };
    std::vector<const char*> h_expected{ "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN" };

    cudf::test::fixed_width_column_wrapper<float> floats( h_floats.begin(), h_floats.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    auto results = cudf::strings::from_floats(floats);

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::FLOAT32}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::from_floats(zero_size_column);
    cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeIntegersColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::to_floats(zero_size_column, cudf::data_type{cudf::FLOAT32});
    EXPECT_EQ(0,results->size());
}
