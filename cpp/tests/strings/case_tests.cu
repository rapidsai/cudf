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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/case.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsCaseTest : public cudf::test::BaseFixture {};

TEST_F(StringsCaseTest, ToLower)
{
    std::vector<const char*> h_strings{ "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", "" };
    std::vector<const char*> h_expected{ "éxamples abc", "123 456", nullptr, "are the", "tést strings", "" };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    auto results = cudf::strings::to_lower(strings_view);

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsCaseTest, ToUpper)
{
    std::vector<const char*> h_strings{ "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", "" };
    std::vector<const char*> h_expected{ "ÉXAMPLES ABC", "123 456", nullptr, "ARE THE", "TÉST STRINGS", "" };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    auto results = cudf::strings::to_upper(strings_view);

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsCaseTest, Swapcase)
{
    std::vector<const char*> h_strings{ "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", "" };
    std::vector<const char*> h_expected{ "éXAMPLES AbC", "123 456", nullptr, "are the", "TÉST STRINGS", "" };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    auto results = cudf::strings::swapcase(strings_view);

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsCaseTest, EmptyStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::to_lower(strings_view);
    auto view = results->view();
    cudf::test::expect_strings_empty(results->view());
}
