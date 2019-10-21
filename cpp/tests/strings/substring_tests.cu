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
#include <cudf/strings/substring.hpp>
#include <utilities/error_utils.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsSubstringsTest : public cudf::test::BaseFixture {};

TEST_F(StringsSubstringsTest, Substring)
{
    // cannot initialize std::string with a nullptr so use "<null>" as a place-holder
    cudf::test::strings_column_wrapper h_strings({ "Héllo", "thesé", "<null>", "ARE THE", "tést strings", "" },
                                                 {      1,       1,        0,        1,              1,   1});
    cudf::test::strings_column_wrapper h_expected({ "llo", "esé", "<null>", "E T", "st ", "" },
                                                  {    1,     1,        0,       1,     1,    1 });

    auto strings_view = cudf::strings_column_view(h_strings);
    auto results = cudf::strings::substring(strings_view, 2, 5);
    cudf::test::expect_columns_equal(*results, h_expected);
}

class StringsSubstringParmsTest : public StringsSubstringsTest,
                                  public testing::WithParamInterface<int32_t> {};

TEST_P(StringsSubstringParmsTest, Substring)
{
    std::vector<const char*> h_strings{ "Héllo", "thesé", nullptr, "ARE THE", "tést strings", "" };
    std::vector<const char*> expecteds{ "éllo",  "hesé",  nullptr, "RE THE",  "ést strings", "",
                                        "llo",   "esé",   nullptr, "E THE",   "st strings",  "",
                                        "lo",    "sé",    nullptr, " THE",    "t strings",   "" };

    cudf::size_type start = GetParam();

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::substring(strings_view,start);

    auto begin = expecteds.begin() + ((start-1) * h_strings.size());
    std::vector<const char*> h_expected( begin, begin + h_strings.size() );

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);

}

INSTANTIATE_TEST_CASE_P(StringsSubstringsTest, StringsSubstringParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type,3>{1,2,3}));

