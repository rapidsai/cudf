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
#include <cudf/strings/replace.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsReplaceTest : public cudf::test::BaseFixture {};

TEST_F(StringsReplaceTest, Replace)
{
    std::vector<const char*> h_strings{ "the quick brown fox jumps over the lazy dog",
                                "the fat cat lays next to the other accénted cat",
                                 "a slow moving turtlé cannot catch the bird",
                                 "which can be composéd together to form a more complete",
                                 "The result does not include the value in the sum in",
                                 "", nullptr };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    {
        // replace all occurences of 'the ' with '++++ '
        auto results = cudf::strings::replace(strings_view, cudf::string_scalar("the "), cudf::string_scalar("++++ "));
        std::vector<const char*> h_expected{ "++++ quick brown fox jumps over ++++ lazy dog",
                                    "++++ fat cat lays next to ++++ other accénted cat",
                                     "a slow moving turtlé cannot catch ++++ bird",
                                     "which can be composéd together to form a more complete",
                                     "The result does not include ++++ value in ++++ sum in",
                                     "", nullptr };
        cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
            thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        // only remove the first occurrence of 'the '
        auto results = cudf::strings::replace(strings_view, cudf::string_scalar("the "), cudf::string_scalar(""), 1);
        std::vector<const char*> h_expected{ "quick brown fox jumps over the lazy dog",
                                    "fat cat lays next to the other accénted cat",
                                    "a slow moving turtlé cannot catch bird",
                                    "which can be composéd together to form a more complete",
                                    "The result does not include value in the sum in",
                                    "", nullptr };
        cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
            thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}


TEST_F(StringsReplaceTest, EmptyStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::replace(strings_view,cudf::string_scalar("not"), cudf::string_scalar("pertinent"));
    auto view = results->view();
    cudf::test::expect_strings_empty(results->view());
}
