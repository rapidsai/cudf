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
#include <cudf/strings/replace_re.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <gmock/gmock.h>
#include <vector>


struct StringsReplaceTests : public cudf::test::BaseFixture {};


TEST_F(StringsReplaceTests, ReplaceRegexTest)
{
    std::vector<const char*> h_strings{ "the quick brown fox jumps over the lazy dog",
                                "the fat cat lays next to the other accénted cat",
                                "a slow moving turtlé cannot catch the bird",
                                "which can be composéd together to form a more complete",
                                "thé result does not include the value in the sum in",
                                "", nullptr };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    std::vector<const char*> h_expected{ "= quick brown fox jumps over = lazy dog",
                                "= fat cat lays next to = other accénted cat",
                                "a slow moving turtlé cannot catch = bird",
                                "which can be composéd together to form a more complete",
                                "thé result does not include = value in = sum in",
                                "", nullptr };

    std::string pattern = "(\\bthe\\b)";
    auto results = cudf::strings::replace_re(strings_view,pattern,cudf::string_scalar("="));
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsReplaceTests, ReplaceMultiRegexTest)
{
    std::vector<const char*> h_strings{ "the quick brown fox jumps over the lazy dog",
                                "the fat cat lays next to the other accénted cat",
                                "a slow moving turtlé cannot catch the bird",
                                "which can be composéd together to form a more complete",
                                "thé result does not include the value in the sum in",
                                "", nullptr };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    std::vector<const char*> h_expected{ " quick brown fox jumps over  lazy dog",
                                " fat cat lays next to  other accénted cat",
                                "** slow moving turtlé cannot catch  bird",
                                "which can be composéd together to form ** more complete",
                                "thé result does not include  value N  sum N",
                                "", nullptr };

    std::vector<std::string> patterns{"\\bthe\\b","\\bin\\b","\\ba\\b"};
    std::vector<std::string> h_repls{ "",         "N",       "**"};
    cudf::test::strings_column_wrapper repls( h_repls.begin(), h_repls.end() );
    auto repls_view = cudf::strings_column_view(repls);
    auto results = cudf::strings::replace_re(strings_view,patterns,repls_view);
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsReplaceTests, ReplaceBackrefsRegexTest)
{
    std::vector<const char*> h_strings{ "the quick brown fox jumps over the lazy dog",
                                "the fat cat lays next to the other accénted cat",
                                "a slow moving turtlé cannot catch the bird",
                                "which can be composéd together to form a more complete",
                                "thé result does not include the value in the sum in",
                                "", nullptr };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    std::vector<const char*> h_expected{ "the-quick-brown-fox-jumps-over-the-lazy-dog",
                                 "the-fat-cat-lays-next-to-the-other-accénted-cat",
                                 "a-slow-moving-turtlé-cannot-catch-the-bird",
                                 "which-can-be-composéd-together-to-form-a more-complete",
                                 "thé-result-does-not-include-the-value-in-the-sum-in",
                                 "", nullptr };

    std::string pattern = "(\\w) (\\w)";
    std::string repl_template = "\\1-\\2";
    auto results = cudf::strings::replace_with_backrefs(strings_view,pattern,repl_template);
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsReplaceTests, MediumReplaceRegex)
{
    // This results in 95 regex instructions and falls in the 'medium' range.
    std::string medium_regex = "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com";

    std::vector<const char*> h_strings{
        "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com thats all",
        "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"
    };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::replace_re(strings_view, medium_regex);
    std::vector<const char*> h_expected{" thats all", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz" };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsReplaceTests, LargeReplaceRegex)
{
    // This results in 117 regex instructions and falls in the 'large' range.
    std::string large_regex = "hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz";

    std::vector<const char*> h_strings{
        "zzzz hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz",
        "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"
    };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::replace_re(strings_view, large_regex);
    std::vector<const char*> h_expected{"zzzz ", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz" };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}
