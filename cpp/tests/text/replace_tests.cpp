/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <nvtext/replace.hpp>

#include <vector>

struct TextReplaceTest : public cudf::test::BaseFixture {
};

TEST_F(TextReplaceTest, ReplaceTokens)
{
  std::vector<const char*> h_strings{"the fox jumped over the dog",
                                     "is theme of the thesis",
                                     nullptr,
                                     "",
                                     "no change",
                                     "thé is the cheese is"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::strings_column_wrapper targets({"is", "the"});
  cudf::test::strings_column_wrapper repls({"___", ""});
  std::vector<const char*> h_expected{" fox jumped over  dog",
                                      "___ theme of  thesis",
                                      nullptr,
                                      "",
                                      "no change",
                                      "thé ___  cheese ___"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = nvtext::replace_tokens(cudf::strings_column_view(strings),
                                        cudf::strings_column_view(targets),
                                        cudf::strings_column_view(repls));
  cudf::test::expect_columns_equal(*results, expected);
  results = nvtext::replace_tokens(cudf::strings_column_view(strings),
                                   cudf::strings_column_view(targets),
                                   cudf::strings_column_view(repls),
                                   cudf::string_scalar("o "));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(TextReplaceTest, ReplaceTokensSingleRepl)
{
  cudf::test::strings_column_wrapper strings({"this\t is that", "is then \tis", "us them is us"});
  cudf::test::strings_column_wrapper targets({"is", "us"});
  cudf::test::strings_column_wrapper repls({"_"});
  cudf::test::strings_column_wrapper expected({"this\t _ that", "_ then \t_", "_ them _ _"});

  auto results = nvtext::replace_tokens(cudf::strings_column_view(strings),
                                        cudf::strings_column_view(targets),
                                        cudf::strings_column_view(repls));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(TextReplaceTest, ReplaceTokensEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto const results = nvtext::replace_tokens(strings_view, strings_view, strings_view);
  EXPECT_EQ(results->size(), 0);
  EXPECT_EQ(results->has_nulls(), false);
}

TEST_F(TextReplaceTest, ReplaceTokensErrorTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::STRING});
  cudf::strings_column_view strings_view(strings->view());
  cudf::test::strings_column_wrapper notnulls({"", "", ""});
  cudf::strings_column_view notnulls_view(notnulls);
  cudf::test::strings_column_wrapper nulls({"", ""}, {0, 0});
  cudf::strings_column_view nulls_view(nulls);

  EXPECT_THROW(nvtext::replace_tokens(strings_view, nulls_view, notnulls_view), cudf::logic_error);
  EXPECT_THROW(nvtext::replace_tokens(strings_view, notnulls_view, nulls_view), cudf::logic_error);
  EXPECT_THROW(nvtext::replace_tokens(notnulls_view, notnulls_view, strings_view),
               cudf::logic_error);
  EXPECT_THROW(
    nvtext::replace_tokens(notnulls_view, nulls_view, strings_view, cudf::string_scalar("", false)),
    cudf::logic_error);
}
