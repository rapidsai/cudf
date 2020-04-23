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
#include <nvtext/tokenize.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct TextTokenizeTest : public cudf::test::BaseFixture {
};

TEST_F(TextTokenizeTest, Tokenize)
{
  std::vector<const char*> h_strings{"the fox jumped over the dog",
                                     "the dog chased  the cat",
                                     " the cat chased the mouse ",
                                     nullptr,
                                     "",
                                     "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected{
    "the", "fox", "jumped", "over", "the",   "dog", "the",   "dog", "chased", "the",   "cat",
    "the", "cat", "chased", "the",  "mouse", "the", "mousé", "ate", "the",    "cheese"};

  auto results = nvtext::tokenize(strings_view, cudf::string_scalar(" "));
  cudf::test::expect_columns_equal(*results, expected);
  results = nvtext::tokenize(strings_view);
  cudf::test::expect_columns_equal(*results, expected);

  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts{6, 5, 5, 0, 0, 5};
  results = nvtext::count_tokens(strings_view, cudf::string_scalar(": #"));
  cudf::test::expect_columns_equal(*results, expected_counts);
  results = nvtext::count_tokens(strings_view);
  cudf::test::expect_columns_equal(*results, expected_counts);
}

TEST_F(TextTokenizeTest, TokenizeMulti)
{
  std::vector<const char*> h_strings{"the fox jumped over the dog",
                                     "the dog chased  the cat",
                                     "the cat chased the mouse ",
                                     nullptr,
                                     "",
                                     "the over ",
                                     "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper delimiters{"the ", "over "};
  cudf::strings_column_view delimiters_view(delimiters);

  auto results = nvtext::tokenize(strings_view, delimiters_view);

  cudf::test::strings_column_wrapper expected{
    "fox jumped ", "dog", "dog chased  ", "cat", "cat chased ", "mouse ", "mousé ate ", "cheese"};
  cudf::test::expect_columns_equal(*results, expected);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts{2, 2, 2, 0, 0, 0, 2};
  results = nvtext::count_tokens(strings_view, delimiters_view);
  cudf::test::expect_columns_equal(*results, expected_counts);
}

TEST_F(TextTokenizeTest, TokenizeErrorTest)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view strings_view(strings);

  {
    cudf::test::strings_column_wrapper delimiters{};  // empty delimiters
    cudf::strings_column_view delimiters_view(delimiters);
    EXPECT_THROW(nvtext::tokenize(strings_view, delimiters_view), cudf::logic_error);
    EXPECT_THROW(nvtext::count_tokens(strings_view, delimiters_view), cudf::logic_error);
  }
  {
    cudf::test::strings_column_wrapper delimiters({"", ""}, {0, 0});  // null delimiters
    cudf::strings_column_view delimiters_view(delimiters);
    EXPECT_THROW(nvtext::tokenize(strings_view, delimiters_view), cudf::logic_error);
    EXPECT_THROW(nvtext::count_tokens(strings_view, delimiters_view), cudf::logic_error);
  }
}

TEST_F(TextTokenizeTest, TokenizeEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto results = nvtext::tokenize(strings_view);
  EXPECT_EQ(results->size(), 0);
  EXPECT_EQ(results->has_nulls(), false);
  results = nvtext::count_tokens(strings_view);
  EXPECT_EQ(results->size(), 0);
  EXPECT_EQ(results->has_nulls(), false);
}
