/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct TextTokenizeTest : public cudf::test::BaseFixture {};

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
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::tokenize(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts{6, 5, 5, 0, 0, 5};
  results = nvtext::count_tokens(strings_view, cudf::string_scalar(": #"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_counts);
  results = nvtext::count_tokens(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_counts);
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
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts{2, 2, 2, 0, 0, 0, 2};
  results = nvtext::count_tokens(strings_view, delimiters_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_counts);
}

TEST_F(TextTokenizeTest, TokenizeErrorTest)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view strings_view(strings);

  {
    cudf::test::strings_column_wrapper delimiters;  // empty delimiters
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

TEST_F(TextTokenizeTest, CharacterTokenize)
{
  std::vector<const char*> h_strings{"the mousé ate the cheese", nullptr, ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper expected{"t", "h", "e", " ", "m", "o", "u", "s",
                                              "é", " ", "a", "t", "e", " ", "t", "h",
                                              "e", " ", "c", "h", "e", "e", "s", "e"};

  auto results = nvtext::character_tokenize(cudf::strings_column_view(strings));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextTokenizeTest, TokenizeEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::test::strings_column_wrapper all_empty({"", "", ""});
  cudf::test::strings_column_wrapper all_null({"", "", ""}, {0, 0, 0});
  cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 0});

  auto results = nvtext::tokenize(cudf::strings_column_view(strings->view()));
  EXPECT_EQ(results->size(), 0);
  results = nvtext::tokenize(cudf::strings_column_view(all_empty));
  EXPECT_EQ(results->size(), 0);
  results = nvtext::tokenize(cudf::strings_column_view(all_null));
  EXPECT_EQ(results->size(), 0);
  results = nvtext::count_tokens(cudf::strings_column_view(strings->view()));
  EXPECT_EQ(results->size(), 0);
  results = nvtext::count_tokens(cudf::strings_column_view(all_empty));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::count_tokens(cudf::strings_column_view(all_null));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::character_tokenize(cudf::strings_column_view(strings->view()));
  EXPECT_EQ(results->size(), 0);
  results = nvtext::character_tokenize(cudf::strings_column_view(all_empty));
  EXPECT_EQ(results->size(), 0);
  results = nvtext::character_tokenize(cudf::strings_column_view(all_null));
  EXPECT_EQ(results->size(), 0);
}

TEST_F(TextTokenizeTest, Detokenize)
{
  cudf::test::strings_column_wrapper strings{
    "the", "fox", "jumped", "over",   "the", "dog",   "the", "dog",   "chased", "the",
    "cat", "the", "cat",    "chased", "the", "mouse", "the", "mousé", "ate",    "cheese"};

  {
    cudf::test::fixed_width_column_wrapper<int32_t> rows{0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                                         1, 2, 2, 2, 2, 2, 3, 3, 3, 3};
    auto results = nvtext::detokenize(cudf::strings_column_view(strings), rows);
    cudf::test::strings_column_wrapper expected{"the fox jumped over the dog",
                                                "the dog chased the cat",
                                                "the cat chased the mouse",
                                                "the mousé ate cheese"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int16_t> rows{0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                                         1, 2, 2, 2, 2, 2, 3, 3, 3, 0};
    auto results =
      nvtext::detokenize(cudf::strings_column_view(strings), rows, cudf::string_scalar("_"));
    cudf::test::strings_column_wrapper expected{"the_fox_jumped_over_the_dog_cheese",
                                                "the_dog_chased_the_cat",
                                                "the_cat_chased_the_mouse",
                                                "the_mousé_ate"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextTokenizeTest, DetokenizeErrors)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view strings_view(strings);

  cudf::test::fixed_width_column_wrapper<int32_t> one({0});
  cudf::test::fixed_width_column_wrapper<int32_t> none;

  EXPECT_THROW(nvtext::detokenize(strings_view, none), cudf::logic_error);
  EXPECT_THROW(nvtext::detokenize(strings_view, one, cudf::string_scalar("", false)),
               cudf::logic_error);
}
