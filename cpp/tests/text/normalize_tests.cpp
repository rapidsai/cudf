/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/normalize.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct TextNormalizeTest : public cudf::test::BaseFixture {};

TEST_F(TextNormalizeTest, NormalizeSpaces)
{
  std::vector<char const*> h_strings{"the\t fox  jumped over the      dog",
                                     "the dog\f chased  the cat\r",
                                     " the cat  chaséd  the mouse\n",
                                     nullptr,
                                     "",
                                     " \r\t\n",
                                     "no change",
                                     "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::strings_column_view strings_view(strings);

  std::vector<char const*> h_expected{"the fox jumped over the dog",
                                      "the dog chased the cat",
                                      "the cat chaséd the mouse",
                                      nullptr,
                                      "",
                                      "",
                                      "no change",
                                      "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto const results = nvtext::normalize_spaces(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, NormalizeEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto results = nvtext::normalize_spaces(strings_view);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::normalize_characters(strings_view, true);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::normalize_characters(strings_view, false);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(TextNormalizeTest, AllNullStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {false, false, false});
  cudf::strings_column_view strings_view(strings);
  auto results = nvtext::normalize_spaces(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
  results = nvtext::normalize_characters(strings_view, false);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
}

TEST_F(TextNormalizeTest, SomeNullStrings)
{
  cudf::test::strings_column_wrapper strings({"", ".", "a"}, {false, true, true});
  cudf::strings_column_view strings_view(strings);
  auto results = nvtext::normalize_characters(strings_view, false);
  cudf::test::strings_column_wrapper expected({"", " . ", "a"}, {false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, NormalizeCharacters)
{
  // These include punctuation, accents, whitespace, and CJK characters
  std::vector<char const*> h_strings{"abc£def",
                                     nullptr,
                                     "éè â îô\taeio",
                                     "\tĂĆĖÑ  Ü",
                                     "ACEN U",
                                     "P^NP",
                                     "$41.07",
                                     "[a,b]",
                                     "丏丟",
                                     ""};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);
  cudf::strings_column_view strings_view(strings);
  {
    auto results = nvtext::normalize_characters(strings_view, true);
    cudf::test::strings_column_wrapper expected({"abc£def",
                                                 "",
                                                 "ee a io aeio",
                                                 " acen  u",
                                                 "acen u",
                                                 "p ^ np",
                                                 " $ 41 . 07",
                                                 " [ a , b ] ",
                                                 " 丏  丟 ",
                                                 ""},
                                                validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = nvtext::normalize_characters(strings_view, false);
    cudf::test::strings_column_wrapper expected({"abc£def",
                                                 "",
                                                 "éè â îô aeio",
                                                 " ĂĆĖÑ  Ü",
                                                 "ACEN U",
                                                 "P ^ NP",
                                                 " $ 41 . 07",
                                                 " [ a , b ] ",
                                                 " 丏  丟 ",
                                                 ""},
                                                validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextNormalizeTest, NormalizeSlicedColumn)
{
  cudf::test::strings_column_wrapper strings(
    {"abc£def", "éè â îô\taeio", "ACEN U", "P^NP", "$41.07", "[a,b]", "丏丟"});

  std::vector<cudf::column_view> sliced = cudf::split(strings, {4});
  auto results = nvtext::normalize_characters(cudf::strings_column_view(sliced.front()), true);
  cudf::test::strings_column_wrapper expected({"abc£def", "ee a io aeio", "acen u", "p ^ np"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::normalize_characters(cudf::strings_column_view(sliced[1]), false);
  cudf::test::strings_column_wrapper expected2({" $ 41 . 07", " [ a , b ] ", " 丏  丟 "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
}
