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

#include <nvtext/normalize.hpp>

#include <vector>

struct TextNormalizeTest : public cudf::test::BaseFixture {
};

TEST_F(TextNormalizeTest, NormalizeSpaces)
{
  std::vector<const char*> h_strings{"the\t fox  jumped over the      dog",
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

  std::vector<const char*> h_expected{"the fox jumped over the dog",
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
  cudf::test::expect_columns_equal(*results, expected);
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
  cudf::test::strings_column_wrapper strings({"", "", ""}, {0, 0, 0});
  cudf::strings_column_view strings_view(strings);
  auto results = nvtext::normalize_characters(strings_view, false);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(TextNormalizeTest, NormalizeCharacters)
{
  // These include punctuation, accents, whitespace, and CJK characters
  std::vector<const char*> h_strings{"abc£def",
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
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = nvtext::normalize_characters(cudf::strings_column_view(strings), false);
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
    cudf::test::expect_columns_equal(*results, expected);
  }
}
