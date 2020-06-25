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
#include <nvtext/ngrams_tokenize.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct TextNgramsTokenizeTest : public cudf::test::BaseFixture {
};

TEST_F(TextNgramsTokenizeTest, Tokenize)
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

  {
    cudf::test::strings_column_wrapper expected{"the_fox",
                                                "fox_jumped",
                                                "jumped_over",
                                                "over_the",
                                                "the_dog",
                                                "the_dog",
                                                "dog_chased",
                                                "chased_the",
                                                "the_cat",
                                                "the_cat",
                                                "cat_chased",
                                                "chased_the",
                                                "the_mouse",
                                                "the_mousé",
                                                "mousé_ate",
                                                "ate_the",
                                                "the_cheese"};
    auto results = nvtext::ngrams_tokenize(strings_view);
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected{"the:fox:jumped",
                                                "fox:jumped:over",
                                                "jumped:over:the",
                                                "over:the:dog",
                                                "the:dog:chased",
                                                "dog:chased:the",
                                                "chased:the:cat",
                                                "the:cat:chased",
                                                "cat:chased:the",
                                                "chased:the:mouse",
                                                "the:mousé:ate",
                                                "mousé:ate:the",
                                                "ate:the:cheese"};
    auto results = nvtext::ngrams_tokenize(strings_view, 3, std::string{" "}, std::string{":"});
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected{"the--fox--jumped--over",
                                                "fox--jumped--over--the",
                                                "jumped--over--the--dog",
                                                "the--dog--chased--the",
                                                "dog--chased--the--cat",
                                                "the--cat--chased--the",
                                                "cat--chased--the--mouse",
                                                "the--mousé--ate--the",
                                                "mousé--ate--the--cheese"};
    auto results = nvtext::ngrams_tokenize(strings_view, 4, std::string{" "}, std::string{"--"});
    cudf::test::expect_columns_equal(*results, expected);
  }
}

TEST_F(TextNgramsTokenizeTest, TokenizeOneGram)
{
  cudf::test::strings_column_wrapper strings{"aaa bbb", "  ccc  ddd  ", "eee"};
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected{"aaa", "bbb", "ccc", "ddd", "eee"};
  auto results = nvtext::ngrams_tokenize(strings_view, 1);
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(TextNgramsTokenizeTest, TokenizeEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto results = nvtext::ngrams_tokenize(strings_view);
  EXPECT_EQ(results->size(), 0);
  EXPECT_EQ(results->has_nulls(), false);
}

TEST_F(TextNgramsTokenizeTest, TokenizeErrorTest)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view strings_view(strings);
  EXPECT_THROW(nvtext::ngrams_tokenize(strings_view, 0), cudf::logic_error);
}
