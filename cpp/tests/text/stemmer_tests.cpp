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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <nvtext/stemmer.hpp>

#include <vector>

struct TextStemmerTest : public cudf::test::BaseFixture {
};

TEST_F(TextStemmerTest, PorterStemmer)
{
  std::vector<const char*> h_strings{"abandon",
                                     nullptr,
                                     "abbey",
                                     "cleans",
                                     "trouble",
                                     "",
                                     "yearly",
                                     "tree",
                                     "y",
                                     "by",
                                     "oats",
                                     "ivy",
                                     "private",
                                     "orrery"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  cudf::test::fixed_width_column_wrapper<int32_t> expected(
    {3, 0, 2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 2, 2}, validity);
  auto const results = nvtext::porter_stemmer_measure(cudf::strings_column_view(strings));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextStemmerTest, IsLetterIndex)
{
  std::vector<const char*> h_strings{"abandon",
                                     nullptr,
                                     "abbey",
                                     "cleans",
                                     "trouble",
                                     "",
                                     "yearly",
                                     "tree",
                                     "y",
                                     "by",
                                     "oats",
                                     "ivy",
                                     "private",
                                     "orrery"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  cudf::strings_column_view sv(strings);
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::VOWEL, 0);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::CONSONANT, 0);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::VOWEL, 5);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::CONSONANT, 5);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::VOWEL, -2);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::CONSONANT, -2);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextStemmerTest, IsLetterIndices)
{
  std::vector<const char*> h_strings{"abandon",
                                     nullptr,
                                     "abbey",
                                     "cleans",
                                     "trouble",
                                     "",
                                     "yearly",
                                     "tree",
                                     "y",
                                     "by",
                                     "oats",
                                     "ivy",
                                     "private",
                                     "orrery"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);
  cudf::test::fixed_width_column_wrapper<int32_t> indices(
    {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, -1, -2, -3, -4});

  cudf::strings_column_view sv(strings);
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::VOWEL, indices);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = nvtext::is_letter(sv, nvtext::letter_type::CONSONANT, indices);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextStemmerTest, EmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view sv(strings->view());
  auto results = nvtext::porter_stemmer_measure(sv);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::is_letter(sv, nvtext::letter_type::CONSONANT, 0);
  EXPECT_EQ(results->size(), 0);
  auto indices = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  results      = nvtext::is_letter(sv, nvtext::letter_type::VOWEL, indices->view());
  EXPECT_EQ(results->size(), 0);
}

TEST_F(TextStemmerTest, ErrorTest)
{
  auto empty = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::test::fixed_width_column_wrapper<int32_t> indices({0}, {0});
  EXPECT_THROW(nvtext::is_letter(
                 cudf::strings_column_view(empty->view()), nvtext::letter_type::VOWEL, indices),
               cudf::logic_error);
  cudf::test::strings_column_wrapper strings({"abc"});
  EXPECT_THROW(
    nvtext::is_letter(cudf::strings_column_view(strings), nvtext::letter_type::VOWEL, indices),
    cudf::logic_error);
}
