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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/generate_ngrams.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct TextGenerateNgramsTest : public cudf::test::BaseFixture {};

TEST_F(TextGenerateNgramsTest, Ngrams)
{
  cudf::test::strings_column_wrapper strings{"the", "fox", "jumped", "over", "thé", "dog"};
  cudf::strings_column_view strings_view(strings);

  {
    cudf::test::strings_column_wrapper expected{
      "the_fox", "fox_jumped", "jumped_over", "over_thé", "thé_dog"};
    auto const results = nvtext::generate_ngrams(strings_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    cudf::test::strings_column_wrapper expected{
      "the_fox_jumped", "fox_jumped_over", "jumped_over_thé", "over_thé_dog"};
    auto const results = nvtext::generate_ngrams(strings_view, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected{"th",
                                                "he",
                                                "fo",
                                                "ox",
                                                "ju",
                                                "um",
                                                "mp",
                                                "pe",
                                                "ed",
                                                "ov",
                                                "ve",
                                                "er",
                                                "th",
                                                "hé",
                                                "do",
                                                "og"};
    auto const results = nvtext::generate_character_ngrams(strings_view, 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected{
      "the", "fox", "jum", "ump", "mpe", "ped", "ove", "ver", "thé", "dog"};
    auto const results = nvtext::generate_character_ngrams(strings_view, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextGenerateNgramsTest, NgramsWithNulls)
{
  std::vector<char const*> h_strings{"the", "fox", "", "jumped", "over", nullptr, "the", "dog"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::strings_column_view strings_view(strings);
  {
    auto const results = nvtext::generate_ngrams(strings_view, 3);
    cudf::test::strings_column_wrapper expected{
      "the_fox_jumped", "fox_jumped_over", "jumped_over_the", "over_the_dog"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected{
      "the", "fox", "jum", "ump", "mpe", "ped", "ove", "ver", "the", "dog"};
    auto const results = nvtext::generate_character_ngrams(strings_view, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextGenerateNgramsTest, Empty)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results = nvtext::generate_ngrams(cudf::strings_column_view(zero_size_strings_column));
  cudf::test::expect_column_empty(results->view());
  results = nvtext::generate_character_ngrams(cudf::strings_column_view(zero_size_strings_column));
  cudf::test::expect_column_empty(results->view());
}

TEST_F(TextGenerateNgramsTest, Errors)
{
  cudf::test::strings_column_wrapper strings{""};
  // invalid parameter value
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings), 1), cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings), 1),
               cudf::logic_error);
  // not enough strings to generate ngrams
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings), 3), cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings), 3),
               cudf::logic_error);

  std::vector<char const*> h_strings{"", nullptr, "", nullptr};
  cudf::test::strings_column_wrapper strings_no_tokens(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings_no_tokens)),
               cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings_no_tokens)),
               cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
