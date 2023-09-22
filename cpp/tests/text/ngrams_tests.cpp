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
  auto const separator = cudf::string_scalar("_");

  {
    cudf::test::strings_column_wrapper expected{
      "the_fox", "fox_jumped", "jumped_over", "over_thé", "thé_dog"};
    auto const results = nvtext::generate_ngrams(strings_view, 2, separator);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    cudf::test::strings_column_wrapper expected{
      "the_fox_jumped", "fox_jumped_over", "jumped_over_thé", "over_thé_dog"};
    auto const results = nvtext::generate_ngrams(strings_view, 3, separator);
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
  auto const separator = cudf::string_scalar("_");

  cudf::strings_column_view strings_view(strings);
  {
    auto const results = nvtext::generate_ngrams(strings_view, 3, separator);
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
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto const separator = cudf::string_scalar("_");

  auto results =
    nvtext::generate_ngrams(cudf::strings_column_view(zero_size_strings_column), 2, separator);
  cudf::test::expect_column_empty(results->view());
  results = nvtext::generate_character_ngrams(cudf::strings_column_view(zero_size_strings_column));
  cudf::test::expect_column_empty(results->view());
}

TEST_F(TextGenerateNgramsTest, Errors)
{
  cudf::test::strings_column_wrapper strings{""};
  auto const separator = cudf::string_scalar("_");
  // invalid parameter value
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings), 1, separator),
               cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings), 1),
               cudf::logic_error);
  // not enough strings to generate ngrams
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings), 3, separator),
               cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings), 3),
               cudf::logic_error);

  cudf::test::strings_column_wrapper strings_no_tokens({"", "", "", ""}, {1, 0, 1, 0});
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings_no_tokens), 2, separator),
               cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings_no_tokens)),
               cudf::logic_error);
}

TEST_F(TextGenerateNgramsTest, NgramsHash)
{
  auto input =
    cudf::test::strings_column_wrapper({"the quick brown fox", "jumped over the lazy dog."});

  auto view    = cudf::strings_column_view(input);
  auto results = nvtext::hash_character_ngrams(view);

  using LCW = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW expected({LCW{2169381797u, 3924065905u, 1634753325u, 3766025829u,  771291085u,
                    2286480985u, 2815102125u, 2383213292u, 1587939873u, 3417728802u,
                     741580288u, 1721912990u, 3322339040u, 2530504717u, 1448945146u},
                LCW{3542029734u, 2351937583u, 2373822151u, 2610417165u, 1303810911u,
                    2541942822u, 1736466351u, 3466558519u,  408633648u, 1698719372u,
                     620653030u,   16851044u,  608863326u,  948572753u, 3672211877u,
                    4097451013u, 1444462157u, 3762829398u,  743082018u, 2953783152u,
                    2319357747u}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextGenerateNgramsTest, NgramsHashErrors)
{
  auto input = cudf::test::strings_column_wrapper({"1", "2", "3"});
  auto view  = cudf::strings_column_view(input);

  // invalid parameter value
  EXPECT_THROW(nvtext::hash_character_ngrams(view, 1), cudf::logic_error);
  // strings not long enough to generate ngrams
  EXPECT_THROW(nvtext::hash_character_ngrams(view), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
