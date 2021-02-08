/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <tests/strings/utilities.h>
#include <cudf/column/column.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsCharsTest : public cudf::test::BaseFixture {
};

class StringsCharsTestTypes
  : public StringsCharsTest,
    public testing::WithParamInterface<cudf::strings::string_character_types> {
};

TEST_P(StringsCharsTestTypes, AllTypes)
{
  std::vector<const char*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  bool expecteds[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   // decimal
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,   // numeric
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,   // digit
                      1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,   // alpha
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,   // space
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   // upper
                      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};  // lower

  auto is_parm = GetParam();

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::all_characters_of_type(strings_view, is_parm);

  int x             = static_cast<int>(is_parm);
  int index         = 0;
  int strings_count = static_cast<int>(h_strings.size());
  while (x >>= 1) ++index;
  bool* sub_expected = &expecteds[index * strings_count];

  cudf::test::fixed_width_column_wrapper<bool> expected(
    sub_expected,
    sub_expected + strings_count,
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsCharsTestAllTypes,
                        StringsCharsTestTypes,
                        testing::ValuesIn(std::array<cudf::strings::string_character_types, 7>{
                          cudf::strings::string_character_types::DECIMAL,
                          cudf::strings::string_character_types::NUMERIC,
                          cudf::strings::string_character_types::DIGIT,
                          cudf::strings::string_character_types::ALPHA,
                          cudf::strings::string_character_types::SPACE,
                          cudf::strings::string_character_types::UPPER,
                          cudf::strings::string_character_types::LOWER}));

TEST_F(StringsCharsTest, LowerUpper)
{
  cudf::test::strings_column_wrapper strings({"a1", "A1", "a!", "A!", "!1", "aA"});
  auto strings_view = cudf::strings_column_view(strings);
  auto verify_types =
    cudf::strings::string_character_types::LOWER | cudf::strings::string_character_types::UPPER;
  {
    auto results = cudf::strings::all_characters_of_type(
      strings_view, cudf::strings::string_character_types::LOWER, verify_types);
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 0, 1, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::all_characters_of_type(
      strings_view, cudf::strings::string_character_types::UPPER, verify_types);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCharsTest, Alphanumeric)
{
  std::vector<const char*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::all_characters_of_type(
    strings_view, cudf::strings::string_character_types::ALPHANUM);

  std::vector<bool> h_expected{1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCharsTest, AlphaNumericSpace)
{
  std::vector<const char*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto types =
    cudf::strings::string_character_types::ALPHANUM | cudf::strings::string_character_types::SPACE;
  auto results = cudf::strings::all_characters_of_type(
    strings_view, (cudf::strings::string_character_types)types);

  std::vector<bool> h_expected{1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCharsTest, Numerics)
{
  std::vector<const char*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto types = cudf::strings::string_character_types::DIGIT |
               cudf::strings::string_character_types::DECIMAL |
               cudf::strings::string_character_types::NUMERIC;
  auto results = cudf::strings::all_characters_of_type(
    strings_view, (cudf::strings::string_character_types)types);

  std::vector<bool> h_expected{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCharsTest, Integers)
{
  cudf::test::strings_column_wrapper strings1(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", ""});
  auto results = cudf::strings::is_integer(cudf::strings_column_view(strings1));
  cudf::test::fixed_width_column_wrapper<bool> expected1({1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected1);
  EXPECT_FALSE(cudf::strings::all_integer(cudf::strings_column_view(strings1)));

  cudf::test::strings_column_wrapper strings2(
    {"0", "+0", "-0", "1234567890", "-27341132", "+012", "023", "-045"});
  results = cudf::strings::is_integer(cudf::strings_column_view(strings2));
  cudf::test::fixed_width_column_wrapper<bool> expected2({1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
  EXPECT_TRUE(cudf::strings::all_integer(cudf::strings_column_view(strings2)));
}

TEST_F(StringsCharsTest, Floats)
{
  cudf::test::strings_column_wrapper strings1({"+175",
                                               "-9.8",
                                               "7+2",
                                               "+-4",
                                               "6.7e17",
                                               "-1.2e-5",
                                               "e",
                                               ".e",
                                               "1.e+-2",
                                               "00.00",
                                               "1.0e+1.0",
                                               "1.2.3",
                                               "+",
                                               "--",
                                               ""});
  auto results = cudf::strings::is_float(cudf::strings_column_view(strings1));
  cudf::test::fixed_width_column_wrapper<bool> expected1(
    {1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected1);
  EXPECT_FALSE(cudf::strings::all_float(cudf::strings_column_view(strings1)));

  cudf::test::strings_column_wrapper strings2(
    {"+175", "-34", "9.8", "1234567890", "6.7e17", "-917.2e5"});
  results = cudf::strings::is_float(cudf::strings_column_view(strings2));
  cudf::test::fixed_width_column_wrapper<bool> expected2({1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
  EXPECT_TRUE(cudf::strings::all_float(cudf::strings_column_view(strings2)));
}

TEST_F(StringsCharsTest, EmptyStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""});
  auto strings_view = cudf::strings_column_view(strings);
  cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 0});
  auto results = cudf::strings::all_characters_of_type(
    strings_view, cudf::strings::string_character_types::ALPHANUM);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::is_integer(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  EXPECT_FALSE(cudf::strings::all_integer(strings_view));
  results = cudf::strings::is_float(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  EXPECT_FALSE(cudf::strings::all_float(strings_view));
}

TEST_F(StringsCharsTest, FilterCharTypes)
{
  // The example strings are based on issue 5520
  cudf::test::strings_column_wrapper strings(
    {"abc£def", "01234 56789", "℉℧ is not alphanumeric", "but Αγγλικά is", ""});
  auto results =
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar(" "),
                                             cudf::strings::string_character_types::ALPHANUM);
  {
    cudf::test::strings_column_wrapper expected(
      {"abc def", "01234 56789", "   is not alphanumeric", "but Αγγλικά is", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results = cudf::strings::filter_characters_of_type(
    cudf::strings_column_view(strings), cudf::strings::string_character_types::ALPHANUM);
  {
    cudf::test::strings_column_wrapper expected({"£", " ", "℉℧   ", "  ", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results = cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                                     cudf::strings::string_character_types::SPACE);
  {
    cudf::test::strings_column_wrapper expected(
      {"abc£def", "0123456789", "℉℧isnotalphanumeric", "butΑγγλικάis", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results =
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar("+"),
                                             cudf::strings::string_character_types::SPACE);
  {
    cudf::test::strings_column_wrapper expected(
      {"+++++++", "+++++ +++++", "++ ++ +++ ++++++++++++", "+++ +++++++ ++", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results = cudf::strings::filter_characters_of_type(
    cudf::strings_column_view(strings), cudf::strings::string_character_types::NUMERIC);
  {
    cudf::test::strings_column_wrapper expected(
      {"abc£def", " ", "℉℧ is not alphanumeric", "but Αγγλικά is", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results =
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar(""),
                                             cudf::strings::string_character_types::NUMERIC);
  {
    cudf::test::strings_column_wrapper expected({"", "0123456789", "", "", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCharsTest, FilterCharTypesErrors)
{
  cudf::test::strings_column_wrapper strings({"strings left intentionally blank"});
  EXPECT_THROW(
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar(""),
                                             cudf::strings::string_character_types::ALL_TYPES),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALPHANUM,
                                             cudf::string_scalar(""),
                                             cudf::strings::string_character_types::NUMERIC),
    cudf::logic_error);
}

TEST_F(StringsCharsTest, EmptyStringsColumn)
{
  cudf::test::strings_column_wrapper strings;
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::all_characters_of_type(
    strings_view, cudf::strings::string_character_types::ALPHANUM);
  EXPECT_EQ(cudf::type_id::BOOL8, results->view().type().id());
  EXPECT_EQ(0, results->view().size());

  results = cudf::strings::is_integer(strings_view);
  EXPECT_EQ(cudf::type_id::BOOL8, results->view().type().id());
  EXPECT_EQ(0, results->view().size());

  results = cudf::strings::is_float(strings_view);
  EXPECT_EQ(cudf::type_id::BOOL8, results->view().type().id());
  EXPECT_EQ(0, results->view().size());

  results = cudf::strings::filter_characters_of_type(
    strings_view, cudf::strings::string_character_types::NUMERIC);
  EXPECT_EQ(cudf::type_id::STRING, results->view().type().id());
  EXPECT_EQ(0, results->view().size());
}
