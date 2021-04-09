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

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <tests/strings/utilities.h>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <thrust/iterator/transform_iterator.h>

struct StringsCombineTest : public cudf::test::BaseFixture {
};

TEST_F(StringsCombineTest, Concatenate)
{
  std::vector<const char*> h_strings1{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings1(
    h_strings1.begin(),
    h_strings1.end(),
    thrust::make_transform_iterator(h_strings1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_strings2{"xyz", "abc", "d", "éa", "", nullptr, "f"};
  cudf::test::strings_column_wrapper strings2(
    h_strings2.begin(),
    h_strings2.end(),
    thrust::make_transform_iterator(h_strings2.begin(), [](auto str) { return str != nullptr; }));

  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(strings1);
  strings_columns.push_back(strings2);

  cudf::table_view table(strings_columns);

  {
    std::vector<const char*> h_expected{"eeexyz", "bbabc", nullptr, "éa", "aa", nullptr, "éééf"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results = cudf::strings::concatenate(table);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<const char*> h_expected{
      "eee:xyz", "bb:abc", nullptr, ":éa", "aa:", nullptr, "ééé:f"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results = cudf::strings::concatenate(table, cudf::string_scalar(":"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<const char*> h_expected{"eee:xyz", "bb:abc", "_:d", ":éa", "aa:", "bbb:_", "ééé:f"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results =
      cudf::strings::concatenate(table, cudf::string_scalar(":"), cudf::string_scalar("_"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<const char*> h_expected{"eeexyz", "bbabc", "d", "éa", "aa", "bbb", "éééf"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results =
      cudf::strings::concatenate(table, cudf::string_scalar(""), cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCombineTest, ConcatZeroSizeStringsColumns)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(zero_size_strings_column);
  strings_columns.push_back(zero_size_strings_column);
  cudf::table_view table(strings_columns);
  auto results = cudf::strings::concatenate(table);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsCombineTest, Join)
{
  std::vector<const char*> h_strings{"eee", "bb", nullptr, "zzzz", "", "aaa", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto view1 = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::join_strings(view1);

    cudf::test::strings_column_wrapper expected{"eeebbzzzzaaaééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::join_strings(view1, cudf::string_scalar("+"));

    cudf::test::strings_column_wrapper expected{"eee+bb+zzzz++aaa+ééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results =
      cudf::strings::join_strings(view1, cudf::string_scalar("+"), cudf::string_scalar("___"));

    cudf::test::strings_column_wrapper expected{"eee+bb+___+zzzz++aaa+ééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCombineTest, JoinZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::join_strings(strings_view);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsCombineTest, JoinAllNullStringsColumn)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {0, 0, 0});

  auto results = cudf::strings::join_strings(cudf::strings_column_view(strings));
  cudf::test::strings_column_wrapper expected1({""}, {0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected1);

  results = cudf::strings::join_strings(
    cudf::strings_column_view(strings), cudf::string_scalar(""), cudf::string_scalar("3"));
  cudf::test::strings_column_wrapper expected2({"333"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);

  results = cudf::strings::join_strings(
    cudf::strings_column_view(strings), cudf::string_scalar("-"), cudf::string_scalar("*"));
  cudf::test::strings_column_wrapper expected3({"*-*-*"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected3);
}

struct StringsConcatenateWithColSeparatorTest : public cudf::test::BaseFixture {
};

TEST_F(StringsConcatenateWithColSeparatorTest, ExceptionTests)
{
  // Exception tests
  // 0. 0 columns passed
  // 1. > 0 columns passed; some using non string data types
  // 2. separator column of different size to column size
  {
    EXPECT_THROW(cudf::strings::concatenate(cudf::table_view{},
                                            cudf::strings_column_view{cudf::column_view{}}),
                 cudf::logic_error);
  }

  {
    cudf::column_view col0(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
    cudf::test::fixed_width_column_wrapper<int64_t> col1{{1}};

    EXPECT_THROW(
      cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(col0)),
      cudf::logic_error);
  }

  {
    auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
    auto sep_col = cudf::test::strings_column_wrapper({"", ""}, {true, false});

    EXPECT_THROW(
      cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(sep_col)),
      cudf::logic_error);
  }
}

TEST_F(StringsConcatenateWithColSeparatorTest, ZeroSizedColumns)
{
  cudf::column_view col0(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(col0));
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnEmptyAndNullStringsNoReplacements)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});

  auto exp_results =
    cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, false});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnEmptyAndNullStringsSeparatorReplacement)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});
  auto sep_rep = cudf::string_scalar("");

  auto exp_results =
    cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnEmptyAndNullStringsColumnReplacement)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});
  auto col_rep = cudf::string_scalar("");

  auto exp_results =
    cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});

  auto results = cudf::strings::concatenate(cudf::table_view{{col0}},
                                            cudf::strings_column_view(sep_col),
                                            cudf::string_scalar("", false),
                                            col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest,
       SingleColumnEmptyAndNullStringsSeparatorAndColumnReplacement)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});
  auto sep_rep = cudf::string_scalar("");
  auto col_rep = cudf::string_scalar("");

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", ""});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep, col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixNoReplacements)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, true, true, true, true, true, false, true, true});

  auto exp_results = cudf::test::strings_column_wrapper(
    {"", "", "", "bbabc", "", "d", "éa", "", "bbb", "éééf"},
    {false, false, true, true, false, true, true, false, true, true});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixSeparatorReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, false, true, true, true, true, false, true, true});
  auto sep_rep = cudf::string_scalar("-");

  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyz", "", "", "bbabc", "", "d", "éa", "", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, false, true, true, true, true, false, true, true});
  auto col_rep = cudf::string_scalar("goobly");

  auto exp_results = cudf::test::strings_column_wrapper(
    {"", "", "", "bbabc", "goobly", "d", "éa", "", "bbb", "éééf"},
    {false, false, false, true, true, true, true, false, true, true});

  auto results = cudf::strings::concatenate(cudf::table_view{{col0}},
                                            cudf::strings_column_view(sep_col),
                                            cudf::string_scalar("", false),
                                            col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixSeparatorAndColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, false, true, true, true, true, false, true, true});
  auto sep_rep = cudf::string_scalar("-");
  auto col_rep = cudf::string_scalar("goobly");

  // All valid, as every invalid element is replaced - a non nullable column
  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyz", "goobly", "", "bbabc", "goobly", "d", "éa", "goobly", "bbb", "éééf"});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep, col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnEmptyAndNullStringsNoReplacements)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, false, true, true, true, true, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, false, true, true, false, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {true, false, true, false, true, false, true, false});

  auto exp_results = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, false, true, false, true, false, true, false});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixNoReplacements)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});

  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyzfoo", "<null>~~~", "", "éééf", "", "", "", "valid", "doo", "", "", ""},
    {true, true, false, true, false, true, false, true, true, false, false, false});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixSeparatorReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});
  auto sep_rep = cudf::string_scalar("!!!!!!!!!!");

  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyzfoo",
     "<null>~~~",
     "!!!!!!!!!!éaff",
     "éééf",
     "éa",
     "",
     "éaff",
     "valid",
     "doo",
     "",
     "",
     ""},
    {true, true, true, true, true, true, true, true, true, false, false, false});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col), sep_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});
  auto col_rep = cudf::string_scalar("_col_replacement_");

  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyzfoo",
     "<null>~~~",
     "",
     "éééf@_col_replacement_",
     "",
     "_col_replacement_",
     "",
     "_col_replacement_^^^^valid",
     "_col_replacement_doo",
     "_col_replacement_--_col_replacement_",
     "_col_replacement_*****_col_replacement_",
     "_col_replacement_######_col_replacement_"},
    {true, true, false, true, false, true, false, true, true, true, true, true});

  auto results = cudf::strings::concatenate(cudf::table_view{{col0, col1}},
                                            cudf::strings_column_view(sep_col),
                                            cudf::string_scalar("", false),
                                            col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixSeparatorAndColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});
  auto sep_rep = cudf::string_scalar("!!!!!!!!!!");
  auto col_rep = cudf::string_scalar("_col_replacement_");

  // Every null item (separator/column) is replaced - a non nullable column
  auto exp_results =
    cudf::test::strings_column_wrapper({"eeexyzfoo",
                                        "<null>~~~",
                                        "!!!!!!!!!!éaff",
                                        "éééf@_col_replacement_",
                                        "éa!!!!!!!!!!_col_replacement_",
                                        "_col_replacement_",
                                        "_col_replacement_!!!!!!!!!!éaff",
                                        "_col_replacement_^^^^valid",
                                        "_col_replacement_doo",
                                        "_col_replacement_--_col_replacement_",
                                        "_col_replacement_*****_col_replacement_",
                                        "_col_replacement_######_col_replacement_"});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col), sep_rep, col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnNonNullableStrings)
{
  auto col0 =
    cudf::test::strings_column_wrapper({"eeexyz", "<null>", "éaff", "éééf", "", "", "", ""});
  auto col1 = cudf::test::strings_column_wrapper({"foo", "nan", "", "", "NULL", "éaff", "", ""});
  auto sep_col = cudf::test::strings_column_wrapper({"", "~~~", "", "@", "", "+++", "", "^^^^"});

  // Every item (separator/column) is used, as everything is valid producing a non nullable column
  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyzfoo", "<null>~~~nan", "éaff", "éééf@", "NULL", "+++éaff", "", "^^^^"});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

struct StringsListsConcatenateTest : public cudf::test::BaseFixture {
};

namespace {
using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;
using STR_COL   = cudf::test::strings_column_wrapper;
using INT_LISTS = cudf::test::lists_column_wrapper<int32_t>;

constexpr bool print_all{true};

auto null_at(cudf::size_type idx)
{
  return cudf::detail::make_counting_transform_iterator(0, [idx](auto i) { return i != idx; });
}

auto all_nulls()
{
  return cudf::detail::make_counting_transform_iterator(0, [](auto) { return false; });
}

auto nulls_from_nullptr(std::vector<const char*> const& strs)
{
  return thrust::make_transform_iterator(strs.begin(), [](auto ptr) { return ptr != nullptr; });
}

}  // namespace

TEST_F(StringsListsConcatenateTest, InvalidInput)
{
  // Invalid list type
  {
    auto const l = INT_LISTS{{1, 2, 3}, {4, 5, 6}}.release();
    EXPECT_THROW(cudf::strings::concatenate(cudf::lists_column_view(l->view())), cudf::logic_error);
  }

  // Invalid separator
  {
    auto const l  = STR_LISTS{STR_LISTS{""}, STR_LISTS{"", "", ""}, STR_LISTS{"", ""}}.release();
    auto const lv = cudf::lists_column_view(l->view());
    EXPECT_THROW(cudf::strings::concatenate(cudf::lists_column_view(l->view()),
                                            cudf::string_scalar("", false)),
                 cudf::logic_error);
  }
}

TEST_F(StringsListsConcatenateTest, EmptyInput)
{
  auto const l        = STR_LISTS{}.release();
  auto const lv       = cudf::lists_column_view(l->view());
  auto const results  = cudf::strings::concatenate(lv);
  auto const expected = STR_COL{};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
}

TEST_F(StringsListsConcatenateTest, ZeroSizeStringsInput)
{
  auto const l       = STR_LISTS{STR_LISTS{""}, STR_LISTS{"", "", ""}, STR_LISTS{"", ""}}.release();
  auto const lv      = cudf::lists_column_view(l->view());
  auto const results = cudf::strings::concatenate(lv);
  auto const expected = STR_COL{"", "", ""};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
}

TEST_F(StringsListsConcatenateTest, AllNullsStringsInput)
{
  auto const l = STR_LISTS{STR_LISTS{{""}, all_nulls()},
                           STR_LISTS{{"", "", ""}, all_nulls()},
                           STR_LISTS{{"", ""}, all_nulls()}}
                   .release();
  auto const lv       = cudf::lists_column_view(l->view());
  auto const results  = cudf::strings::concatenate(lv);
  auto const expected = STR_COL{{"", "", ""}, all_nulls()};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
}

TEST_F(StringsListsConcatenateTest, ScalarSeparator)
{
  auto const l = STR_LISTS{{STR_LISTS{{"a", "bb" /*NULL*/, "ccc"}, null_at(1)},
                            STR_LISTS{}, /*NULL*/
                            STR_LISTS{{"ddd" /*NULL*/, "efgh", "ijk"}, null_at(0)},
                            STR_LISTS{"zzz", "xxxxx"}},
                           null_at(1)}
                   .release();
  auto const lv = cudf::lists_column_view(l->view());

  // No null replacement
  {
    auto const results = cudf::strings::concatenate(lv, cudf::string_scalar("+++"));
    std::vector<const char*> h_expected{nullptr, nullptr, nullptr, "zzz+++xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // With null replacement
  {
    auto const results =
      cudf::strings::concatenate(lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<const char*> h_expected{
      "a+++___+++ccc", nullptr, "___+++efgh+++ijk", "zzz+++xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }
}

TEST_F(StringsListsConcatenateTest, SlicedListsWithScalarSeparator)
{
  auto const l = STR_LISTS{
    {STR_LISTS{{"a", "bb" /*NULL*/, "ccc"}, null_at(1)},
     STR_LISTS{}, /*NULL*/
     STR_LISTS{{"ddd" /*NULL*/, "efgh", "ijk"}, null_at(0)},
     STR_LISTS{"zzz", "xxxxx"},
     STR_LISTS{"11111", "11111", "11111", "11111", "11111"}, /*NULL*/
     STR_LISTS{{"abcdef", "012345", "" /*NULL*/, "xxx000"}, null_at(2)},
     STR_LISTS{{"xyz" /*NULL*/, "11111", "00000"}, null_at(0)},
     STR_LISTS{"0a0b0c", "5x5y5z"},
     STR_LISTS{"xxx"}, /*NULL*/
     STR_LISTS{"ééé", "12345abcdef"},
     STR_LISTS{"aaaééébbbéééccc", "12345"}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 1 && i != 4 && i != 8;
    })}.release();

  // Sliced the entire lists column, no null replacement
  {
    auto const lv      = cudf::lists_column_view(cudf::slice(l->view(), {0, 11})[0]);
    auto const results = cudf::strings::concatenate(lv, cudf::string_scalar("+++"));
    std::vector<const char*> h_expected{nullptr,
                                        nullptr,
                                        nullptr,
                                        "zzz+++xxxxx",
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        "0a0b0c+++5x5y5z",
                                        nullptr,
                                        "ééé+++12345abcdef",
                                        "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // Sliced the entire lists column, with null replacement
  {
    auto const lv = cudf::lists_column_view(cudf::slice(l->view(), {0, 11})[0]);
    auto const results =
      cudf::strings::concatenate(lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<const char*> h_expected{"a+++___+++ccc",
                                        nullptr,
                                        "___+++efgh+++ijk",
                                        "zzz+++xxxxx",
                                        nullptr,
                                        "abcdef+++012345+++___+++xxx000",
                                        "___+++11111+++00000",
                                        "0a0b0c+++5x5y5z",
                                        nullptr,
                                        "ééé+++12345abcdef",
                                        "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // Sliced the first half of the lists column, no null replacement
  {
    auto const lv      = cudf::lists_column_view(cudf::slice(l->view(), {0, 5})[0]);
    auto const results = cudf::strings::concatenate(lv, cudf::string_scalar("+++"));
    std::vector<const char*> h_expected{nullptr, nullptr, nullptr, "zzz+++xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // Sliced the first half of the lists column, with null replacement
  {
    auto const lv = cudf::lists_column_view(cudf::slice(l->view(), {0, 11})[0]);
    auto const results =
      cudf::strings::concatenate(lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<const char*> h_expected{
      "a+++___+++ccc", nullptr, "___+++efgh+++ijk", "zzz+++xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // Sliced the second half of the lists column, no null replacement
  {
    auto const lv      = cudf::lists_column_view(cudf::slice(l->view(), {5, 11})[0]);
    auto const results = cudf::strings::concatenate(lv, cudf::string_scalar("+++"));
    std::vector<const char*> h_expected{
      nullptr, nullptr, "0a0b0c+++5x5y5z", nullptr, "ééé+++12345abcdef", "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // Sliced the second half of the lists column, with null replacement
  {
    auto const lv = cudf::lists_column_view(cudf::slice(l->view(), {5, 11})[0]);
    auto const results =
      cudf::strings::concatenate(lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<const char*> h_expected{"abcdef+++012345+++___+++xxx000",
                                        "___+++11111+++00000",
                                        "0a0b0c+++5x5y5z",
                                        nullptr,
                                        "ééé+++12345abcdef",
                                        "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // Sliced the middle part of the lists column, no null replacement
  {
    auto const lv      = cudf::lists_column_view(cudf::slice(l->view(), {3, 8})[0]);
    auto const results = cudf::strings::concatenate(lv, cudf::string_scalar("+++"));
    std::vector<const char*> h_expected{
      "zzz+++xxxxx", nullptr, nullptr, nullptr, "0a0b0c+++5x5y5z"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }

  // Sliced the middle part of the lists column, with null replacement
  {
    auto const lv = cudf::lists_column_view(cudf::slice(l->view(), {3, 8})[0]);
    auto const results =
      cudf::strings::concatenate(lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<const char*> h_expected{"zzz+++xxxxx",
                                        nullptr,
                                        "abcdef+++012345+++___+++xxx000",
                                        "___+++11111+++00000",
                                        "0a0b0c+++5x5y5z"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptr(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
  }
}

TEST_F(StringsListsConcatenateTest, ColumnSeparatorNoReplacements)
{
  auto const l = STR_LISTS{STR_LISTS{{""}, all_nulls()},
                           STR_LISTS{{"", "", ""}, all_nulls()},
                           STR_LISTS{{"", ""}, all_nulls()}}
                   .release();
  auto const lv       = cudf::lists_column_view(l->view());
  auto const results  = cudf::strings::concatenate(lv);
  auto const expected = STR_COL{{"", "", ""}, all_nulls()};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
}

TEST_F(StringsListsConcatenateTest, ColumnSeparatorWithReplacements)
{
  auto const l = STR_LISTS{STR_LISTS{{""}, all_nulls()},
                           STR_LISTS{{"", "", ""}, all_nulls()},
                           STR_LISTS{{"", ""}, all_nulls()}}
                   .release();
  auto const lv       = cudf::lists_column_view(l->view());
  auto const results  = cudf::strings::concatenate(lv);
  auto const expected = STR_COL{{"", "", ""}, all_nulls()};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
}

TEST_F(StringsListsConcatenateTest, SlicedListsWithColumnSeparator)
{
  auto const l = STR_LISTS{STR_LISTS{{""}, all_nulls()},
                           STR_LISTS{{"", "", ""}, all_nulls()},
                           STR_LISTS{{"", ""}, all_nulls()}}
                   .release();
  auto const lv       = cudf::lists_column_view(l->view());
  auto const results  = cudf::strings::concatenate(lv);
  auto const expected = STR_COL{{"", "", ""}, all_nulls()};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, print_all);
}
