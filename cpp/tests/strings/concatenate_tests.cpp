/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>

#include <vector>

struct StringsConcatenateTest : public cudf::test::BaseFixture {};

TEST_F(StringsConcatenateTest, Concatenate)
{
  std::vector<char const*> h_strings{"aaa",
                                     "bb",
                                     "",
                                     "cccc",
                                     "d",
                                     "ééé",
                                     "ff",
                                     "gggg",
                                     "",
                                     "h",
                                     "iiii",
                                     "jjj",
                                     "k",
                                     "lllllll",
                                     "mmmmm",
                                     "n",
                                     "oo",
                                     "ppp"};
  cudf::test::strings_column_wrapper strings1(h_strings.data(), h_strings.data() + 6);
  cudf::test::strings_column_wrapper strings2(h_strings.data() + 6, h_strings.data() + 10);
  cudf::test::strings_column_wrapper strings3(h_strings.data() + 10,
                                              h_strings.data() + h_strings.size());

  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(strings1);
  strings_columns.push_back(zero_size_strings_column);
  strings_columns.push_back(strings2);
  strings_columns.push_back(strings3);

  auto results = cudf::concatenate(strings_columns);

  cudf::test::strings_column_wrapper expected(h_strings.begin(), h_strings.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConcatenateTest, ZeroSizeStringsColumns)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(zero_size_strings_column);
  strings_columns.push_back(zero_size_strings_column);
  strings_columns.push_back(zero_size_strings_column);
  auto results = cudf::concatenate(strings_columns);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsConcatenateTest, ZeroSizeStringsPlusNormal)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(zero_size_strings_column);

  std::vector<char const*> h_strings{"aaa",
                                     "bb",
                                     "",
                                     "cccc",
                                     "d",
                                     "ééé",
                                     "ff",
                                     "gggg",
                                     "",
                                     "h",
                                     "iiii",
                                     "jjj",
                                     "k",
                                     "lllllll",
                                     "mmmmm",
                                     "n",
                                     "oo",
                                     "ppp"};
  cudf::test::strings_column_wrapper strings1(h_strings.data(),
                                              h_strings.data() + h_strings.size());
  strings_columns.push_back(strings1);

  auto results = cudf::concatenate(strings_columns);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings1);
}
