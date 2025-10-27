/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

#include <vector>

struct StringsFillTest : public cudf::test::BaseFixture {};

TEST_F(StringsFillTest, Fill)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper input(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));

  {
    auto results = cudf::fill(input, 1, 5, cudf::string_scalar("zz"));

    std::vector<char const*> h_expected{"eee", "zz", "zz", "zz", "zz", "bbb", "ééé"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::fill(input, 2, 4, cudf::string_scalar("", false));

    std::vector<char const*> h_expected{"eee", "bb", nullptr, nullptr, "aa", "bbb", "ééé"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::fill(input, 5, 5, cudf::string_scalar("zz"));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, input);
  }
  {
    auto results = cudf::fill(input, 0, 7, cudf::string_scalar(""));
    cudf::test::strings_column_wrapper expected({"", "", "", "", "", "", ""},
                                                {true, true, true, true, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::fill(input, 0, 7, cudf::string_scalar("", false));
    cudf::test::strings_column_wrapper expected({"", "", "", "", "", "", ""},
                                                {false, false, false, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TEST_F(StringsFillTest, ZeroSizeStringsColumns)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto results = cudf::fill(zero_size_strings_column, 0, 0, cudf::string_scalar(""));
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsFillTest, FillRangeError)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper input(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));

  EXPECT_THROW(cudf::fill(input, 5, 1, cudf::string_scalar("")), cudf::logic_error);
  EXPECT_THROW(cudf::fill(input, 5, 9, cudf::string_scalar("")), cudf::logic_error);
}
