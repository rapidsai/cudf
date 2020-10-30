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

#include <tests/strings/utilities.h>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reshape.hpp>

#include <type_traits>
#include "cudf/utilities/traits.hpp"

using namespace cudf::test;

template <typename T>
struct InterleaveColumnsTest : public BaseFixture {
};

TYPED_TEST_CASE(InterleaveColumnsTest, cudf::test::FixedWidthTypes);

TYPED_TEST(InterleaveColumnsTest, NoColumns)
{
  cudf::table_view in(std::vector<cudf::column_view>{});

  EXPECT_THROW(cudf::interleave_columns(in), cudf::logic_error);
}

TYPED_TEST(InterleaveColumnsTest, OneColumn)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T, int32_t> a({-1, 0, 1});

  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto expected = fixed_width_column_wrapper<T, int32_t>({-1, 0, 1});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, TwoColumns)
{
  using T = TypeParam;

  auto a = fixed_width_column_wrapper<T, int32_t>({0, 2});
  auto b = fixed_width_column_wrapper<T, int32_t>({1, 3});

  cudf::table_view in(std::vector<cudf::column_view>{a, b});

  auto expected = fixed_width_column_wrapper<T, int32_t>({0, 1, 2, 3});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, ThreeColumns)
{
  using T = TypeParam;

  auto a = fixed_width_column_wrapper<T, int32_t>({0, 3, 6});
  auto b = fixed_width_column_wrapper<T, int32_t>({1, 4, 7});
  auto c = fixed_width_column_wrapper<T, int32_t>({2, 5, 8});

  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto expected = fixed_width_column_wrapper<T, int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, OneColumnEmpty)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> a({});

  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto expected = fixed_width_column_wrapper<T>({});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, ThreeColumnsEmpty)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> a({});
  fixed_width_column_wrapper<T> b({});
  fixed_width_column_wrapper<T> c({});

  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto expected = fixed_width_column_wrapper<T>({});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, OneColumnNullable)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T, int32_t> a({1, 2, 3}, {0, 1, 0});

  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto expected = fixed_width_column_wrapper<T, int32_t>({0, 2, 0}, {0, 1, 0});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, TwoColumnNullable)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T, int32_t> a({1, 2, 3}, {0, 1, 0});
  fixed_width_column_wrapper<T, int32_t> b({4, 5, 6}, {1, 0, 1});

  cudf::table_view in(std::vector<cudf::column_view>{a, b});

  auto expected = fixed_width_column_wrapper<T, int32_t>({0, 4, 2, 0, 0, 6}, {0, 1, 1, 0, 0, 1});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, ThreeColumnsNullable)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T, int32_t> a({1, 4, 7}, {1, 0, 1});
  fixed_width_column_wrapper<T, int32_t> b({2, 5, 8}, {0, 1, 0});
  fixed_width_column_wrapper<T, int32_t> c({3, 6, 9}, {1, 0, 1});

  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto expected = fixed_width_column_wrapper<T, int32_t>({1, 0, 3, 0, 5, 0, 7, 0, 9},
                                                         {1, 0, 1, 0, 1, 0, 1, 0, 1});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, MismatchedDtypes)
{
  using T = TypeParam;

  if (not std::is_same<int, T>::value and not cudf::is_fixed_point<T>()) {
    fixed_width_column_wrapper<int32_t> input_a({1, 4, 7}, {1, 0, 1});
    fixed_width_column_wrapper<T, int32_t> input_b({2, 5, 8}, {0, 1, 0});

    cudf::table_view input(std::vector<cudf::column_view>{input_a, input_b});

    EXPECT_THROW(cudf::interleave_columns(input), cudf::logic_error);
  }
}

struct InterleaveStringsColumnsTest : public BaseFixture {
};

TEST_F(InterleaveStringsColumnsTest, ZeroSizedColumns)
{
  cudf::column_view col0(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);

  auto results = cudf::interleave_columns(cudf::table_view{{col0}});
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(InterleaveStringsColumnsTest, SingleColumn)
{
  auto col0 = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});

  auto results = cudf::interleave_columns(cudf::table_view{{col0}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, col0, true);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnNullAndEmpty)
{
  auto col0 = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto col1 = cudf::test::strings_column_wrapper({"", "", "", ""}, {true, false, true, false});

  auto exp_results = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, true, true, false, true, true, false, false});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnEmptyNonNullable)
{
  auto col0 = cudf::test::strings_column_wrapper({"", "", "", ""});
  auto col1 = cudf::test::strings_column_wrapper({"", "", "", ""});

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", "", "", ""});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnStringMix)
{
  auto col0 = cudf::test::strings_column_wrapper({"null", "null", "", "valid", "", "valid"},
                                                 {false, false, true, true, true, true});
  auto col1 = cudf::test::strings_column_wrapper({"", "valid", "null", "null", "valid", ""},
                                                 {true, true, false, false, true, true});
  auto col2 = cudf::test::strings_column_wrapper({"valid", "", "valid", "", "null", "null"},
                                                 {true, true, true, true, false, false});

  auto exp_results = cudf::test::strings_column_wrapper({"null",
                                                         "",
                                                         "valid",
                                                         "null",
                                                         "valid",
                                                         "",
                                                         "",
                                                         "null",
                                                         "valid",
                                                         "valid",
                                                         "null",
                                                         "",
                                                         "",
                                                         "valid",
                                                         "null",
                                                         "valid",
                                                         "",
                                                         "null"},
                                                        {false,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         false});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnStringMixNonNullable)
{
  auto col0 = cudf::test::strings_column_wrapper({"c00", "c01", "", "valid", "", "valid"});
  auto col1 = cudf::test::strings_column_wrapper({"", "valid", "c13", "c14", "valid", ""});
  auto col2 = cudf::test::strings_column_wrapper({"valid", "", "valid", "", "c24", "c25"});

  auto exp_results = cudf::test::strings_column_wrapper({"c00",
                                                         "",
                                                         "valid",
                                                         "c01",
                                                         "valid",
                                                         "",
                                                         "",
                                                         "c13",
                                                         "valid",
                                                         "valid",
                                                         "c14",
                                                         "",
                                                         "",
                                                         "valid",
                                                         "c24",
                                                         "valid",
                                                         "",
                                                         "c25"});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnStringMixNullableMix)
{
  auto col0 = cudf::test::strings_column_wrapper({"c00", "c01", "", "valid", "", "valid"});
  auto col1 = cudf::test::strings_column_wrapper({"", "valid", "null", "null", "valid", ""},
                                                 {true, true, false, false, true, true});
  auto col2 = cudf::test::strings_column_wrapper({"valid", "", "valid", "", "c24", "c25"});

  auto exp_results = cudf::test::strings_column_wrapper({"c00",
                                                         "",
                                                         "valid",
                                                         "c01",
                                                         "valid",
                                                         "",
                                                         "",
                                                         "null",
                                                         "valid",
                                                         "valid",
                                                         "null",
                                                         "",
                                                         "",
                                                         "valid",
                                                         "c24",
                                                         "valid",
                                                         "",
                                                         "c25"},
                                                        {true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, true);
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, FixedPointInterleave)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  for (int i = 0; i > -4; --i) {
    auto const ONE  = decimalXX{1, scale_type{i}};
    auto const TWO  = decimalXX{2, scale_type{i}};
    auto const FOUR = decimalXX{4, scale_type{i}};
    auto const FIVE = decimalXX{5, scale_type{i}};

    auto const a = cudf::test::fixed_width_column_wrapper<decimalXX>({ONE, FOUR});
    auto const b = cudf::test::fixed_width_column_wrapper<decimalXX>({TWO, FIVE});

    auto const input    = cudf::table_view{std::vector<cudf::column_view>{a, b}};
    auto const expected = cudf::test::fixed_width_column_wrapper<decimalXX>({ONE, TWO, FOUR, FIVE});
    auto const actual   = cudf::interleave_columns(input);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
  }
}

CUDF_TEST_PROGRAM_MAIN()
