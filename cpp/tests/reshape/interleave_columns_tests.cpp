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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/reshape.hpp>

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

namespace {
using StrListsCol = cudf::test::lists_column_wrapper<cudf::string_view>;
using IntListsCol = cudf::test::lists_column_wrapper<int32_t>;
using IntCol      = cudf::test::fixed_width_column_wrapper<int32_t>;
using TView       = cudf::table_view;

constexpr bool print_all{true};  // For debugging
constexpr int32_t null{0};

auto null_at(cudf::size_type idx)
{
  return cudf::detail::make_counting_transform_iterator(0, [idx](auto i) { return i != idx; });
}
auto null_at(std::vector<cudf::size_type> const& indices)
{
  return cudf::detail::make_counting_transform_iterator(0, [&indices](auto i) {
    return std::find(indices.cbegin(), indices.cend(), i) == indices.cend();
  });
}

auto all_nulls()
{
  return cudf::detail::make_counting_transform_iterator(0, [](auto) { return false; });
}

}  // namespace

struct ListsColumnsInterleaveTest : public cudf::test::BaseFixture {
};

TEST_F(ListsColumnsInterleaveTest, InvalidInput)
{
  // Input table contains non-list column
  {
    auto const col1 = IntCol{}.release();
    auto const col2 = IntListsCol{}.release();
    EXPECT_THROW(cudf::interleave_columns(TView{{col1->view(), col2->view()}}), cudf::logic_error);
  }

  // Types mismatch
  {
    auto const col1 = IntListsCol{}.release();
    auto const col2 = StrListsCol{}.release();
    EXPECT_THROW(cudf::interleave_columns(TView{{col1->view(), col2->view()}}), cudf::logic_error);
  }

  // Nested types are not supported
  {
    auto const col = IntListsCol{{IntListsCol{1, 2, 3}, IntListsCol{4, 5, 6}}}.release();
    EXPECT_THROW(cudf::interleave_columns(TView{{col->view(), col->view()}}), cudf::logic_error);
  }
}

template <typename T>
struct ListsColumnsInterleaveTypedTest : public cudf::test::BaseFixture {
};
#define ListsCol cudf::test::lists_column_wrapper<TypeParam>

using TypesForTest =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_CASE(ListsColumnsInterleaveTypedTest, TypesForTest);

TYPED_TEST(ListsColumnsInterleaveTypedTest, InterleaveEmptyColumns)
{
  auto const col     = ListsCol{}.release();
  auto const results = cudf::interleave_columns(TView{{col->view(), col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, print_all);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InterleaveOneColumnNotNull)
{
  auto const col     = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const results = cudf::interleave_columns(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, print_all);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InterleaveOneColumnWithNulls)
{
  auto const col = ListsCol{{ListsCol{{1, 2, null}, null_at(2)},
                             ListsCol{} /*NULL*/,
                             ListsCol{{null, 3, 4, 4, 4, 4}, null_at(0)},
                             ListsCol{5, 6}},
                            null_at(1)}
                     .release();
  auto const results = cudf::interleave_columns(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, print_all);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SimpleInputNoNull)
{
  auto const col1     = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const col2     = ListsCol{{7, 8}, {9, 10}, {11, 12}}.release();
  auto const expected = ListsCol{{1, 2}, {7, 8}, {3, 4}, {9, 10}, {5, 6}, {11, 12}}.release();
  auto const results  = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TEST_F(ListsColumnsInterleaveTest, SimpleInputStringsColumnsNoNull)
{
  auto const col1 = StrListsCol{
    StrListsCol{"Tomato", "Apple"},
    StrListsCol{"Banana", "Kiwi", "Cherry"},
    StrListsCol{
      "Coconut"}}.release();
  auto const col2 =
    StrListsCol{StrListsCol{"Orange"}, StrListsCol{"Lemon", "Peach"}, StrListsCol{}}.release();
  auto const expected = StrListsCol{
    StrListsCol{"Tomato", "Apple"},
    StrListsCol{"Orange"},
    StrListsCol{"Banana", "Kiwi", "Cherry"},
    StrListsCol{"Lemon", "Peach"},
    StrListsCol{"Coconut"},
    StrListsCol{}}.release();
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SimpleInputWithNulls)
{
  auto const col1 = ListsCol{{ListsCol{{1, null, 3, 4}, null_at(1)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{1, 2, null, 4}, null_at(2)},
                              ListsCol{{1, 2, 3, null}, null_at(3)}},
                             null_at(3)}
                      .release();
  auto const col2 = ListsCol{{ListsCol{{10, 11, 12, null}, null_at(3)},
                              ListsCol{{13, 14, 15, 16, 17, null}, null_at(5)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{null, 18}, null_at(0)},
                              ListsCol{{19, 20, null}, null_at(2)},
                              ListsCol{{null}, null_at(0)}},
                             null_at(2)}
                      .release();
  auto const col3 = ListsCol{{ListsCol{} /*NULL*/,
                              ListsCol{{20, null}, null_at(1)},
                              ListsCol{{null, 21, null, null}, null_at({0, 2, 3})},
                              ListsCol{},
                              ListsCol{22, 23, 24, 25},
                              ListsCol{{null, null, null, null, null}, all_nulls()}},
                             null_at(0)}
                      .release();

  auto const results  = cudf::interleave_columns(TView{{col1->view(), col2->view(), col3->view()}});
  auto const expected = ListsCol{{ListsCol{{1, null, 3, 4}, null_at(1)},
                                  ListsCol{{10, 11, 12, null}, null_at(3)},
                                  ListsCol{} /*NULL*/,
                                  ListsCol{{null, 2, 3, 4}, null_at(0)},
                                  ListsCol{{13, 14, 15, 16, 17, null}, null_at(5)},
                                  ListsCol{{20, null}, null_at(1)},
                                  ListsCol{{null, 2, 3, 4}, null_at(0)},
                                  ListsCol{} /*NULL*/,
                                  ListsCol{{null, 21, null, null}, null_at({0, 2, 3})},
                                  ListsCol{} /*NULL*/,
                                  ListsCol{{null, 18}, null_at(0)},
                                  ListsCol{},
                                  ListsCol{{1, 2, null, 4}, null_at(2)},
                                  ListsCol{{19, 20, null}, null_at(2)},
                                  ListsCol{22, 23, 24, 25},
                                  ListsCol{{1, 2, 3, null}, null_at(3)},
                                  ListsCol{{null}, null_at(0)},
                                  ListsCol{{null, null, null, null, null}, all_nulls()}},
                                 null_at({2, 7, 9})}
                          .release();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TEST_F(ListsColumnsInterleaveTest, SimpleInputStringsColumnsWithNulls)
{
  auto const col1 = StrListsCol{
    StrListsCol{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
    StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, null_at({1, 4})},
    StrListsCol{
      "Coconut"}}.release();
  auto const col2 =
    StrListsCol{
      {StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, null_at({1, 2, 3})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{{"Deer" /*NULL*/, "Snake" /*NULL*/, "Horse" /*NULL*/}, all_nulls()}}, /*NULL*/
      null_at(2)}
      .release();

  // Ignore null list elements
  {
    auto const results  = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
    auto const expected = StrListsCol{
      StrListsCol{{"Tomato",
                   "Bear" /*NULL*/,
                   "Apple",
                   "Orange",
                   "Dog" /*NULL*/,
                   "Fox" /*NULL*/,
                   "Duck" /*NULL*/},
                  null_at({1, 4, 5, 6})},
      StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/, "Lemon", "Peach"},
                  null_at({1, 4})},
      StrListsCol{
        "Coconut"}}.release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Null list rows result in null list rows
  {
    auto const results =
      cudf::interleave_columns(TView{{col1->view(), col2->view()}},
                               cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      StrListsCol{
        {StrListsCol{{"Tomato",
                      "Bear" /*NULL*/,
                      "Apple",
                      "Orange",
                      "Dog" /*NULL*/,
                      "Fox" /*NULL*/,
                      "Duck" /*NULL*/},
                     null_at({1, 4, 5, 6})},
         StrListsCol{
           {"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/, "Lemon", "Peach"},
           null_at({1, 4})},
         StrListsCol{""} /*NULL*/},
        null_at(2)}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedColumnsInputNoNull)
{
  auto const col_original = ListsCol{{1, 2, 3}, {2, 3}, {3, 4, 5, 6}, {5, 6}, {}, {7}}.release();

  // Slice first half
  {
    auto const col = cudf::slice(col_original->view(), {0, 3})[0];
    auto const expected =
      ListsCol{
        {1, 2, 3, 1, 2, 3, 1, 2, 3}, {2, 3, 2, 3, 2, 3}, {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6}}
        .release();
    auto const results = cudf::interleave_columns(TView{{col, col, col}});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Slice second half
  {
    auto const col      = cudf::slice(col_original->view(), {3, 6})[0];
    auto const expected = ListsCol{{5, 6, 5, 6, 5, 6}, {}, {7, 7, 7}}.release();
    auto const results  = cudf::interleave_columns(TView{{col, col, col}});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Slice middle segment
  {
    auto const col = cudf::slice(col_original->view(), {2, 4})[0];
    auto const expected =
      ListsCol{{3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6}, {5, 6, 5, 6, 5, 6}}.release();
    auto const results = cudf::interleave_columns(TView{{col, col, col}});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedColumnsInputWithNulls)
{
  auto const col_original = ListsCol{{ListsCol{{null, 2, 3}, null_at(0)},
                                      ListsCol{{2, 3}}, /*NULL*/
                                      ListsCol{{3, null, 5, 6}, null_at(1)},
                                      ListsCol{{5, 6}}, /*NULL*/
                                      ListsCol{},       /*NULL*/
                                      ListsCol{7},
                                      ListsCol{8, 9, 10}},
                                     null_at({1, 3, 4})}
                              .release();

  // Slice first half
  {
    auto const col      = cudf::slice(col_original->view(), {0, 3})[0];
    auto const expected = ListsCol{{ListsCol{{null, 2, 3, null, 2, 3}, null_at({0, 3})},
                                    ListsCol{}, /*NULL*/
                                    ListsCol{{3, null, 5, 6, 3, null, 5, 6}, null_at({1, 5})}},
                                   null_at(1)}
                            .release();
    auto const results = cudf::interleave_columns(TView{{col, col}});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Slice second half
  {
    auto const col      = cudf::slice(col_original->view(), {3, 7})[0];
    auto const expected = ListsCol{{ListsCol{}, /*NULL*/
                                    ListsCol{}, /*NULL*/
                                    ListsCol{7, 7},
                                    ListsCol{8, 9, 10, 8, 9, 10}},
                                   null_at({0, 1})}
                            .release();
    auto const results = cudf::interleave_columns(TView{{col, col}});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Slice middle segment
  {
    auto const col      = cudf::slice(col_original->view(), {2, 6})[0];
    auto const expected = ListsCol{{ListsCol{{3, null, 5, 6, 3, null, 5, 6}, null_at({1, 5})},
                                    ListsCol{}, /*NULL*/
                                    ListsCol{}, /*NULL*/
                                    ListsCol{7, 7}},
                                   null_at({1, 2})}
                            .release();
    auto const results = cudf::interleave_columns(TView{{col, col}});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }
}

CUDF_TEST_PROGRAM_MAIN()
