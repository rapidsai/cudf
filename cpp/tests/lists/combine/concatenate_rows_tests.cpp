/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/combine.hpp>

using namespace cudf::test::iterators;

namespace {
using StrListsCol = cudf::test::lists_column_wrapper<cudf::string_view>;
using IntListsCol = cudf::test::lists_column_wrapper<int32_t>;
using IntCol      = cudf::test::fixed_width_column_wrapper<int32_t>;
using TView       = cudf::table_view;

constexpr bool print_all{false};  // For debugging
constexpr int32_t null{0};
}  // namespace

struct ListConcatenateRowsTest : public cudf::test::BaseFixture {
};

TEST_F(ListConcatenateRowsTest, InvalidInput)
{
  // Empty input table
  EXPECT_THROW(cudf::lists::concatenate_rows(TView{}), cudf::logic_error);

  // Input table contains non-list column
  {
    auto const col1 = IntCol{}.release();
    auto const col2 = IntListsCol{}.release();
    EXPECT_THROW(cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}}),
                 cudf::logic_error);
  }

  // Types mismatch
  {
    auto const col1 = IntListsCol{}.release();
    auto const col2 = StrListsCol{}.release();
    EXPECT_THROW(cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}}),
                 cudf::logic_error);
  }

  // Nested types are not supported
  {
    auto const col = IntListsCol{{IntListsCol{1, 2, 3}, IntListsCol{4, 5, 6}}}.release();
    EXPECT_THROW(cudf::lists::concatenate_rows(TView{{col->view(), col->view()}}),
                 cudf::logic_error);
  }
}

template <typename T>
struct ListConcatenateRowsTypedTest : public cudf::test::BaseFixture {
};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::FixedPointTypes>;
TYPED_TEST_CASE(ListConcatenateRowsTypedTest, TypesForTest);

TYPED_TEST(ListConcatenateRowsTypedTest, ConcatenateEmptyColumns)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col     = ListsCol{}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col->view(), col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, print_all);
}

TYPED_TEST(ListConcatenateRowsTypedTest, ConcatenateOneColumnNotNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col     = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, print_all);
}

TYPED_TEST(ListConcatenateRowsTypedTest, ConcatenateOneColumnWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col = ListsCol{{ListsCol{{1, 2, null}, null_at(2)},
                             ListsCol{} /*NULL*/,
                             ListsCol{{null, 3, 4, 4, 4, 4}, null_at(0)},
                             ListsCol{5, 6}},
                            null_at(1)}
                     .release();
  auto const results = cudf::lists::concatenate_rows(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, print_all);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SimpleInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1     = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const col2     = ListsCol{{7, 8}, {9, 10}, {11, 12}}.release();
  auto const expected = ListsCol{{1, 2, 7, 8}, {3, 4, 9, 10}, {5, 6, 11, 12}}.release();
  auto const results  = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SimpleInputWithNullableChild)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1 = ListsCol{{1, 2}, ListsCol{{null}, null_at(0)}, {5, 6}}.release();
  auto const col2 = ListsCol{{7, 8}, {9, 10}, {11, 12}}.release();
  auto const expected =
    ListsCol{{1, 2, 7, 8}, ListsCol{{null, 9, 10}, null_at(0)}, {5, 6, 11, 12}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsNoNull)
{
  auto const col1 = StrListsCol{
    StrListsCol{"Tomato", "Apple"},
    StrListsCol{"Banana", "Kiwi", "Cherry"},
    StrListsCol{
      "Coconut"}}.release();
  auto const col2 =
    StrListsCol{StrListsCol{"Orange"}, StrListsCol{"Lemon", "Peach"}, StrListsCol{}}.release();
  auto const expected = StrListsCol{
    StrListsCol{"Tomato", "Apple", "Orange"},
    StrListsCol{"Banana", "Kiwi", "Cherry", "Lemon", "Peach"},
    StrListsCol{
      "Coconut"}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsWithNullableChild)
{
  auto const col1 = StrListsCol{
    StrListsCol{"Tomato", "Apple"},
    StrListsCol{"Banana", "Kiwi", "Cherry"},
    StrListsCol{
      "Coconut"}}.release();
  auto const col2 = StrListsCol{
    StrListsCol{"Orange"},
    StrListsCol{{"Lemon", "Peach"}, null_at(1)},
    StrListsCol{}}.release();
  auto const expected = StrListsCol{
    StrListsCol{"Tomato", "Apple", "Orange"},
    StrListsCol{{"Banana", "Kiwi", "Cherry", "Lemon", "Peach"}, null_at(4)},
    StrListsCol{
      "Coconut"}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SimpleInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1 = ListsCol{{ListsCol{{1, null, 3, 4}, null_at(1)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{1, 2, null, 4}, null_at(2)},
                              ListsCol{{1, 2, 3, null}, null_at(3)},
                              ListsCol{} /*NULL*/},
                             nulls_at({3, 6})}
                      .release();
  auto const col2 = ListsCol{{ListsCol{{10, 11, 12, null}, null_at(3)},
                              ListsCol{{13, 14, 15, 16, 17, null}, null_at(5)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{null, 18}, null_at(0)},
                              ListsCol{{19, 20, null}, null_at(2)},
                              ListsCol{{null}, null_at(0)},
                              ListsCol{} /*NULL*/},
                             nulls_at({2, 6})}
                      .release();
  auto const col3 = ListsCol{{ListsCol{} /*NULL*/,
                              ListsCol{{20, null}, null_at(1)},
                              ListsCol{{null, 21, null, null}, nulls_at({0, 2, 3})},
                              ListsCol{},
                              ListsCol{22, 23, 24, 25},
                              ListsCol{{null, null, null, null, null}, all_nulls()},
                              ListsCol{} /*NULL*/},
                             nulls_at({0, 6})}
                      .release();

  // Ignore null list elements
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}});
    auto const expected =
      ListsCol{{ListsCol{{1, null, 3, 4, 10, 11, 12, null}, nulls_at({1, 7})},
                ListsCol{{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
                ListsCol{{null, 2, 3, 4, null, 21, null, null}, nulls_at({0, 4, 6, 7})},
                ListsCol{{null, 18}, null_at(0)},
                ListsCol{{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
                ListsCol{{1, 2, 3, null, null, null, null, null, null, null},
                         nulls_at({3, 4, 5, 6, 7, 8, 9})},
                ListsCol{} /*NULL*/},
               null_at(6)}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Null list rows result in null list rows
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}},
                                    cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      ListsCol{{ListsCol{} /*NULL*/,
                ListsCol{{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
                ListsCol{} /*NULL*/,
                ListsCol{} /*NULL*/,
                ListsCol{{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
                ListsCol{{1, 2, 3, null, null, null, null, null, null, null},
                         nulls_at({3, 4, 5, 6, 7, 8, 9})},
                ListsCol{} /*NULL*/},
               nulls_at({0, 2, 3, 6})}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsWithNulls)
{
  auto const col1 = StrListsCol{
    StrListsCol{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
    StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
    StrListsCol{
      "Coconut"}}.release();
  auto const col2 =
    StrListsCol{
      {StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/},
                   nulls_at({1, 2, 3})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{{"Deer" /*NULL*/, "Snake" /*NULL*/, "Horse" /*NULL*/}, all_nulls()}}, /*NULL*/
      null_at(2)}
      .release();

  // Ignore null list elements
  {
    auto const results  = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
    auto const expected = StrListsCol{
      StrListsCol{{"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
                  nulls_at({1, 4, 5, 6})},
      StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"},
                  nulls_at({1, 4})},
      StrListsCol{
        "Coconut"}}.release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Null list rows result in null list rows
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}},
                                    cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      StrListsCol{
        {StrListsCol{
           {"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
           nulls_at({1, 4, 5, 6})},
         StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"},
                     nulls_at({1, 4})},
         StrListsCol{""} /*NULL*/},
        null_at(2)}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsWithEmptyLists)
{
  auto const col1 =
    StrListsCol{StrListsCol{{"" /*NULL*/}, null_at(0)}, StrListsCol{"One"}}.release();
  auto const col2 = StrListsCol{
    StrListsCol{{"Tomato", "" /*NULL*/, "Apple"}, null_at(1)},
    StrListsCol{
      "Two"}}.release();
  auto const col3 =
    StrListsCol{{StrListsCol{"Lemon", "Peach"}, StrListsCol{"Three"} /*NULL*/}, null_at(1)}
      .release();

  // Ignore null list elements
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}});
    auto const expected = StrListsCol{
      StrListsCol{{"" /*NULL*/, "Tomato", "" /*NULL*/, "Apple", "Lemon", "Peach"},
                  nulls_at({0, 2})},
      StrListsCol{"One",
                  "Two"}}.release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  // Null list rows result in null list rows
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}},
                                    cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      StrListsCol{{StrListsCol{{"" /*NULL*/, "Tomato", "" /*NULL*/, "Apple", "Lemon", "Peach"},
                               nulls_at({0, 2})},
                   StrListsCol{""} /*NULL*/},
                  null_at(1)}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }
}

TYPED_TEST(ListConcatenateRowsTypedTest, SlicedColumnsInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col_original = ListsCol{{1, 2, 3}, {2, 3}, {3, 4, 5, 6}, {5, 6}, {}, {7}}.release();
  auto const col1         = cudf::slice(col_original->view(), {0, 3})[0];
  auto const col2         = cudf::slice(col_original->view(), {1, 4})[0];
  auto const col3         = cudf::slice(col_original->view(), {2, 5})[0];
  auto const col4         = cudf::slice(col_original->view(), {3, 6})[0];
  auto const expected     = ListsCol{
    {1, 2, 3, 2, 3, 3, 4, 5, 6, 5, 6},
    {2, 3, 3, 4, 5, 6, 5, 6},
    {3, 4, 5, 6, 5, 6, 7}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1, col2, col3, col4}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SlicedColumnsInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col_original = ListsCol{{ListsCol{{null, 2, 3}, null_at(0)},
                                      ListsCol{2, 3}, /*NULL*/
                                      ListsCol{{3, null, 5, 6}, null_at(1)},
                                      ListsCol{5, 6}, /*NULL*/
                                      ListsCol{},     /*NULL*/
                                      ListsCol{7},
                                      ListsCol{8, 9, 10}},
                                     nulls_at({1, 3, 4})}
                              .release();
  auto const col1     = cudf::slice(col_original->view(), {0, 3})[0];
  auto const col2     = cudf::slice(col_original->view(), {1, 4})[0];
  auto const col3     = cudf::slice(col_original->view(), {2, 5})[0];
  auto const col4     = cudf::slice(col_original->view(), {3, 6})[0];
  auto const col5     = cudf::slice(col_original->view(), {4, 7})[0];
  auto const expected = ListsCol{
    ListsCol{{null, 2, 3, 3, null, 5, 6}, nulls_at({0, 4})},
    ListsCol{{3, null, 5, 6, 7}, null_at(1)},
    ListsCol{{3, null, 5, 6, 7, 8, 9, 10},
             null_at(1)}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1, col2, col3, col4, col5}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
}

TEST_F(ListConcatenateRowsTest, SlicedStringsColumnsInputWithNulls)
{
  auto const col =
    StrListsCol{
      {StrListsCol{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
       StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/},
                   nulls_at({1, 4})},
       StrListsCol{"Coconut"},
       StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/},
                   nulls_at({1, 2, 3})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{{"Deer" /*NULL*/, "Snake" /*NULL*/, "Horse" /*NULL*/}, all_nulls()}}, /*NULL*/
      null_at(5)}
      .release();
  auto const col1 = cudf::slice(col->view(), {0, 3})[0];
  auto const col2 = cudf::slice(col->view(), {1, 4})[0];
  auto const col3 = cudf::slice(col->view(), {2, 5})[0];
  auto const col4 = cudf::slice(col->view(), {3, 6})[0];

  {
    auto const results  = cudf::lists::concatenate_rows(TView{{col1, col2, col3, col4}});
    auto const expected = StrListsCol{StrListsCol{{"Tomato",
                                                   "" /*NULL*/,
                                                   "Apple",
                                                   "Banana",
                                                   "" /*NULL*/,
                                                   "Kiwi",
                                                   "Cherry",
                                                   "" /*NULL*/,
                                                   "Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "" /*NULL*/},
                                                  nulls_at({1, 4, 7, 10, 11, 12})},
                                      StrListsCol{{"Banana",
                                                   "" /*NULL*/,
                                                   "Kiwi",
                                                   "Cherry",
                                                   "" /*NULL*/,
                                                   "Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "", /*NULL*/
                                                   "Lemon",
                                                   "Peach"},
                                                  nulls_at({1, 4, 7, 8, 9})},
                                      StrListsCol{{
                                                    "Coconut",
                                                    "Orange",
                                                    "" /*NULL*/,
                                                    "" /*NULL*/,
                                                    "", /*NULL*/
                                                    "Lemon",
                                                    "Peach",
                                                  },
                                                  nulls_at({2, 3, 4})}}
                            .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }

  {
    auto const results = cudf::lists::concatenate_rows(
      TView{{col1, col2, col3, col4}}, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol{{StrListsCol{{"Tomato",
                                                    "" /*NULL*/,
                                                    "Apple",
                                                    "Banana",
                                                    "" /*NULL*/,
                                                    "Kiwi",
                                                    "Cherry",
                                                    "" /*NULL*/,
                                                    "Coconut",
                                                    "Orange",
                                                    "" /*NULL*/,
                                                    "" /*NULL*/,
                                                    "" /*NULL*/},
                                                   nulls_at({1, 4, 7, 10, 11, 12})},
                                       StrListsCol{{"Banana",
                                                    "" /*NULL*/,
                                                    "Kiwi",
                                                    "Cherry",
                                                    "" /*NULL*/,
                                                    "Coconut",
                                                    "Orange",
                                                    "" /*NULL*/,
                                                    "" /*NULL*/,
                                                    "", /*NULL*/
                                                    "Lemon",
                                                    "Peach"},
                                                   nulls_at({1, 4, 7, 8, 9})},
                                       StrListsCol{} /*NULL*/},
                                      null_at(2)}
                            .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, print_all);
  }
}
