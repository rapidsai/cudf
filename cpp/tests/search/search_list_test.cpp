/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>

using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

using bools_col   = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using strings_col = cudf::test::strings_column_wrapper;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};       // Mark for null child elements at the current level
constexpr int32_t XXX{0};        // Mark for null elements at all levels
constexpr int32_t dont_care{0};  // Mark for elements that will be sliced off

using TestTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::DurationTypes,
                                     cudf::test::TimestampTypes>;

template <typename T>
struct TypedListsContainsTestScalarNeedle : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(TypedListsContainsTestScalarNeedle, TestTypes);

TYPED_TEST(TypedListsContainsTestScalarNeedle, EmptyInput)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{};

  auto const needle1 = [] {
    auto child = tdata_col{};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{1, 2, 3};
    return cudf::list_scalar(child);
  }();

  EXPECT_FALSE(cudf::contains(haystack, needle1));
  EXPECT_FALSE(cudf::contains(haystack, needle2));
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, TrivialInput)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{{1, 2}, {1}, {}, {1, 3}, {4}, {1, 1}};

  auto const needle1 = [] {
    auto child = tdata_col{1, 2};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{2, 1};
    return cudf::list_scalar(child);
  }();

  EXPECT_TRUE(cudf::contains(haystack, needle1));

  // Lists are order-sensitive.
  EXPECT_FALSE(cudf::contains(haystack, needle2));
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, SlicedColumnInput)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original =
    lists_col{{dont_care, dont_care}, {dont_care}, {1, 2}, {1}, {}, {1, 3}, {dont_care, dont_care}};
  auto const haystack = cudf::slice(haystack_original, {2, 6})[0];

  auto const needle1 = [] {
    auto child = tdata_col{1, 2};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{};
    return cudf::list_scalar(child);
  }();
  auto const needle3 = [] {
    auto child = tdata_col{dont_care, dont_care};
    return cudf::list_scalar(child);
  }();

  EXPECT_TRUE(cudf::contains(haystack, needle1));
  EXPECT_TRUE(cudf::contains(haystack, needle2));
  EXPECT_FALSE(cudf::contains(haystack, needle3));
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, SimpleInputWithNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Test with invalid scalar.
  {
    auto const haystack = lists_col{{1, 2}, {1}, {}, {1, 3}, {4}, {}, {1, 1}};
    auto const needle   = [] {
      auto child = tdata_col{};
      return cudf::list_scalar(child, false);
    }();

    EXPECT_FALSE(cudf::contains(haystack, needle));
  }

  // Test with nulls at the top level.
  {
    auto const haystack =
      lists_col{{{1, 2}, {1}, {} /*NULL*/, {1, 3}, {4}, {} /*NULL*/, {1, 1}}, nulls_at({2, 5})};

    auto const needle1 = [] {
      auto child = tdata_col{1, 2};
      return cudf::list_scalar(child);
    }();
    auto const needle2 = [] {
      auto child = tdata_col{};
      return cudf::list_scalar(child, false);
    }();

    EXPECT_TRUE(cudf::contains(haystack, needle1));
    EXPECT_TRUE(cudf::contains(haystack, needle2));
  }

  // Test with nulls at the children level.
  {
    auto const haystack = lists_col{{lists_col{1, 2},
                                     lists_col{1},
                                     lists_col{{1, null}, null_at(1)},
                                     lists_col{} /*NULL*/,
                                     lists_col{1, 3},
                                     lists_col{1, 4},
                                     lists_col{4},
                                     lists_col{} /*NULL*/,
                                     lists_col{1, 1}},
                                    nulls_at({3, 7})};

    auto const needle1 = [] {
      auto child = tdata_col{{1, null}, null_at(1)};
      return cudf::list_scalar(child);
    }();
    auto const needle2 = [] {
      auto child = tdata_col{{null, 1}, null_at(0)};
      return cudf::list_scalar(child);
    }();
    auto const needle3 = [] {
      auto child = tdata_col{1, 0};
      return cudf::list_scalar(child);
    }();

    EXPECT_TRUE(cudf::contains(haystack, needle1));
    EXPECT_FALSE(cudf::contains(haystack, needle2));
    EXPECT_FALSE(cudf::contains(haystack, needle3));
  }
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, SlicedInputHavingNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original = lists_col{{{dont_care, dont_care},
                                            {dont_care} /*NULL*/,
                                            lists_col{{1, null}, null_at(1)},
                                            {1},
                                            {} /*NULL*/,
                                            {1, 3},
                                            {4},
                                            {} /*NULL*/,
                                            {1, 1},
                                            {dont_care}},
                                           nulls_at({1, 4, 7})};
  auto const haystack          = cudf::slice(haystack_original, {2, 9})[0];

  auto const needle1 = [] {
    auto child = tdata_col{{1, null}, null_at(1)};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{};
    return cudf::list_scalar(child);
  }();
  auto const needle3 = [] {
    auto child = tdata_col{dont_care, dont_care};
    return cudf::list_scalar(child);
  }();

  EXPECT_TRUE(cudf::contains(haystack, needle1));
  EXPECT_FALSE(cudf::contains(haystack, needle2));
  EXPECT_FALSE(cudf::contains(haystack, needle3));
}

template <typename T>
struct TypedListContainsTestColumnNeedles : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedListContainsTestColumnNeedles, TestTypes);

TYPED_TEST(TypedListContainsTestColumnNeedles, EmptyInput)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{};
  auto const needles  = lists_col{};
  auto const expected = bools_col{};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, TrivialInput)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{{0, 1}, {2}, {3, 4, 5}, {2, 3, 4}, {}, {0, 2, 0}};
  auto const needles  = lists_col{{0, 1}, {1}, {3, 5, 4}, {}};

  auto const expected = bools_col{1, 0, 0, 1};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, SlicedInputNoNulls)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original = lists_col{
    {dont_care, dont_care}, {dont_care}, {0, 1}, {2}, {3, 4, 5}, {2, 3, 4}, {}, {0, 2, 0}};
  auto const haystack = cudf::slice(haystack_original, {2, 8})[0];

  auto const needles_original =
    lists_col{{dont_care}, {0, 1}, {0, 0}, {3, 5, 4}, {}, {dont_care, dont_care}, {} /*dont_care*/};
  auto const needles = cudf::slice(needles_original, {1, 5})[0];

  auto const expected = bools_col{1, 0, 0, 1};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, SlicedInputHavingNulls)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original = lists_col{{{dont_care, dont_care},
                                            {dont_care} /*NULL*/,
                                            lists_col{{1, null}, null_at(1)},
                                            {1},
                                            {} /*NULL*/,
                                            {1, 3},
                                            {4},
                                            {} /*NULL*/,
                                            {1, 1},
                                            {dont_care}},
                                           nulls_at({1, 4, 7})};
  auto const haystack          = cudf::slice(haystack_original, {2, 9})[0];

  auto const needles_original = lists_col{{{dont_care, dont_care},
                                           {dont_care} /*NULL*/,
                                           lists_col{{1, null}, null_at(1)},
                                           {1},
                                           {} /*NULL*/,
                                           {1, 3, 1},
                                           {4},
                                           {} /*NULL*/,
                                           {},
                                           {dont_care}},
                                          nulls_at({1, 4, 7})};
  auto const needles          = cudf::slice(needles_original, {2, 9})[0];

  auto const expected = bools_col{{1, 1, null, 0, 1, null, 0}, nulls_at({2, 5})};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, ListsOfStructs)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const haystack = [] {
    auto offsets = int32s_col{0, 2, 3, 5, 8, 10};
    // clang-format off
    auto data1 = tdata_col{1, 2,     //
                           1,        //
                           0, 1,     //
                           1, 3, 4,  //
                           0, 0      //
    };
    auto data2 = tdata_col{1, 3,     //
                           2,        //
                           1, 1,     //
                           0, 2, 0,  //
                           1, 2      //
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(5, offsets.release(), child.release(), 0, {});
  }();

  auto const needles = [] {
    auto offsets = int32s_col{0, 3, 4, 6, 9, 11};
    // clang-format off
    auto data1 = tdata_col{1, 2, 1,  //
                           1,        //
                           0, 1,     //
                           1, 3, 4,  //
                           0, 0      //
    };
    auto data2 = tdata_col{1, 3, 0,  //
                           2,        //
                           1, 1,     //
                           0, 2, 2,  //
                           1, 1      //
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(5, offsets.release(), child.release(), 0, {});
  }();

  auto const expected = bools_col{0, 1, 1, 0, 0};
  auto const result   = cudf::contains(*haystack, *needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

struct ListLowerBound : public cudf::test::BaseFixture {};

TEST_F(ListLowerBound, ListWithNulls)
{
  {
    using lcw           = cudf::test::lists_column_wrapper<double>;
    auto const haystack = lcw{
      lcw{-3.45967821e+12},  // 0
      lcw{-3.6912186e-32},   // 1
      lcw{9.721175},         // 2
    };

    auto const needles = lcw{
      lcw{{0, 4.22671e+32}, null_at(0)},
    };

    auto const expect = int32s_col{0};
    auto const result = cudf::lower_bound(cudf::table_view{{haystack}},
                                          cudf::table_view{{needles}},
                                          {cudf::order::ASCENDING},
                                          {cudf::null_order::BEFORE});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *result);
  }

  {
    using lcw       = cudf::test::lists_column_wrapper<int32_t, int32_t>;
    auto const col1 = lcw{
      lcw{{0}, null_at(0)},  // 0
      lcw{-80},              // 1
      lcw{-17},              // 2
    };

    auto const col2 = lcw{
      lcw{27},               // 0
      lcw{{0}, null_at(0)},  // 1
      lcw{},                 // 2
    };

    auto const val1 = lcw{
      lcw{87},
    };

    auto const val2 = lcw{
      lcw{},
    };

    cudf::table_view input{{col1, col2}};
    cudf::table_view values{{val1, val2}};
    std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
    std::vector<cudf::null_order> null_order_flags{cudf::null_order::BEFORE,
                                                   cudf::null_order::BEFORE};

    auto const expect = int32s_col{3};
    auto const result = cudf::lower_bound(input, values, column_order, null_order_flags);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *result);
  }
}
