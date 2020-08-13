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
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/lists/lists_column_view.hpp>

template <typename T>
class GatherTestListTyped : public cudf::test::BaseFixture {
};
using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(GatherTestListTyped, FixedWidthTypesNotBool);

class GatherTestList : public cudf::test::BaseFixture {
};

TYPED_TEST(GatherTestListTyped, Gather)
{
  using T = TypeParam;

  // List<T>
  cudf::test::lists_column_wrapper<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

  cudf::table_view source_table({list});
  auto results = cudf::gather(source_table, gather_map);

  cudf::test::lists_column_wrapper<T> expected{{1, 2, 3, 4}, {6, 7}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TYPED_TEST(GatherTestListTyped, GatherNothing)
{
  using T = TypeParam;
  using namespace cudf;

  // List<T>
  {
    cudf::test::lists_column_wrapper<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected;

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  }

  // List<T>
  {
    cudf::test::lists_column_wrapper<int> list{{{{1, 2, 3, 4}, {5}}}, {{{6, 7}, {8, 9, 10}}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{};

    cudf::table_view source_table({list});
    auto result = cudf::gather(source_table, gather_map);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(result->view().column(0));
    EXPECT_EQ(lcv.size(), 0);
    EXPECT_EQ(lcv.child().type().id(), type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(lists_column_view(lcv.child()).child().type().id(), type_id::LIST);
    EXPECT_EQ(lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().type().id(),
              type_id::INT32);
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().size(), 0);
  }
}

TYPED_TEST(GatherTestListTyped, GatherNulls)
{
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<T>
  cudf::test::lists_column_wrapper<T> list{
    {{1, 2, 3, 4}, valids}, {5}, {{6, 7}, valids}, {{8, 9, 10}, valids}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

  cudf::table_view source_table({list});
  auto results = cudf::gather(source_table, gather_map);

  cudf::test::lists_column_wrapper<T> expected{{{1, 2, 3, 4}, valids}, {{6, 7}, valids}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TYPED_TEST(GatherTestListTyped, GatherNested)
{
  using T = TypeParam;
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{{{2, 3}, {4, 5}},
                                             {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                             {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{2, 3}, {4, 5}}, {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  }

  // List<List<List<T>>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
      {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
      {{LCW{0}}},
      {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
       {{0, 1, 3}, {5}},
       {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      {{{10, 20}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{1, 2, 4};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
      {{LCW{0}}},
      {{{10, 20}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  }
}

TYPED_TEST(GatherTestListTyped, GatherOutOfOrder)
{
  using T = TypeParam;
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{{{2, 3}, {4, 5}},
                                             {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                             {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{1, 2, 0};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                                 {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
                                                 {{2, 3}, {4, 5}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  }
}

TYPED_TEST(GatherTestListTyped, GatherNestedNulls)
{
  using T = TypeParam;
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{{2, 3}, valids}, {4, 5}},
      {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
      {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
      {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 1, 3};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{{2, 3}, valids}, {4, 5}},
      {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
      {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  }

  // List<List<List<T>>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
      {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
      {{LCW{0}}},
      {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
       {{0, 1, 3}, {5}},
       {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      {{{{{10, 20}, valids}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}, valids}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{1, 2, 4};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
      {{LCW{0}}},
      {{{{{10, 20}, valids}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}, valids}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  }
}

TYPED_TEST(GatherTestListTyped, GatherNestedWithEmpties)
{
  using T = TypeParam;
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;

  cudf::test::lists_column_wrapper<T> list{
    {{2, 3}, LCW{}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, {LCW{}}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

  cudf::table_view source_table({list});
  auto results = cudf::gather(source_table, gather_map);

  cudf::test::lists_column_wrapper<T> expected{{{2, 3}, LCW{}}, {LCW{}}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TYPED_TEST(GatherTestListTyped, GatherDetailInvalidIndex)
{
  using T = TypeParam;
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{{{2, 3}, {4, 5}},
                                             {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                             {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 15, 16, 2};

    cudf::table_view source_table({list});
    auto results = cudf::detail::gather(source_table,
                                        gather_map,
                                        cudf::detail::out_of_bounds_policy::IGNORE,
                                        cudf::detail::negative_index_policy::NOT_ALLOWED);

    std::vector<int32_t> expected_validity{1, 0, 0, 1};
    cudf::test::lists_column_wrapper<T> expected{
      {{{2, 3}, {4, 5}}, LCW{}, LCW{}, {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
      expected_validity.begin()};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  }
}

TEST_F(GatherTestList, GatherIncompleteHierarchies)
{
  using LCW = cudf::test::lists_column_wrapper<int32_t>;
  using namespace cudf;

  {
    // List<List<List<int>, but rows 1 and 2 are empty at the very top.
    // We expect to get back a "full" hierarchy of type List<List<List<int>> anyway.
    cudf::test::lists_column_wrapper<int32_t> list{{{{1, 2}}}, LCW{}, LCW{}};

    cudf::table_view source_table({list});

    cudf::test::fixed_width_column_wrapper<int32_t> row1_map{1};
    auto result = cudf::gather(source_table, row1_map);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(result->view().column(0));
    EXPECT_EQ(lcv.size(), 1);
    EXPECT_EQ(lcv.child().type().id(), type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(lists_column_view(lcv.child()).child().type().id(), type_id::LIST);
    EXPECT_EQ(lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().type().id(),
              type_id::INT32);
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().size(), 0);
  }

  {
    // List<List<List<int>, gathering nothing.
    // We expect to get back a "full" hierarchy of type List<List<List<int>> anyway.
    cudf::test::lists_column_wrapper<int32_t> list{{{{1, 2}}}, LCW{}};

    cudf::table_view source_table({list});

    cudf::test::fixed_width_column_wrapper<int32_t> empty_map{};
    auto result = cudf::gather(source_table, empty_map);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(result->view().column(0));
    EXPECT_EQ(lcv.size(), 0);
    EXPECT_EQ(lcv.child().type().id(), type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(lists_column_view(lcv.child()).child().type().id(), type_id::LIST);
    EXPECT_EQ(lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().type().id(),
              type_id::INT32);
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().size(), 0);
  }
}
