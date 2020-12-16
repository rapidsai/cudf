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
#include <bits/stdint-intn.h>
#include <tests/strings/utilities.h>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/lists_column_view.hpp>

template <typename T>
class SegmentedGatherTest : public cudf::test::BaseFixture {
};
using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::DurationTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(SegmentedGatherTest, FixedWidthTypesNotBool);

class SegmentedGatherTestList : public cudf::test::BaseFixture {
};

// to disambiguate between {} == 0 and {} == List{0}
// Also, see note about compiler issues when declaring nested
// empty lists in lists_column_wrapper documentation
template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

TYPED_TEST(SegmentedGatherTest, Gather)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  LCW<int> gather_map{{3, 2, 1, 0}, {0}, {0, 1}, {0, 2, 1}};
  LCW<T> expected{{4, 3, 2, 1}, {5}, {6, 7}, {8, 10, 9}};

  auto results = cudf::lists::detail::segmented_gather(list, gather_map);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TYPED_TEST(SegmentedGatherTest, GatherNothing)
{
  using T = TypeParam;
  using namespace cudf;

  // List<T>
  {
    LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
    LCW<int> gather_map{LCW<int>{}, LCW<int>{}, LCW<int>{}, LCW<int>{}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    LCW<T> expected{LCW<T>{}, LCW<T>{}, LCW<T>{}, LCW<T>{}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // List<List<T>>
  {
    LCW<T> list{{{1, 2, 3, 4}, {5}}, {{6, 7}}, {{}, {8, 9, 10}}};
    LCW<int> gather_map{LCW<int>{}, LCW<int>{}, LCW<int>{}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    // hack to get column of empty list of list
    LCW<T> expected_dummy{{{1, 2, 3, 4}, {5}}, LCW<T>{}, LCW<T>{}, LCW<T>{}};
    auto expected = cudf::split(expected_dummy, {1})[1];
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  // List<List<List<T>>>
  {
    LCW<T> list{{{{1, 2, 3, 4}, {5}}}, {{{6, 7}, {8, 9, 10}}}};
    LCW<int> gather_map{LCW<int>{}, LCW<int>{}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    LCW<T> expected_dummy{{{{1, 2, 3, 4}}},  // hack to get column of empty list of list of list
                          LCW<T>{},
                          LCW<T>{}};
    auto expected = cudf::split(expected_dummy, {1})[1];
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(results->view());
    EXPECT_EQ(lcv.size(), 2);
    EXPECT_EQ(lcv.child().type().id(), type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(lists_column_view(lcv.child()).child().type().id(), type_id::LIST);
    EXPECT_EQ(lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().type().id(),
              type_to_id<T>());
    EXPECT_EQ(lists_column_view(lists_column_view(lcv.child()).child()).child().size(), 0);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNulls)
{
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<T>
  LCW<T> list{{{1, 2, 3, 4}, valids}, {5}, {{6, 7}, valids}, {{8, 9, 10}, valids}};
  LCW<int> gather_map{{0, 1}, LCW<int>{}, {1}, {2, 1, 0}};

  auto results = cudf::lists::detail::segmented_gather(list, gather_map);

  LCW<T> expected{{{1, 2}, valids}, LCW<T>{}, {{7}, valids + 1}, {{10, 9, 8}, valids}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TYPED_TEST(SegmentedGatherTest, GatherNested)
{
  using T = TypeParam;

  // List<List<T>>
  {
    LCW<T> list{{{2, 3}, {4, 5}},
                {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {-17, -18}}};
    LCW<int> gather_map{{0, 2, -2}, {1}, {1, 0, -1, 5}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    LCW<T> expected{
      {{2, 3}, {2, 3}, {2, 3}}, {{9, 10, 11}}, {{17, 18}, {15, 16}, {-17, -18}, {15, 16}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // List<List<List<T>>>
  {
    LCW<T> list{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
                {{LCW<T>{0}}},
                {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                 {{0, 1, 3}, {5}},
                 {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                {{{10, 20}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}};
    LCW<int> gather_map{{1}, LCW<int>{}, {0}, {1}, {0, -1, 1}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    LCW<T> expected{{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                    LCW<T>{},
                    {{LCW<T>{0}}},
                    {{{0, 1, 3}, {5}}},
                    {{{10, 20}}, {{40, 50}, {60, 70, 80}}, {LCW<T>{30}}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherOutOfOrder)
{
  using T = TypeParam;

  // List<List<T>>
  {
    LCW<T> list{{{2, 3}, {4, 5}},
                {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    LCW<int> gather_map{{1, 0}, {1, 2, 0}, {5, 4, 3, 2, 1, 0}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    LCW<T> expected{{{4, 5}, {2, 3}},
                    {{9, 10, 11}, {12, 13, 14}, {6, 7, 8}},
                    {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNegatives)
{
  using T = TypeParam;

  // List<List<T>>
  {
    LCW<T> list{{{2, 3}, {4, 5}},
                {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    LCW<int> gather_map{{-1, 0}, {-2, -1, 0}, {-5, -4, -3, -2, -1, 0}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    LCW<T> expected{{{4, 5}, {2, 3}},
                    {{9, 10, 11}, {12, 13, 14}, {6, 7, 8}},
                    {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNestedNulls)
{
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<List<T>>
  {
    LCW<T> list{{{{2, 3}, valids}, {4, 5}},
                {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
                {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};

    LCW<int> gather_map{{0, 1}, {0, 2}, LCW<int>{}, {0, 1, 4}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

  auto trues = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return true; });

    LCW<T> expected{
      {{{2, 3}, valids}, {4, 5}},
      {{{6, 7, 8}, {12, 13, 14}}, trues},
      LCW<T>{},
      {{{{25, 26}, valids}, {27, 28}, {33, 34}}, valids}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // List<List<List<List<T>>>>
  {
    LCW<T> list{{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
                {{LCW<T>{0}}},
                {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                 {{0, 1, 3}, {5}},
                 {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}}};

    LCW<int> gather_map{{1, 2, 4}};

    auto results = cudf::lists::detail::segmented_gather(list, gather_map);

    LCW<T> expected{{{{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
                    {{LCW<T>{0}}},
                    {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNestedWithEmpties)
{
  using T = TypeParam;

  LCW<T> list{{{2, 3}, LCW<T>{}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, {LCW<T>{}}};
  LCW<int> gather_map{LCW<int>{0}, LCW<int>{0}, LCW<int>{0}};

  auto results = cudf::lists::detail::segmented_gather(list, gather_map);

  // skip one null, gather one null.
  LCW<T> expected{{{2, 3}}, {{6, 7, 8}}, {LCW<T>{}}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TYPED_TEST(SegmentedGatherTest, GatherSliced)
{
  using T = TypeParam;
  {
    LCW<T> a{
      {{1, 1, 1}, {2, 2}, {3, 3}},
      {{4, 4, 4}, {5, 5}, {6, 6}},
      {{7, 7, 7}, {8, 8}, {9, 9}},
      {{10, 10, 10}, {11, 11}, {12, 12}},
      {{20, 20, 20, 20}, {25}},
      {{30, 30, 30, 30}, {40}},
      {{50, 50, 50, 50}, {6, 13}},
      {{70, 70, 70, 70}, {80}},
    };
    auto split_a = cudf::split(a, {3});

    auto result0 =
      cudf::lists::detail::segmented_gather(split_a[0], LCW<int>{{1, 2}, {0, 2}, {0, 1}});
    LCW<T> expected0{
      {           {2, 2}, {3, 3}},
      {{4, 4, 4},         {6, 6}},
      {{7, 7, 7}, {8, 8}        },
    };
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected0, result0->view());

    auto result1 = cudf::lists::detail::segmented_gather(split_a[1], LCW<int>{{0, 1}, LCW<int>{}, LCW<int>{}, {0,1}, LCW<int>{}});
    LCW<T> expected1{
      {{10, 10, 10}, {11, 11}},
      LCW<T>{},
      LCW<T>{},
      {{50, 50, 50, 50}, {6, 13}},
      LCW<T>{}
    };
    cudf::test::print(expected1);
    cudf::test::print(*result1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result1->view());
  }
/*
  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  // List<List<List<T>>>
  {
    LCW<T> list{
      // slice 0
      {{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},

      {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}},
       {{11, 12}, {{42, 43, 44}, valids}, {{77, 78}, valids}}},

      // slice 1
      {{LCW<T>{0}}},
      {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
       {{0, 1, 3}, {5}},
       {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},

      // slice 2
      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids},
      {{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids}};

    auto sliced = cudf::slice(list, {0, 1, 2, 5, 5, 7});

    // gather from slice 0
    {
      cudf::table_view tbl({sliced[0]});

      cudf::test::fixed_width_column_wrapper<int> map{0};
      auto result = cudf::gather(tbl, map);
      LCW<T> expected{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}}};
      cudf::test::expect_columns_equivalent(expected, result->get_column(0).view());
    }

    // gather from slice 1
    {
      cudf::table_view tbl({sliced[1]});

      cudf::test::fixed_width_column_wrapper<int> map{1, 2, 0, 1};
      auto result = cudf::gather(tbl, map);
      LCW<T> expected{
        {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
         {{0, 1, 3}, {5}},
         {{11, 12, 13, 14, 15}, {16, 17}, {0}}},

        {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},

        {{LCW<T>{0}}},

        {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
         {{0, 1, 3}, {5}},
         {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      };
      cudf::test::expect_columns_equivalent(expected, result->get_column(0).view());
    }

    // gather from slice 2
    {
      cudf::table_view tbl({sliced[2]});

      cudf::test::fixed_width_column_wrapper<int> map{1, 0, 0, 1, 1, 0};
      auto result = cudf::gather(tbl, map);
      LCW<T> expected{{{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids},
                      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids},
                      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids},
                      {{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids},
                      {{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids},
                      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}};
      cudf::test::expect_columns_equivalent(expected, result->get_column(0).view());
    }
  }
//*/
}

TYPED_TEST(SegmentedGatherTest, child_index)
{
  using T = int32_t;
  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  LCW<int8_t> gather_map{{3, 2, 1, 0}, {0}, {}, {2, 1}};

  auto results = cudf::lists::detail::segmented_gather(list, gather_map);

  LCW<T> expected{{4, 3, 2, 1}, {5}, {}, {10, 9}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  return;

  LCW<T> g1{{-1, -2}, {-4, -5, -6}};
  LCW<T> e1{{-1, -2}, LCW<T>{}, {-4, -5, -6}};
  cudf::test::print(g1, std::cout << "g1=");
  cudf::test::print(e1, std::cout << "e1=");
  LCW<T> g2{{{-1, -2}}, {{-4, -5, -6}}};
  LCW<T> e2{{{-1, -2}}, LCW<T>{}, {{-4, -5, -6}}};
  cudf::test::print(g2, std::cout << "g2=");
  cudf::test::print(e2, std::cout << "e2=");
  LCW<T> g3{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}};
  LCW<T> e3{{{2, 3}, {4, 5}}, LCW<T>{}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}};
  cudf::test::print(g3, std::cout << "g3=");
  cudf::test::print(e3, std::cout << "e3=");
  LCW<T> a3{{{-2, -3}, {-4, -5}},  // hack to get column of List<List<int>
            LCW<T>{},
            LCW<T>{},
            LCW<T>{}};
  auto a33 = cudf::split(a3, {1})[1];
  LCW<int8_t> gm3{LCW<int8_t>{}, LCW<int8_t>{}, LCW<int8_t>{}};
  cudf::test::print(gm3, std::cout << "gm3=");
  auto r3 = cudf::lists::detail::segmented_gather(e3, gm3);
  cudf::test::print(a3, std::cout << "a3=");
  cudf::test::print(a33, std::cout << "a33=");
  cudf::test::print(*r3, std::cout << "r3=");
}
