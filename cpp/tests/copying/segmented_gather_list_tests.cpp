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
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/lists/lists_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

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
using cudf::lists_column_view;

TYPED_TEST(SegmentedGatherTest, Gather)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  LCW<int> gather_map{{3, 2, 1, 0}, {0}, {0, 1}, {0, 2, 1}};
  LCW<T> expected{{4, 3, 2, 1}, {5}, {6, 7}, {8, 10, 9}};

  auto results =
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

    LCW<T> expected{LCW<T>{}, LCW<T>{}, LCW<T>{}, LCW<T>{}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // List<List<T>>
  {
    LCW<T> list{{{1, 2, 3, 4}, {5}}, {{6, 7}}, {{}, {8, 9, 10}}};
    LCW<int> gather_map{LCW<int>{}, LCW<int>{}, LCW<int>{}};

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

    // hack to get column of empty list of list
    LCW<T> expected_dummy{{{1, 2, 3, 4}, {5}}, LCW<T>{}, LCW<T>{}, LCW<T>{}};
    auto expected = cudf::split(expected_dummy, {1})[1];
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  // List<List<List<T>>>
  {
    LCW<T> list{{{{1, 2, 3, 4}, {5}}}, {{{6, 7}, {8, 9, 10}}}};
    LCW<int> gather_map{LCW<int>{}, LCW<int>{}};

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

  auto results =
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

    auto trues = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

    LCW<T> expected{{{{2, 3}, valids}, {4, 5}},
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

    auto results =
      cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

  auto results =
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

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

    auto result0 = cudf::lists::detail::segmented_gather(
      lists_column_view{split_a[0]}, lists_column_view{LCW<int>{{1, 2}, {0, 2}, {0, 1}}});
    LCW<T> expected0{
      {{2, 2}, {3, 3}},
      {{4, 4, 4}, {6, 6}},
      {{7, 7, 7}, {8, 8}},
    };
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected0, result0->view());

    auto result1 = cudf::lists::detail::segmented_gather(
      lists_column_view{split_a[1]},
      lists_column_view{LCW<int>{{0, 1}, LCW<int>{}, LCW<int>{}, {0, 1}, LCW<int>{}}});
    LCW<T> expected1{
      {{10, 10, 10}, {11, 11}}, LCW<T>{}, LCW<T>{}, {{50, 50, 50, 50}, {6, 13}}, LCW<T>{}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result1->view());
  }

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
      LCW<int> map{{0, 1}};
      auto result =
        cudf::lists::detail::segmented_gather(lists_column_view{sliced[0]}, lists_column_view{map});
      LCW<T> expected{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}}};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    }

    // gather from slice 1
    {
      LCW<int16_t> map{{0}, {1, 2, 0, 1}, {0, 1, 2}};
      auto result =
        cudf::lists::detail::segmented_gather(lists_column_view{sliced[1]}, lists_column_view{map});
      LCW<T> expected{
        {{LCW<T>{0}}},

        {{{0, 1, 3}, {5}},
         {{11, 12, 13, 14, 15}, {16, 17}, {0}},
         {{10}, {20, 30, 40, 50}, {60, 70, 80}},
         {{0, 1, 3}, {5}}},

        {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},
      };
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    }

    // gather from slice 2
    {
      LCW<int> map{{1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 2}};
      auto result =
        cudf::lists::detail::segmented_gather(lists_column_view{sliced[2]}, lists_column_view{map});
      std::vector<bool> expected_valids = {false, true, true, false, false, true};

      LCW<T> expected{{{{LCW<T>{30}},
                        {{{10, 20}, valids}},
                        {{{10, 20}, valids}},
                        {LCW<T>{30}},
                        {LCW<T>{30}},
                        {{{10, 20}, valids}}},
                       expected_valids.begin()},
                      {{{LCW<T>{30}},
                        {{10, 20, 30}},
                        {{10, 20, 30}},
                        {LCW<T>{30}},
                        {LCW<T>{30}},
                        {{{20, 30}, valids}, {62, 72, 82}}},
                       expected_valids.begin()}};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    }
  }
}

using SegmentedGatherTestString = SegmentedGatherTest<cudf::string_view>;
TEST_F(SegmentedGatherTestString, StringGather)
{
  using T = cudf::string_view;
  // List<T>
  LCW<T> list{{"a", "b", "c", "d"}, {"1", "22", "333", "4"}, {"x", "y", "z"}};
  LCW<int8_t> gather_map{{0, 1, 3, 2}, {1, 0, 3, 2}, LCW<int8_t>{}};
  LCW<T> expected{{"a", "b", "d", "c"}, {"22", "1", "4", "333"}, LCW<T>{}};

  auto result =
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

using SegmentedGatherTestFloat = SegmentedGatherTest<float>;
TEST_F(SegmentedGatherTestFloat, GatherMapSliced)
{
  using T = float;

  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}, {11, 12}, {13, 14, 15, 16}};
  LCW<int> gather_map{{3, 2, 1, 0}, {0}, {0, 1}, {0, 2, 1}, {0}, {1}};
  // gather_map.offset: 0, 4, 5, 7, 10, 11, 12
  LCW<T> expected{{4, 3, 2, 1}, {5}, {6, 7}, {8, 10, 9}, {11}, {14}};

  auto results =
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{gather_map});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  auto sliced  = cudf::split(list, {1, 4});
  auto split_m = cudf::split(gather_map, {1, 4});
  auto split_e = cudf::split(expected, {1, 4});

  auto result0 = cudf::lists::detail::segmented_gather(lists_column_view{sliced[0]},
                                                       lists_column_view{split_m[0]});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[0], result0->view());
  auto result1 = cudf::lists::detail::segmented_gather(lists_column_view{sliced[1]},
                                                       lists_column_view{split_m[1]});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[1], result1->view());
  auto result2 = cudf::lists::detail::segmented_gather(lists_column_view{sliced[2]},
                                                       lists_column_view{split_m[2]});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[2], result2->view());
}

TEST_F(SegmentedGatherTestFloat, Fails)
{
  using T = float;
  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  LCW<int8_t> size_mismatch_map{{3, 2, 1, 0}, {0}, {0, 1}};
  cudf::test::fixed_width_column_wrapper<int> nonlist_map0{1, 2, 0, 1};
  cudf::test::strings_column_wrapper nonlist_map1{"1", "2", "0", "1"};
  LCW<cudf::string_view> nonlist_map2{{"1", "2", "0", "1"}};

  CUDF_EXPECT_THROW_MESSAGE(
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{nonlist_map0}),
    "lists_column_view only supports lists");

  CUDF_EXPECT_THROW_MESSAGE(
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{nonlist_map1}),
    "lists_column_view only supports lists");

  CUDF_EXPECT_THROW_MESSAGE(
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{nonlist_map2}),
    "Gather map should be list column of index type");

  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
  LCW<int8_t> nulls_map{{{3, 2, 1, 0}, {0}, {0}, {0, 1}}, valids};
  CUDF_EXPECT_THROW_MESSAGE(
    cudf::lists::detail::segmented_gather(lists_column_view{list}, lists_column_view{nulls_map}),
    "Gather map contains nulls");

  CUDF_EXPECT_THROW_MESSAGE(cudf::lists::detail::segmented_gather(
                              lists_column_view{list}, lists_column_view{size_mismatch_map}),
                            "Gather map and list column should be same size");
}
