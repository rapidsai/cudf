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
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/lists/explode.hpp>

using namespace cudf::test;
using FCW = fixed_width_column_wrapper<int32_t>;
using LCW = lists_column_wrapper<int32_t>;

class ExplodeTest : public cudf::test::BaseFixture {
};

class ExplodeOuterTest : public cudf::test::BaseFixture {
};

template <typename T>
class ExplodeTypedTest : public cudf::test::BaseFixture {
};

template <typename T>
class ExplodeOuterTypedTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ExplodeTypedTest, cudf::test::FixedPointTypes);

TYPED_TEST_CASE(ExplodeOuterTypedTest, cudf::test::FixedPointTypes);

TEST_F(ExplodeTest, Empty)
{
  cudf::table_view t({LCW{}, FCW{}});

  auto ret = cudf::explode(t, 0);

  cudf::table_view expected({FCW{}, FCW{}});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  auto pos_ret = cudf::explode_position(t, 0);

  cudf::table_view pos_expected({FCW{}, FCW{}, FCW{}});

  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NonList)
{
  cudf::table_view t({FCW{100, 200, 300}, FCW{100, 200, 300}});

  EXPECT_THROW(cudf::explode(t, 1), cudf::logic_error);
  EXPECT_THROW(cudf::explode_position(t, 1), cudf::logic_error);
}

TEST_F(ExplodeTest, Basics)
{
  //    a                   b                  c
  //    100                [1, 2, 7]           string0
  //    200                [5, 6]              string1
  //    300                [0, 3]              string2

  FCW a{100, 200, 300};
  LCW b{LCW{1, 2, 7}, LCW{5, 6}, LCW{0, 3}};
  strings_column_wrapper c{"string0", "string1", "string2"};

  FCW expected_a{100, 100, 100, 200, 200, 300, 300};
  FCW expected_b{1, 2, 7, 5, 6, 0, 3};
  strings_column_wrapper expected_c{
    "string0", "string0", "string0", "string1", "string1", "string2", "string2"};

  cudf::table_view t({a, b, c});
  cudf::table_view expected({expected_a, expected_b, expected_c});

  auto ret = cudf::explode(t, 1);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_a, expected_pos_col, expected_b, expected_c});

  auto pos_ret = cudf::explode_position(t, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, SingleNull)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    [5, 6]              200
  //    []                  300
  //    [0, 3]              400

  auto first_invalid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i == 0 ? false : true; });

  LCW a({LCW{1, 2, 7}, LCW{5, 6}, LCW{}, LCW{0, 3}}, first_invalid);
  FCW b({100, 200, 300, 400});

  FCW expected_a{5, 6, 0, 3};
  FCW expected_b{200, 200, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, Nulls)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    [5, 6]              200
  //    [0, 3]              300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  auto always_valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  LCW a({LCW{1, 2, 7}, LCW{5, 6}, LCW{0, 3}}, valids);
  FCW b({100, 200, 300}, valids);

  FCW expected_a({1, 2, 7, 0, 3});
  FCW expected_b({100, 100, 100, 300, 300}, always_valid);

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NullsInList)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    [5, 6, 0, 9]        200
  //    []                  300
  //    [0, 3, 8]           400

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a{LCW({1, 2, 7}, valids), LCW({5, 6, 0, 9}, valids), LCW{}, LCW({0, 3, 8}, valids)};
  FCW b{100, 200, 300, 400};

  FCW expected_a({1, 2, 7, 5, 6, 0, 9, 0, 3, 8}, {1, 0, 1, 1, 0, 1, 0, 1, 0, 1});
  FCW expected_b{100, 100, 100, 200, 200, 200, 200, 400, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 2, 3, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, Nested)
{
  //    a                      b
  //    [[1, 2], [7, 6, 5]]    100
  //    [[5, 6]]               200
  //    [[0, 3],[],[5],[2, 1]] 300

  LCW a{LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{5, 6}}, LCW{LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}}};
  FCW b{100, 200, 300};

  LCW expected_a{LCW{1, 2}, LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}};
  FCW expected_b{100, 100, 200, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2, 3};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NestedNulls)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] 100
  //    [[5, 6]]            200
  //    [[0, 3],[5],[2, 1]] 300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  auto always_valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  LCW a({LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{5, 6}}, LCW{LCW{0, 3}, LCW{5}, LCW{2, 1}}}, valids);
  FCW b({100, 200, 300}, valids);

  LCW expected_a{LCW{1, 2}, LCW{7, 6, 5}, LCW{0, 3}, LCW{5}, LCW{2, 1}};
  FCW expected_b({100, 100, 300, 300, 300}, always_valid);

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NullsInNested)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] 100
  //    [[5, 6]]            200
  //    [[0, 3],[5],[2, 1]] 300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{LCW({1, 2}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)}});
  FCW b({100, 200, 300});

  LCW expected_a{
    LCW({1, 2}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)};
  FCW expected_b{100, 100, 200, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NullsInNestedDoubleExplode)
{
  //    a                       b
  //    [[1, 2], [], [7, 6, 5]] 100
  //    [[5, 6]]                200
  //    [[0, 3],[5],[2, 1]]     300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a{LCW{LCW({1, 2}, valids), LCW{}, LCW{7, 6, 5}},
        LCW{LCW{5, 6}},
        LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)}};
  FCW b{100, 200, 300};

  FCW expected_a({1, 2, 7, 6, 5, 5, 6, 0, 3, 5, 2, 1}, {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  FCW expected_b{100, 100, 100, 100, 100, 200, 200, 300, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto first_explode_ret = cudf::explode(t, 0);
  auto ret               = cudf::explode(first_explode_ret->view(), 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(first_explode_ret->view(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NestedStructs)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] {100, "100"}
  //    [[5, 6]]            {200, "200"}
  //    [[0, 3],[5],[2, 1]] {300, "300"}

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{LCW({1, 2}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)}});
  FCW b1({100, 200, 300});
  strings_column_wrapper b2{"100", "200", "300"};
  structs_column_wrapper b({b1, b2});

  LCW expected_a{
    LCW({1, 2}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)};
  FCW expected_b1{100, 100, 200, 300, 300, 300};
  strings_column_wrapper expected_b2{"100", "100", "200", "300", "300", "300"};
  structs_column_wrapper expected_b({expected_b1, expected_b2});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TYPED_TEST(ExplodeTypedTest, ListOfStructs)
{
  //  a                        b
  //  [{70, "70"}, {75, "75"}] 100
  //  [{50, "50"}, {55, "55"}] 200
  //  [{35, "35"}, {45, "45"}] 300
  //  [{25, "25"}, {30, "30"}] 400
  //  [{15, "15"}, {20, "20"}] 500

  auto numeric_col =
    fixed_width_column_wrapper<TypeParam, int32_t>{{70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  strings_column_wrapper string_col{"70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};
  auto struct_col = structs_column_wrapper{{numeric_col, string_col}}.release();
  auto a          = cudf::make_lists_column(
    5, FCW{0, 2, 4, 6, 8, 10}.release(), std::move(struct_col), cudf::UNKNOWN_NULL_COUNT, {});

  FCW b{100, 200, 300, 400, 500};

  cudf::table_view t({a->view(), b});
  auto ret = cudf::explode(t, 0);

  auto expected_numeric_col =
    fixed_width_column_wrapper<TypeParam, int32_t>{{70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  strings_column_wrapper expected_string_col{
    "70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};

  auto expected_a = structs_column_wrapper{{expected_numeric_col, expected_string_col}}.release();
  FCW expected_b{100, 100, 200, 200, 300, 300, 400, 400, 500, 500};

  cudf::table_view expected({expected_a->view(), expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a->view(), expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, SlicedList)
{
  //    a                        b
  //    [[1, 2],[7, 6, 5]]       100
  //    [[5, 6]]                 200
  //    [[0, 3],[5],[2, 1]]      300
  //    [[8, 3],[],[4, 3, 1, 2]] 400
  //    [[2, 3, 4],[9, 8]]       500

  //    slicing the top 2 rows and the bottom row off

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{LCW({1, 2}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)},
         LCW{LCW{8, 3}, LCW{}, LCW({4, 3, 1, 2}, valids)},
         LCW{LCW{2, 3, 4}, LCW{9, 8}}});
  FCW b({100, 200, 300, 400, 500});

  LCW expected_a{
    LCW{0, 3}, LCW{5}, LCW({2, 1}, valids), LCW{8, 3}, LCW{}, LCW({4, 3, 1, 2}, valids)};
  FCW expected_b{300, 300, 300, 400, 400, 400};

  cudf::table_view t({a, b});
  auto sliced_t = cudf::slice(t, {2, 4});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(sliced_t[0], 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(sliced_t[0], 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, Empty)
{
  LCW a{};
  FCW b{};

  cudf::table_view t({LCW{}, FCW{}});

  auto ret = cudf::explode_outer(t, 0);

  FCW expected_a{};
  FCW expected_b{};
  cudf::table_view expected({FCW{}, FCW{}});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);
}

TEST_F(ExplodeOuterTest, NonList)
{
  cudf::table_view t({FCW{100, 200, 300}, FCW{100, 200, 300}});

  EXPECT_THROW(cudf::explode_outer(t, 1), cudf::logic_error);
  EXPECT_THROW(cudf::explode_outer_position(t, 1), cudf::logic_error);
}

TEST_F(ExplodeOuterTest, Basics)
{
  //    a                   b                  c
  //    100                [1, 2, 7]           string0
  //    200                [5, 6]              string1
  //    300                [0, 3]              string2

  FCW a{100, 200, 300};
  LCW b{LCW{1, 2, 7}, LCW{5, 6}, LCW{0, 3}};
  strings_column_wrapper c{"string0", "string1", "string2"};

  FCW expected_a{100, 100, 100, 200, 200, 300, 300};
  FCW expected_b{1, 2, 7, 5, 6, 0, 3};
  strings_column_wrapper expected_c{
    "string0", "string0", "string0", "string1", "string1", "string2", "string2"};

  cudf::table_view t({a, b, c});
  cudf::table_view expected({expected_a, expected_b, expected_c});

  auto ret = cudf::explode_outer(t, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_a, expected_pos_col, expected_b, expected_c});

  auto pos_ret = cudf::explode_outer_position(t, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, SingleNull)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    [5, 6]              200
  //    []                  300
  //    [0, 3]              400

  auto first_invalid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i == 0 ? false : true; });

  LCW a({LCW{1, 2, 7}, LCW{5, 6}, LCW{}, LCW{0, 3}}, first_invalid);
  FCW b({100, 200, 300, 400});

  FCW expected_a{{0, 5, 6, 0, 0, 3}, {0, 1, 1, 0, 1, 1}};
  FCW expected_b{100, 200, 200, 300, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 0, 1, 0, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, Nulls)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    [5, 6]              200
  //    [0, 3]              300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{1, 2, 7}, LCW{5, 6}, LCW{0, 3}}, valids);
  FCW b({100, 200, 300}, valids);

  FCW expected_a({1, 2, 7, 0, 0, 3}, {1, 1, 1, 0, 1, 1});
  FCW expected_b({100, 100, 100, 200, 300, 300}, {1, 1, 1, 0, 1, 1});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NullsInList)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    [5, 6, 0, 9]        200
  //    []                  300
  //    [0, 3, 8]           400

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a{LCW({1, 2, 7}, valids), LCW({5, 6, 0, 9}, valids), LCW{}, LCW({0, 3, 8}, valids)};
  FCW b{100, 200, 300, 400};

  FCW expected_a({1, 2, 7, 5, 6, 0, 9, 0, 0, 3, 8}, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1});
  FCW expected_b{100, 100, 100, 200, 200, 200, 200, 300, 400, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 2, 3, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, Nested)
{
  //    a                      b
  //    [[1, 2], [7, 6, 5]]    100
  //    [[5, 6]]               200
  //    [[0, 3],[],[5],[2, 1]] 300

  LCW a{LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{5, 6}}, LCW{LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}}};
  FCW b{100, 200, 300};

  LCW expected_a{LCW{1, 2}, LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}};
  FCW expected_b{100, 100, 200, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2, 3};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NestedNulls)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] 100
  //    [[5, 6]]            200
  //    [[0, 3],[5],[2, 1]] 300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{5, 6}}, LCW{LCW{0, 3}, LCW{5}, LCW{2, 1}}}, valids);
  FCW b({100, 200, 300});

  auto expected_valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i == 2 ? false : true; });
  LCW expected_a({LCW{1, 2}, LCW{7, 6, 5}, LCW{}, LCW{0, 3}, LCW{5}, LCW{2, 1}}, expected_valids);
  FCW expected_b({100, 100, 200, 300, 300, 300});
  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NullsInNested)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] 100
  //    [[5, 6]]            200
  //    [[0, 3],[5],[2, 1]] 300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{LCW({1, 2}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)}});
  FCW b({100, 200, 300});

  LCW expected_a{
    LCW({1, 2}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)};
  FCW expected_b{100, 100, 200, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NullsInNestedDoubleExplode)
{
  //    a                       b
  //    [[1, 2], [], [7, 6, 5]] 100
  //    [[5, 6]]                200
  //    [[0, 3],[5],[2, 1]]     300

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a{LCW{LCW({1, 2}, valids), LCW{}, LCW{7, 6, 5}},
        LCW{LCW{5, 6}},
        LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)}};
  FCW b{100, 200, 300};

  FCW expected_a({1, 2, 0, 7, 6, 5, 5, 6, 0, 3, 5, 2, 1}, {1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  FCW expected_b{100, 100, 100, 100, 100, 100, 200, 200, 300, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto first_explode_ret = cudf::explode_outer(t, 0);
  auto ret               = cudf::explode_outer(first_explode_ret->view(), 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(first_explode_ret->view(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NestedStructs)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] {100, "100"}
  //    [[5, 6]]            {200, "200"}
  //    [[0, 3],[5],[2, 1]] {300, "300"}

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{LCW({1, 2}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)}});
  FCW b1({100, 200, 300});
  strings_column_wrapper b2{"100", "200", "300"};
  structs_column_wrapper b({b1, b2});

  LCW expected_a{
    LCW({1, 2}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)};
  FCW expected_b1{100, 100, 200, 300, 300, 300};
  strings_column_wrapper expected_b2{"100", "100", "200", "300", "300", "300"};
  structs_column_wrapper expected_b({expected_b1, expected_b2});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TYPED_TEST(ExplodeOuterTypedTest, ListOfStructs)
{
  //  a                        b
  //  [{70, "70"}, {75, "75"}] 100
  //  [{50, "50"}, {55, "55"}] 200
  //  [{35, "35"}, {45, "45"}] 300
  //  [{25, "25"}, {30, "30"}] 400
  //  [{15, "15"}, {20, "20"}] 500

  auto numeric_col =
    fixed_width_column_wrapper<TypeParam, int32_t>{{70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  strings_column_wrapper string_col{"70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};
  auto struct_col = structs_column_wrapper{{numeric_col, string_col}}.release();
  auto a          = cudf::make_lists_column(
    5, FCW{0, 2, 4, 6, 8, 10}.release(), std::move(struct_col), cudf::UNKNOWN_NULL_COUNT, {});

  FCW b{100, 200, 300, 400, 500};

  cudf::table_view t({a->view(), b});
  auto ret = cudf::explode_outer(t, 0);

  auto expected_numeric_col =
    fixed_width_column_wrapper<TypeParam, int32_t>{{70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  strings_column_wrapper expected_string_col{
    "70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};

  auto expected_a = structs_column_wrapper{{expected_numeric_col, expected_string_col}}.release();
  FCW expected_b{100, 100, 200, 200, 300, 300, 400, 400, 500, 500};

  cudf::table_view expected({expected_a->view(), expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a->view(), expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, SlicedList)
{
  //    a                        b
  //    [[1, 2],[7, 6, 5]]       100
  //    [[5, 6]]                 200
  //    [[0, 3],[5],[2, 1]]      300
  //    [[8, 3],[],[4, 3, 1, 2]] 400
  //    [[2, 3, 4],[9, 8]]       500

  //    slicing the top 2 rows and the bottom row off

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  LCW a({LCW{LCW({1, 2}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)},
         LCW{LCW{8, 3}, LCW{}, LCW({4, 3, 1, 2}, valids)},
         LCW{LCW{2, 3, 4}, LCW{9, 8}}});
  FCW b({100, 200, 300, 400, 500});

  LCW expected_a{
    LCW{0, 3}, LCW{5}, LCW({2, 1}, valids), LCW{8, 3}, LCW{}, LCW({4, 3, 1, 2}, valids)};
  FCW expected_b{300, 300, 300, 400, 400, 400};

  cudf::table_view t({a, b});
  auto sliced_t = cudf::slice(t, {2, 4});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(sliced_t[0], 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(sliced_t[0], 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}
