/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

using int32s_lists = cudf::test::lists_column_wrapper<int32_t>;
using int32s_col   = cudf::test::fixed_width_column_wrapper<int32_t>;
using strings_col  = cudf::test::strings_column_wrapper;
using structs_col  = cudf::test::structs_column_wrapper;

using namespace cudf::test::iterators;

constexpr auto null{0};

struct NestedStructTest : public cudf::test::BaseFixture {
};

TEST_F(NestedStructTest, SimpleStructsOfListsNoNulls)
{
  auto const input = [] {
    auto child = int32s_lists{{4, 2, 0}, {2}, {0, 5}, {1, 5}, {4, 1}};
    return structs_col{{child}};
  }();

  {
    auto const expected_order = int32s_col{2, 3, 1, 4, 0};
    auto const order          = cudf::sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{0, 4, 1, 3, 2};
    auto const order = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, SimpleStructsOfListsWithNulls)
{
  auto const input = [] {
    auto child =
      int32s_lists{{{4, 2, null}, null_at(2)}, {2}, {{null, 5}, null_at(0)}, {0, 5}, {4, 1}};
    return structs_col{{child}};
  }();

  {
    auto const expected_order = int32s_col{2, 3, 1, 4, 0};
    auto const order          = cudf::sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{0, 4, 1, 3, 2};
    auto const order = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsHaveListsNoNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 = int32s_lists{{4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child1 = int32s_col{1, 2, 5, 0, 3, 3, 4};
    return structs_col{{child0, child1}};
  }();

  {
    auto const expected_order = int32s_col{1, 5, 6, 4, 3, 0, 2};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsHaveListsWithNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 =
      int32s_lists{{{4, 2, null}, null_at(2)}, {}, {} /*NULL*/, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child1 = int32s_col{{1, 2, null, 5, null, 3, 3, 4}, nulls_at({2, 4})};
    return structs_col{{child0, child1}, null_at(2)};
  }();

  {
    auto const expected_order = int32s_col{2, 1, 6, 7, 5, 4, 0, 3};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, SlicedStructsHaveListsNoNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  // The original input has 3 first elements repeated at the beginning and the end.
  auto const input_original = [] {
    auto child0 = int32s_lists{
      {4, 2, 0}, {}, {5}, {4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}, {4, 2, 0}, {}, {5}};
    auto child1 = int32s_col{1, 2, 5, 1, 2, 5, 0, 3, 3, 4, 1, 2, 5};
    return structs_col{{child0, child1}};
  }();

  auto const input = cudf::slice(input_original, {3, 10})[0];

  {
    auto const expected_order = int32s_col{1, 5, 6, 4, 3, 0, 2};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, SlicedStructsHaveListsWithNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  // The original input has 2 first elements repeated at the beginning and the end.
  auto const input_original = [] {
    auto child0 = int32s_lists{{{4, 2, null}, null_at(2)},
                               {},
                               {{4, 2, null}, null_at(2)},
                               {},
                               {} /*NULL*/,
                               {5},
                               {4, 1},
                               {4, 0},
                               {},
                               {},
                               {{4, 2, null}, null_at(2)},
                               {}};
    auto child1 = int32s_col{{1, 2, 1, 2, null, 5, null, 3, 3, 4, 1, 2}, nulls_at({4, 6})};
    return structs_col{{child0, child1}, null_at(4)};
  }();

  auto const input = cudf::slice(input_original, {2, 10})[0];

  {
    auto const expected_order = int32s_col{2, 1, 6, 7, 5, 4, 0, 3};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsOfStructsHaveListsNoNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 = [] {
      auto child0 = int32s_lists{{4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}};
      auto child1 = int32s_col{1, 2, 5, 0, 3, 3, 4};
      return structs_col{{child0, child1}};
    }();
    auto child1 = int32s_lists{{4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child2 = int32s_col{1, 2, 5, 0, 3, 3, 4};
    return structs_col{{child0, child1, child2}};
  }();

  {
    auto const expected_order = int32s_col{1, 5, 6, 4, 3, 0, 2};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsOfStructsHaveListsWithNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 = [] {
      auto child0 =
        int32s_lists{{{4, 2, null}, null_at(2)}, {}, {} /*NULL*/, {5}, {4, 1}, {4, 0}, {}, {}};
      auto child1 = int32s_col{{1, 2, null, 5, null, 3, 3, 4}, nulls_at({2, 4})};
      return structs_col{{child0, child1}, null_at(2)};
    }();
    auto child1 =
      int32s_lists{{{4, 2, null}, null_at(2)}, {}, {} /*NULL*/, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child2 = int32s_col{{1, 2, null, 5, null, 3, 3, 4}, nulls_at({2, 4})};
    return structs_col{{child0, child1, child2}, null_at(2)};
  }();

  {
    auto const expected_order = int32s_col{2, 1, 6, 7, 5, 4, 0, 3};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}
