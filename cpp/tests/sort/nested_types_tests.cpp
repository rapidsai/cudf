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

#include <cudf/sorting.hpp>

using int32s_lists = cudf::test::lists_column_wrapper<int32_t>;
using int32s_col   = cudf::test::fixed_width_column_wrapper<int32_t>;
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
    auto const expected_sort  = [] {
      auto child = int32s_lists{{0, 5}, {1, 5}, {2}, {4, 1}, {4, 2, 0}};
      return structs_col{{child}};
    }();

    auto const order  = cudf::sorted_order(cudf::table_view{{input}});
    auto const sorted = cudf::sort(cudf::table_view{{input}});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }

  {
    auto const expected_order = int32s_col{0, 4, 1, 3, 2};
    auto const expected_sort  = [] {
      auto child = int32s_lists{{4, 2, 0}, {4, 1}, {2}, {1, 5}, {0, 5}};
      return structs_col{{child}};
    }();
    auto const order  = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    auto const sorted = cudf::sort(cudf::table_view{{input}}, {cudf::order::DESCENDING});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
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
    auto const expected_sort  = [] {
      auto child =
        int32s_lists{{{null, 5}, null_at(0)}, {0, 5}, {2}, {4, 1}, {{4, 2, null}, null_at(2)}};
      return structs_col{{child}};
    }();

    auto const order  = cudf::sorted_order(cudf::table_view{{input}});
    auto const sorted = cudf::sort(cudf::table_view{{input}});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }

  {
    auto const expected_order = int32s_col{0, 4, 1, 3, 2};
    auto const expected_sort  = [] {
      auto child =
        int32s_lists{{{4, 2, null}, null_at(2)}, {4, 1}, {2}, {0, 5}, {{null, 5}, null_at(0)}};
      return structs_col{{child}};
    }();
    auto const order  = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    auto const sorted = cudf::sort(cudf::table_view{{input}}, {cudf::order::DESCENDING});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }
}

TEST_F(NestedStructTest, StructsHaveListsNoNulls)
{
  auto const input = [] {
    auto child0 = int32s_lists{{4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child1 = int32s_col{1, 2, 5, 0, 3, 3, 4};
    return structs_col{{child0, child1}};
  }();

  {
    auto const expected_order = int32s_col{1, 5, 6, 4, 3, 0, 2};
    auto const expected_sort  = [] {
      auto child0 = int32s_lists{{}, {}, {}, {4, 0}, {4, 1}, {4, 2, 0}, {5}};
      auto child1 = int32s_col{2, 3, 4, 3, 0, 1, 5};
      return structs_col{{child0, child1}};
    }();

    auto const order  = cudf::sorted_order(cudf::table_view{{input}});
    auto const sorted = cudf::sort(cudf::table_view{{input}});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const expected_sort  = [] {
      auto child0 = int32s_lists{{5}, {4, 2, 0}, {4, 1}, {4, 0}, {}, {}, {}};
      auto child1 = int32s_col{5, 1, 0, 3, 4, 3, 2};
      return structs_col{{child0, child1}};
    }();
    auto const order  = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    auto const sorted = cudf::sort(cudf::table_view{{input}}, {cudf::order::DESCENDING});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }
}

TEST_F(NestedStructTest, StructsHaveListsWithNulls)
{
  auto const input = [] {
    auto child0 =
      int32s_lists{{{4, 2, null}, null_at(2)}, {}, {} /*NULL*/, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child1 = int32s_col{{1, 2, null, 5, null, 3, 3, 4}, nulls_at({2, 4})};
    return structs_col{{child0, child1}, null_at(2)};
  }();

  {
    auto const expected_order = int32s_col{2, 1, 6, 7, 5, 4, 0, 3};
    auto const expected_sort  = [] {
      auto child0 =
        int32s_lists{{} /*NULL*/, {}, {}, {}, {4, 0}, {4, 1}, {{4, 2, null}, null_at(2)}, {5}};
      auto child1 = int32s_col{{null, 2, 3, 4, 3, null, 1, 5}, nulls_at({0, 5})};
      return structs_col{{child0, child1}, null_at(0)};
    }();

    auto const order  = cudf::sorted_order(cudf::table_view{{input}});
    auto const sorted = cudf::sort(cudf::table_view{{input}});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const expected_sort  = [] {
      auto child0 =
        int32s_lists{{5}, {{4, 2, null}, null_at(2)}, {4, 1}, {4, 0}, {}, {}, {}, {} /*NULL*/};
      auto child1 = int32s_col{{5, 1, null, 3, 4, 3, 2, null}, nulls_at({2, 7})};
      return structs_col{{child0, child1}, null_at(7)};
    }();
    auto const order  = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    auto const sorted = cudf::sort(cudf::table_view{{input}}, {cudf::order::DESCENDING});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }
}

TEST_F(NestedStructTest, StructsOfStructsHaveListsNoNulls)
{
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
    auto const expected_sort  = [] {
      auto child0 = [] {
        auto child0 = int32s_lists{{}, {}, {}, {4, 0}, {4, 1}, {4, 2, 0}, {5}};
        auto child1 = int32s_col{2, 3, 4, 3, 0, 1, 5};
        return structs_col{{child0, child1}};
      }();
      auto child1 = int32s_lists{{}, {}, {}, {4, 0}, {4, 1}, {4, 2, 0}, {5}};
      auto child2 = int32s_col{2, 3, 4, 3, 0, 1, 5};
      return structs_col{{child0, child1, child2}};
    }();

    auto const order  = cudf::sorted_order(cudf::table_view{{input}});
    auto const sorted = cudf::sort(cudf::table_view{{input}});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const expected_sort  = [] {
      auto child0 = [] {
        auto child0 = int32s_lists{{5}, {4, 2, 0}, {4, 1}, {4, 0}, {}, {}, {}};
        auto child1 = int32s_col{5, 1, 0, 3, 4, 3, 2};
        return structs_col{{child0, child1}};
      }();
      auto child1 = int32s_lists{{5}, {4, 2, 0}, {4, 1}, {4, 0}, {}, {}, {}};
      auto child2 = int32s_col{5, 1, 0, 3, 4, 3, 2};
      return structs_col{{child0, child1, child2}};
    }();
    auto const order  = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    auto const sorted = cudf::sort(cudf::table_view{{input}}, {cudf::order::DESCENDING});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }
}

TEST_F(NestedStructTest, StructsOfStructsHaveListsWithNulls)
{
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
    auto const expected_sort  = [] {
      auto child0 = [] {
        auto child0 =
          int32s_lists{{} /*NULL*/, {}, {}, {}, {4, 0}, {4, 1}, {{4, 2, null}, null_at(2)}, {5}};
        auto child1 = int32s_col{{null, 2, 3, 4, 3, null, 1, 5}, nulls_at({0, 5})};
        return structs_col{{child0, child1}, null_at(0)};
      }();
      auto child1 =
        int32s_lists{{} /*NULL*/, {}, {}, {}, {4, 0}, {4, 1}, {{4, 2, null}, null_at(2)}, {5}};
      auto child2 = int32s_col{{null, 2, 3, 4, 3, null, 1, 5}, nulls_at({0, 5})};
      return structs_col{{child0, child1, child2}, null_at(0)};
    }();

    auto const order  = cudf::sorted_order(cudf::table_view{{input}});
    auto const sorted = cudf::sort(cudf::table_view{{input}});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const expected_sort  = [] {
      auto child0 = [] {
        auto child0 =
          int32s_lists{{5}, {{4, 2, null}, null_at(2)}, {4, 1}, {4, 0}, {}, {}, {}, {} /*NULL*/};
        auto child1 = int32s_col{{5, 1, null, 3, 4, 3, 2, null}, nulls_at({2, 7})};
        return structs_col{{child0, child1}, null_at(7)};
      }();
      auto child1 =
        int32s_lists{{5}, {{4, 2, null}, null_at(2)}, {4, 1}, {4, 0}, {}, {}, {}, {} /*NULL*/};
      auto child2 = int32s_col{{5, 1, null, 3, 4, 3, 2, null}, nulls_at({2, 7})};
      return structs_col{{child0, child1, child2}, null_at(7)};
    }();
    auto const order  = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    auto const sorted = cudf::sort(cudf::table_view{{input}}, {cudf::order::DESCENDING});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sort, sorted->get_column(0).view());
  }
}

struct NestedListTest : public cudf::test::BaseFixture {
};

TEST_F(NestedListTest, SimpleListsOfStructs)
{
  auto const input = [] {
    auto const get_structs = [] {
      auto child1 = int32s_col{1, 2, 3, 0, 1, 2, 4, 5, 0, 1};
      //      auto child2 = int32s_col{11, 12, 13, 10, 11, 12, 14, 15, 10, 11};
      return structs_col{{child1 /*, child2*/}};
    };

    return cudf::make_lists_column(
      4, int32s_col{0, 3, 6, 8, 10}.release(), get_structs().release(), 0, {});
  }();

  printf("line %d\n", __LINE__);
  cudf::test::print(*input);

  if (1) {
    auto const order = cudf::sorted_order(cudf::table_view{{*input}});

    printf("line %d\n", __LINE__);
    cudf::test::print(*order);

    auto const sorted = cudf::sort(cudf::table_view{{*input}});

    printf("line %d\n", __LINE__);
    cudf::test::print(sorted->get_column(0).view());
  }

  {
    auto const order = cudf::sorted_order(cudf::table_view{{*input}}, {cudf::order::DESCENDING});

    printf("line %d\n", __LINE__);
    cudf::test::print(*order);

    auto const sorted = cudf::sort(cudf::table_view{{*input}}, {cudf::order::DESCENDING});
    printf("line %d\n", __LINE__);
    cudf::test::print(sorted->get_column(0).view());
  }
}
