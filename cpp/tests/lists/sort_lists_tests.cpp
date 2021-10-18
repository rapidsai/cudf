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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/sorting.hpp>

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

namespace cudf {
namespace test {

auto generate_sorted_lists(lists_column_view const& input,
                           order column_order,
                           null_order null_precedence)
{
  return std::pair{lists::sort_lists(input, column_order, null_precedence),
                   lists::stable_sort_lists(input, column_order, null_precedence)};
}

template <typename T>
struct SortLists : public BaseFixture {
};

TYPED_TEST_SUITE(SortLists, NumericTypes);
using SortListsInt = SortLists<int>;

/*
empty case
  empty list
  single row with empty list
  multi row with empty lists
single case
  single list with single element
  single list with multi element
normal case without nulls
Null cases
  null rows
  null elements in list.
Error:
  depth>1
*/
TYPED_TEST(SortLists, NoNull)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{3, 2, 1, 4}, {5}, {10, 8, 9}, {6, 7}};

  // Ascending
  // LCW<int>  order{{2, 1, 0, 3}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  {
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::ASCENDING, null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::ASCENDING, null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected2{{4, 3, 2, 1}, {5}, {10, 9, 8}, {7, 6}};
  {
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::DESCENDING, null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::DESCENDING, null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
}

TYPED_TEST(SortLists, Null)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) return;
  std::vector<bool> valids_o{1, 1, 0, 1};
  std::vector<bool> valids_a{1, 1, 1, 0};
  std::vector<bool> valids_b{0, 1, 1, 1};

  // List<T>
  LCW<T> list{{{3, 2, 4, 1}, valids_o.begin()}, {5}, {10, 8, 9}, {6, 7}};
  // LCW<int>  order{{2, 1, 3, 0}, {0}, {1, 2, 0},  {0, 1}};

  {
    LCW<T> expected{{{1, 2, 3, 4}, valids_a.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::ASCENDING, null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{4, 1, 2, 3}, valids_b.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::ASCENDING, null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  {
    LCW<T> expected{{{4, 3, 2, 1}, valids_b.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::DESCENDING, null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{3, 2, 1, 4}, valids_a.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{list}, order::DESCENDING, null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

TEST_F(SortListsInt, Empty)
{
  using T = int;

  {
    LCW<T> l{};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}, LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, Single)
{
  using T = int;

  {
    LCW<T> l{1};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{{1, 2, 3}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, NullRows)
{
  using T = int;
  std::vector<int> valids{0, 1, 0};
  LCW<T> l{{{1, 2, 3}, {4, 5, 6}, {7}}, valids.begin()};  // offset 0, 0, 3, 3

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(lists_column_view{l}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
}

/*
// Disabling this test.
// Reason: After this exception "cudaErrorAssert device-side assert triggered", further tests fail
TEST_F(SortListsInt, Depth)
{
  using T = int;
  LCW<T> l1{LCW<T>{{1, 2}, {3}}, LCW<T>{{4, 5}}};
  // device exception
  EXPECT_THROW(sort_lists(lists_column_view{l1}, {}, {}), std::exception);
}
*/

TEST_F(SortListsInt, Sliced)
{
  using T = int;
  LCW<T> l{{3, 2, 1, 4}, {7, 5, 6}, {8, 9}, {10}};

  {
    auto const sliced_list = cudf::slice(l, {0, 4})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 4})[0];
    auto const expected    = LCW<T>{{5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 2})[0];
    auto const expected    = LCW<T>{{5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {0, 2})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

}  // namespace test
}  // namespace cudf
