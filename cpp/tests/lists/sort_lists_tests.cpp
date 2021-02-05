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
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <exception>
#include <type_traits>
#include <vector>

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;
using cudf::lists_column_view;
using cudf::lists::sort_lists;

namespace cudf {
namespace test {

template <typename T>
struct SortLists : public BaseFixture {
};

TYPED_TEST_CASE(SortLists, NumericTypes);
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
  auto results = sort_lists(lists_column_view{list}, order::ASCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  results = sort_lists(lists_column_view{list}, order::ASCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected2{{4, 3, 2, 1}, {5}, {10, 9, 8}, {7, 6}};
  results = sort_lists(lists_column_view{list}, order::DESCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);

  results = sort_lists(lists_column_view{list}, order::DESCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);
}

TYPED_TEST(SortLists, Null)
{
  using T = TypeParam;
  if (std::is_same<T, bool>::value) return;
  std::vector<bool> valids_o{1, 1, 0, 1};
  std::vector<bool> valids_a{1, 1, 1, 0};
  std::vector<bool> valids_b{0, 1, 1, 1};

  // List<T>
  LCW<T> list{{{3, 2, 4, 1}, valids_o.begin()}, {5}, {10, 8, 9}, {6, 7}};
  // LCW<int>  order{{2, 1, 3, 0}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected1{{{1, 2, 3, 4}, valids_a.begin()}, {5}, {8, 9, 10}, {6, 7}};
  LCW<T> expected2{{{4, 1, 2, 3}, valids_b.begin()}, {5}, {8, 9, 10}, {6, 7}};
  auto results = sort_lists(lists_column_view{list}, order::ASCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected1);

  results = sort_lists(lists_column_view{list}, order::ASCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected3{{{4, 3, 2, 1}, valids_b.begin()}, {5}, {10, 9, 8}, {7, 6}};
  LCW<T> expected4{{{3, 2, 1, 4}, valids_a.begin()}, {5}, {10, 9, 8}, {7, 6}};
  results = sort_lists(lists_column_view{list}, order::DESCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected3);

  results = sort_lists(lists_column_view{list}, order::DESCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected4);
}

TEST_F(SortListsInt, Empty)
{
  using T = int;
  LCW<T> l1{};
  LCW<T> l2{LCW<T>{}};
  LCW<T> l3{LCW<T>{}, LCW<T>{}};

  auto results = sort_lists(lists_column_view{l1}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), l1);
  results = sort_lists(lists_column_view{l2}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), l2);
  results = sort_lists(lists_column_view{l3}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), l3);
}

TEST_F(SortListsInt, Single)
{
  using T = int;
  LCW<T> l1{{1}};
  LCW<T> l2{{1, 2, 3}};

  auto results = sort_lists(lists_column_view{l1}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), l1);
  results = sort_lists(lists_column_view{l2}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), l2);
}

TEST_F(SortListsInt, NullRows)
{
  using T = int;
  std::vector<int> valids{0, 1, 0};
  LCW<T> l1{{{1, 2, 3}, {4, 5, 6}, {7}}, valids.begin()};  // offset 0, 0, 3, 3

  auto results = sort_lists(lists_column_view{l1}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), l1);
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
  LCW<T> l1{{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}};
  auto sliced_list = cudf::slice(l1, {1, 4})[0];

  auto results = sort_lists(lists_column_view{sliced_list}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), sliced_list);
}

}  // namespace test
}  // namespace cudf
