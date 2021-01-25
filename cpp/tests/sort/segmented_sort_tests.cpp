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
//#include <cudf/detail/sorting.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <type_traits>
#include <vector>

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

namespace cudf {
namespace test {

template <typename T>
struct SegmentedSort : public BaseFixture {
};

// using NumericTypesNotBool = Concat<IntegralTypesNotBool, FloatingPointTypes>;
TYPED_TEST_CASE(SegmentedSort, NumericTypes);

TYPED_TEST(SegmentedSort, NoNull)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list1{{3, 2, 1, 4, 4, 4}, {5}, {9, 8, 9}, {6, 7}};
  LCW<T> list2{{3, 1, 2, 3, 1, 2}, {0}, {10, 9, 9}, {6, 7}};
  table_view input{{list1, list2}};

  // Ascending
  // LCW<int>  order{{2, 1, 0, 4, 5, 3}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected1{{1, 2, 3, 4, 4, 4}, {5}, {8, 9, 9}, {6, 7}};
  LCW<T> expected2{{2, 1, 3, 1, 2, 3}, {0}, {9, 9, 10}, {6, 7}};
  table_view expected_table1{{expected1, expected2}};
  auto results = cudf::segmented_sort(
    input, {order::ASCENDING, order::ASCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1);

  results = cudf::segmented_sort(
    input, {order::ASCENDING, order::ASCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1);

  // Descending
  // LCW<int>  order{{3, 5, 4, 0, 1, 2}, {0}, {0, 2, 1},  {1, 0}};
  LCW<T> expected3{{4, 4, 4, 3, 2, 1}, {5}, {9, 9, 8}, {7, 6}};
  LCW<T> expected4{{3, 2, 1, 3, 1, 2}, {0}, {10, 9, 9}, {7, 6}};
  table_view expected_table2{{expected3, expected4}};
  results = cudf::segmented_sort(
    input, {order::DESCENDING, order::DESCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2);

  results = cudf::segmented_sort(
    input, {order::DESCENDING, order::DESCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2);
}

TYPED_TEST(SegmentedSort, Nulls)
{
  using T = TypeParam;
  if (std::is_same<T, bool>::value) return;

  // List<T>
  std::vector<bool> valids1{1, 1, 1, 0, 1, 1};
  std::vector<bool> valids1a{1, 1, 1, 1, 1, 0};
  std::vector<bool> valids2{1, 1, 0};
  std::vector<bool> valids2b{1, 0, 1};
  LCW<T> list1{{{3, 2, 1, 4, 4, 4}, valids1.begin()}, {5}, {9, 8, 9}, {6, 7}};
  LCW<T> list2{{3, 1, 2, 2, 1, 3}, {0}, {{10, 9, 9}, valids2.begin()}, {6, 7}};
  table_view input{{list1, list2}};
  // nulls = (4-NULL, 2), (9,9-NULL)
  //  (8,9), (9,10), (9,N)

  // Ascending
  // LCW<int>  order{{2, 1, 0, 4, 5, 3}, {0}, {1, 0, 2},  {0, 1}};
  LCW<T> expected1a{{{1, 2, 3, 4, 4, 4}, valids1a.begin()}, {5}, {8, 9, 9}, {6, 7}};
  LCW<T> expected2a{{2, 1, 3, 1, 3, 2}, {0}, {{9, 10, 9}, valids2.begin()}, {6, 7}};
  table_view expected_table1a{{expected1a, expected2a}};
  auto results = cudf::segmented_sort(
    input, {order::ASCENDING, order::ASCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1a);

  // LCW<int>  order{{3, 2, 1, 0, 4, 5}, {0}, {2, 1, 0},  {0, 1}};
  LCW<T> expected1b{{{4, 1, 2, 3, 4, 4}, valids1a.rbegin()}, {5}, {8, 9, 9}, {6, 7}};
  LCW<T> expected2b{{2, 2, 1, 3, 1, 3}, {0}, {{9, 9, 10}, valids2b.begin()}, {6, 7}};
  table_view expected_table1b{{expected1b, expected2b}};
  results = cudf::segmented_sort(
    input, {order::ASCENDING, order::ASCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1b);

  // Descending
  LCW<T> expected3a{{{4, 4, 4, 3, 2, 1}, valids1a.rbegin()}, {5}, {9, 9, 8}, {7, 6}};
  LCW<T> expected4a{{2, 3, 1, 3, 1, 2}, {0}, {{9, 10, 9}, valids2.rbegin()}, {7, 6}};
  table_view expected_table2a{{expected3a, expected4a}};
  results = cudf::segmented_sort(
    input, {order::DESCENDING, order::DESCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2a);

  LCW<T> expected3b{{{4, 4, 3, 2, 1, 4}, valids1a.begin()}, {5}, {9, 9, 8}, {7, 6}};
  LCW<T> expected4b{{3, 1, 3, 1, 2, 2}, {0}, {{10, 9, 9}, valids2b.begin()}, {7, 6}};
  table_view expected_table2b{{expected3b, expected4b}};
  results = cudf::segmented_sort(
    input, {order::DESCENDING, order::DESCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2b);
}

}  // namespace test
}  // namespace cudf
