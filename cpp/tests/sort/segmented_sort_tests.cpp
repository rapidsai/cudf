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
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <type_traits>
#include <vector>

#include <cudf/lists/sorting.hpp>

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;
using cudf::lists_column_view;
using cudf::segmented_sort;

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
  auto results = segmented_sort(
    input, {order::ASCENDING, order::ASCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1);

  results = segmented_sort(
    input, {order::ASCENDING, order::ASCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1);

  // Descending
  // LCW<int>  order{{3, 5, 4, 0, 1, 2}, {0}, {0, 2, 1},  {1, 0}};
  LCW<T> expected3{{4, 4, 4, 3, 2, 1}, {5}, {9, 9, 8}, {7, 6}};
  LCW<T> expected4{{3, 2, 1, 3, 1, 2}, {0}, {10, 9, 9}, {7, 6}};
  table_view expected_table2{{expected3, expected4}};
  results = segmented_sort(
    input, {order::DESCENDING, order::DESCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2);

  results = segmented_sort(
    input, {order::DESCENDING, order::DESCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2);
}

}  // namespace test
}  // namespace cudf
