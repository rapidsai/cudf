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
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <type_traits>
#include <vector>

#include <cudf/lists/sorting.hpp>

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;
using cudf::lists_column_view;
using cudf::lists::segmented_sort;

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
  LCW<T> list{{3, 2, 1, 4}, {5}, {10, 8, 9}, {6, 7}};

  // Ascending
  // LCW<int>  order{{2, 1, 0, 3}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  auto results = segmented_sort(lists_column_view{list}, order::ASCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  results = segmented_sort(lists_column_view{list}, order::ASCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected2{{4, 3, 2, 1}, {5}, {10, 9, 8}, {7, 6}};
  results = segmented_sort(lists_column_view{list}, order::DESCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);

  results = segmented_sort(lists_column_view{list}, order::DESCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);
}

TYPED_TEST(SegmentedSort, Null)
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
  auto results = segmented_sort(lists_column_view{list}, order::ASCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected1);

  results = segmented_sort(lists_column_view{list}, order::ASCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected3{{{3, 2, 1, 4}, valids_a.begin()}, {5}, {10, 9, 8}, {7, 6}};
  LCW<T> expected4{{{4, 3, 2, 1}, valids_b.begin()}, {5}, {10, 9, 8}, {7, 6}};
  results = segmented_sort(lists_column_view{list}, order::DESCENDING, null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected3);

  results = segmented_sort(lists_column_view{list}, order::DESCENDING, null_order::BEFORE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected4);
}

}  // namespace test
}  // namespace cudf
