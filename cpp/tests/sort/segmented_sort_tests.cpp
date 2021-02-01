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
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <type_traits>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using LCWstr         = cudf::test::lists_column_wrapper<cudf::string_view>;

namespace cudf {
namespace test {

template <typename T>
struct SegmentedSort : public BaseFixture {
};

TYPED_TEST_CASE(SegmentedSort, NumericTypes);
using SegmentedSortInt = SegmentedSort<int>;

/*
normal case
{8, 9, 2, 3, 2, 2, 4, 1, 7, 5, 6}
{0,    2,       5,       8       11}
  without null
  with null
empty case
  key{},
  value{},
  segment_offset{}
single case
  keys{1}, value{1}
  segmented_offset{0}, {0, 1}
corner case
  sliced table,
  sliced segment_offsets
  non-zero start of segment_offsets without offset
  non-zero start of segment_offsets with offset
mismatch sizes
  keys, values num_rows
  order, null_order
  segmented_offsets beyond num_rows
//*/

TYPED_TEST(SegmentedSort, NoNull)
{
  using T = TypeParam;

  // segments                 {0   1   2} {3   4} {5} {6   7   8   9  10}{11  12}{13}{14  15}
  column_wrapper<T> col1{{10, 36, 14, 32, 49, 23, 10, 34, 12, 45, 12, 37, 43, 26, 21, 16}};
  column_wrapper<T> col2{{10, 63, 41, 23, 94, 32, 10, 43, 21, 54, 22, 73, 34, 62, 12, 61}};
  // segment sorted order     {0   2   1} {3   4} {5}  {6   8  10   7  9}{11  12}{13}{15  16}
  column_wrapper<int> segments{0, 3, 5, 5, 5, 6, 11, 13, 14, 16};
  table_view input1{{col1}};
  table_view input2{{col1, col2}};

  // Ascending
  column_wrapper<T> col1_asc{{10, 14, 36, 32, 49, 23, 10, 12, 12, 34, 45, 37, 43, 26, 16, 21}};

  auto results = cudf::segmented_sort(input1, input1, segments, {order::ASCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), table_view{{col1_asc}});

  column_wrapper<T> col1_des{{36, 14, 10, 49, 32, 23, 45, 34, 12, 12, 10, 43, 37, 26, 21, 16}};
  results = cudf::segmented_sort(input1, input1, segments, {order::DESCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), table_view{{col1_des}});

  column_wrapper<T> col1_12_asc{{10, 14, 36, 32, 49, 23, 10, 12, 12, 34, 45, 37, 43, 26, 16, 21}};
  column_wrapper<T> col2_12_asc{{10, 41, 63, 23, 94, 32, 10, 21, 22, 43, 54, 73, 34, 62, 61, 12}};
  column_wrapper<T> col2_12_des{{10, 41, 63, 23, 94, 32, 10, 22, 21, 43, 54, 73, 34, 62, 61, 12}};

  table_view expected12_aa{{col1_12_asc, col2_12_asc}};
  results = cudf::segmented_sort(input2, input2, segments, {});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_aa);

  table_view expected12_ad{{col1_12_asc, col2_12_des}};
  results = cudf::segmented_sort(input2, input2, segments, {order::ASCENDING, order::DESCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_ad);
}

TYPED_TEST(SegmentedSort, Null)
{
  using T = TypeParam;
  if (std::is_same<T, bool>::value) return;

  // segments                 {0   1   2} {3   4} {5} {6   7   8   9  10}{11  12}{13}{14  15}
  column_wrapper<T> col1{{1, 3, 2, 4, 5, 23, 6, 8, 7, 9, 7, 37, 43, 26, 21, 16},
                         {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1}};
  column_wrapper<T> col2{{0, 0, 0, 1, 1, 4, 5, 5, 21, 5, 22, 6, 6, 7, 8, 8},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}};
  column_wrapper<int> segments{0, 3, 5, 5, 5, 6, 11, 13, 14, 16};
  table_view input1{{col1}};
  table_view input2{{col1, col2}};

  // Ascending
  column_wrapper<T> col1_aa{{1, 3, 2, 4, 5, 23, 6, 7, 7, 8, 9, 37, 43, 26, 16, 21},
                            {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1}};
  column_wrapper<T> col1_ab{{2, 1, 3, 4, 5, 23, 9, 6, 7, 7, 8, 37, 43, 26, 16, 21},
                            {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto results = cudf::segmented_sort(input1, input1, segments, {}, {null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), table_view{{col1_aa}});
  results = cudf::segmented_sort(input1, input1, segments, {}, {null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), table_view{{col1_ab}});

  // Descending
  column_wrapper<T> col1_da{{2, 3, 1, 5, 4, 23, 9, 8, 7, 7, 6, 43, 37, 26, 21, 16},
                            {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  column_wrapper<T> col1_db{{3, 1, 2, 5, 4, 23, 8, 7, 7, 6, 9, 43, 37, 26, 21, 16},
                            {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1}};
  results =
    cudf::segmented_sort(input1, input1, segments, {order::DESCENDING}, {null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), table_view{{col1_da}});
  results =
    cudf::segmented_sort(input1, input1, segments, {order::DESCENDING}, {null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), table_view{{col1_db}});

  // second row null order.
  column_wrapper<T> col2_12_aa{{0, 0, 0, 1, 1, 4, 5, 22, 21, 5, 5, 6, 6, 7, 8, 8},
                               {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}};
  column_wrapper<T> col2_12_ab{{0, 0, 0, 1, 1, 4, 5, 5, 21, 22, 5, 6, 6, 7, 8, 8},
                               {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}};
  table_view expected12_aa{{col1_aa, col2_12_aa}};
  table_view expected12_ab{{col1_ab, col2_12_ab}};
  results =
    cudf::segmented_sort(input2, input2, segments, {}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_aa);
  results =
    cudf::segmented_sort(input2, input2, segments, {}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_ab);
}

}  // namespace test
}  // namespace cudf
