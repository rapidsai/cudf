/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include <type_traits>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T, int>;

template <typename T>
struct SegmentedSort : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SegmentedSort, cudf::test::NumericTypes);
using SegmentedSortInt = SegmentedSort<int>;

TEST_F(SegmentedSortInt, Empty)
{
  using T = int;
  column_wrapper<T> col_empty{};
  // clang-format off
  column_wrapper<T>       col1{{8, 9, 2, 3, 2, 2, 4, 1, 7, 5, 6}};
  column_wrapper<int> segments{{0,    2,       5,       8,      11}};
  // clang-format on
  cudf::table_view table_empty{{col_empty}};
  cudf::table_view table_valid{{col1}};

  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(table_valid, table_valid, segments));
  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(table_valid, table_valid, col_empty));
  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(table_empty, table_empty, segments));
  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(table_empty, table_empty, col_empty));

  // Swapping "empty" and "valid" tables is invalid because the keys and values will be of different
  // sizes.
  EXPECT_THROW(cudf::segmented_sort_by_key(table_empty, table_valid, segments), cudf::logic_error);
  EXPECT_THROW(cudf::segmented_sort_by_key(table_empty, table_valid, col_empty), cudf::logic_error);
  EXPECT_THROW(cudf::segmented_sort_by_key(table_valid, table_empty, segments), cudf::logic_error);
  EXPECT_THROW(cudf::segmented_sort_by_key(table_valid, table_empty, col_empty), cudf::logic_error);
}

TEST_F(SegmentedSortInt, Single)
{
  using T = int;
  column_wrapper<T> col1{{1}};
  column_wrapper<T> col3{{8, 9, 2}};
  column_wrapper<int> segments1{{0}};
  column_wrapper<int> segments2{{0, 3}};
  cudf::table_view table_1elem{{col1}};
  cudf::table_view table_1segm{{col3}};

  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(table_1elem, table_1elem, segments1));
  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(table_1segm, table_1segm, segments2));
  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(table_1segm, table_1segm, segments1));
}

TYPED_TEST(SegmentedSort, NoNull)
{
  using T = TypeParam;

  // segments             {0   1   2} {3   4} {5} {6   7   8   9  10}{11  12}{13}{14  15}
  column_wrapper<T> col1{{10, 36, 14, 32, 49, 23, 10, 34, 12, 45, 12, 37, 43, 26, 21, 16}};
  column_wrapper<T> col2{{10, 63, 41, 23, 94, 32, 10, 43, 21, 54, 22, 73, 34, 62, 12, 61}};
  // segment sorted order {0   2   1} {3   4} {5}  {6   8  10   7  9}{11  12}{13}{15  16}
  column_wrapper<int> segments{0, 3, 5, 5, 5, 6, 11, 13, 14, 16};
  cudf::table_view input1{{col1}};
  cudf::table_view input2{{col1, col2}};

  // Ascending
  column_wrapper<T> col1_asc{{10, 14, 36, 32, 49, 23, 10, 12, 12, 34, 45, 37, 43, 26, 16, 21}};

  auto results = cudf::segmented_sort_by_key(input1, input1, segments, {cudf::order::ASCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col1_asc}});

  column_wrapper<T> col1_des{{36, 14, 10, 49, 32, 23, 45, 34, 12, 12, 10, 43, 37, 26, 21, 16}};
  results = cudf::segmented_sort_by_key(input1, input1, segments, {cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col1_des}});

  column_wrapper<T> col1_12_asc{{10, 14, 36, 32, 49, 23, 10, 12, 12, 34, 45, 37, 43, 26, 16, 21}};
  column_wrapper<T> col2_12_asc{{10, 41, 63, 23, 94, 32, 10, 21, 22, 43, 54, 73, 34, 62, 61, 12}};
  column_wrapper<T> col2_12_des{{10, 41, 63, 23, 94, 32, 10, 22, 21, 43, 54, 73, 34, 62, 61, 12}};

  cudf::table_view expected12_aa{{col1_12_asc, col2_12_asc}};
  results = cudf::segmented_sort_by_key(input2, input2, segments, {});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_aa);

  cudf::table_view expected12_ad{{col1_12_asc, col2_12_des}};
  results = cudf::segmented_sort_by_key(
    input2, input2, segments, {cudf::order::ASCENDING, cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_ad);
}

TYPED_TEST(SegmentedSort, Null)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) return;

  // segments            {0  1  2}{3  4} {5}{6  7  8  9 10}{11  12}{13}{14  15}
  column_wrapper<T> col1{{1, 3, 2, 4, 5, 23, 6, 8, 7, 9, 7, 37, 43, 26, 21, 16},
                         {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1}};
  column_wrapper<T> col2{{0, 0, 0, 1, 1, 4, 5, 5, 21, 5, 22, 6, 6, 7, 8, 8},
                         {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}};
  column_wrapper<int> segments{0, 3, 5, 5, 5, 6, 11, 13, 14, 16};
  cudf::table_view input1{{col1}};
  cudf::table_view input2{{col1, col2}};

  // Ascending
  column_wrapper<T> col1_aa{{1, 3, 2, 4, 5, 23, 6, 7, 7, 8, 9, 37, 43, 26, 16, 21},
                            {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1}};
  column_wrapper<T> col1_ab{{2, 1, 3, 4, 5, 23, 9, 6, 7, 7, 8, 37, 43, 26, 16, 21},
                            {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto results =
    cudf::segmented_sort_by_key(input1, input1, segments, {}, {cudf::null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col1_aa}});
  results = cudf::segmented_sort_by_key(input1, input1, segments, {}, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col1_ab}});

  // Descending
  column_wrapper<T> col1_da{{2, 3, 1, 5, 4, 23, 9, 8, 7, 7, 6, 43, 37, 26, 21, 16},
                            {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  column_wrapper<T> col1_db{{3, 1, 2, 5, 4, 23, 8, 7, 7, 6, 9, 43, 37, 26, 21, 16},
                            {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1}};
  results = cudf::segmented_sort_by_key(
    input1, input1, segments, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col1_da}});
  results = cudf::segmented_sort_by_key(
    input1, input1, segments, {cudf::order::DESCENDING}, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col1_db}});

  // second row null order.
  column_wrapper<T> col2_12_aa{{0, 0, 0, 1, 1, 4, 5, 22, 21, 5, 5, 6, 6, 7, 8, 8},
                               {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}};
  column_wrapper<T> col2_12_ab{{0, 0, 0, 1, 1, 4, 5, 5, 21, 22, 5, 6, 6, 7, 8, 8},
                               {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}};
  cudf::table_view expected12_aa{{col1_aa, col2_12_aa}};
  cudf::table_view expected12_ab{{col1_ab, col2_12_ab}};
  results = cudf::segmented_sort_by_key(
    input2, input2, segments, {}, {cudf::null_order::AFTER, cudf::null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_aa);
  results = cudf::segmented_sort_by_key(
    input2, input2, segments, {}, {cudf::null_order::BEFORE, cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected12_ab);
}

TYPED_TEST(SegmentedSort, StableNoNulls)
{
  using T = TypeParam;

  // segments             {0   1   2} {3   4} {5} {6   7   8   9  10}{11  12}{13}{14  15}
  column_wrapper<T> col1{{10, 36, 14, 32, 49, 23, 10, 34, 12, 45, 11, 37, 43, 26, 21, 16}};
  column_wrapper<T> col2{{10, 63, 10, 23, 94, 32, 10, 43, 22, 43, 22, 34, 34, 62, 62, 61}};
  // stable sorted order  {0   2   1} {3   4} {5} {6   8  10   7   9}{11  12}{13}{16  15}
  column_wrapper<int> segments{0, 3, 5, 5, 5, 6, 11, 13, 14, 16};
  auto values = cudf::table_view{{col1}};
  auto keys   = cudf::table_view{{col2}};

  // Ascending
  column_wrapper<T> col_asc{{10, 14, 36, 32, 49, 23, 10, 12, 11, 34, 45, 37, 43, 26, 16, 21}};
  auto results =
    cudf::stable_segmented_sort_by_key(values, keys, segments, {cudf::order::ASCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col_asc}});
  // Descending
  column_wrapper<T> col_des{{36, 10, 14, 49, 32, 23, 34, 45, 12, 11, 10, 37, 43, 26, 21, 16}};
  results = cudf::stable_segmented_sort_by_key(values, keys, segments, {cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col_des}});
}

TYPED_TEST(SegmentedSort, StableWithNulls)
{
  using T = TypeParam;

  // segments             {0   1   2} {3   4} {5} {6   7   8   9  10}{11  12}{13}{14  15}
  column_wrapper<T> col1{{10, 36, 0, 32, 49, 23, 10, 0, 12, 45, 11, 37, 43, 0, 21, 16},
                         {1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1}};
  column_wrapper<T> col2{{10, 0, 10, 23, 94, 32, 0, 43, 0, 43, 0, 34, 34, 62, 62, 61},
                         {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  // stable sorted order  {0   2   1} {3   4} {5} {6   8  10   7   9}{11  12}{13}{16  15}
  column_wrapper<int> segments{0, 3, 5, 5, 5, 6, 11, 13, 14, 16};
  auto values = cudf::table_view{{col1}};
  auto keys   = cudf::table_view{{col2}};

  // Ascending
  column_wrapper<T> col_asc{{36, 10, 0, 32, 49, 23, 10, 12, 11, 0, 45, 37, 43, 0, 16, 21},
                            {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1}};
  auto results =
    cudf::stable_segmented_sort_by_key(values, keys, segments, {cudf::order::ASCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col_asc}});

  // Descending
  column_wrapper<T> col_des{{10, 0, 36, 49, 32, 23, 0, 45, 12, 11, 10, 37, 43, 0, 21, 16},
                            {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1}};
  results = cudf::stable_segmented_sort_by_key(values, keys, segments, {cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), cudf::table_view{{col_des}});
}

TEST_F(SegmentedSortInt, NonZeroSegmentsStart)
{
  using T = int;
  // clang-format off
  column_wrapper<T>        col1{{8, 9, 2, 3, 2, 2, 4, 1, 7, 5, 6}};
  column_wrapper<int> segments1{{0,    2,       5,       8,     11}};
  column_wrapper<int> segments2{{      2,       5,       8,      11}};
  column_wrapper<int> segments3{{                  6,    8,      11}};
  column_wrapper<int> segments4{{                  6,    8}};
  column_wrapper<int> segments5{{0,       3,       6}};
  column_wrapper<int> expected1{{0, 1, 2, 4, 3, 7, 5, 6, 9, 10, 8}};
  column_wrapper<int> expected2{{0, 1, 2, 4, 3, 7, 5, 6, 9, 10, 8}};
  column_wrapper<int> expected3{{0, 1, 2, 3, 4, 5, 7, 6, 9, 10, 8}};
  column_wrapper<int> expected4{{0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10}};
  column_wrapper<int> expected5{{2, 0, 1, 4, 5, 3, 6, 7, 8, 9, 10}};
  // clang-format on
  cudf::table_view input{{col1}};
  auto results = cudf::segmented_sorted_order(input, segments1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected1);
  results = cudf::stable_segmented_sorted_order(input, segments1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected1);

  results = cudf::segmented_sorted_order(input, segments2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);
  results = cudf::stable_segmented_sorted_order(input, segments2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);

  results = cudf::segmented_sorted_order(input, segments3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected3);
  results = cudf::stable_segmented_sorted_order(input, segments3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected3);

  results = cudf::segmented_sorted_order(input, segments4);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected4);
  results = cudf::stable_segmented_sorted_order(input, segments4);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected4);

  results = cudf::segmented_sorted_order(input, segments5);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected5);
  results = cudf::stable_segmented_sorted_order(input, segments5);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected5);
}

TEST_F(SegmentedSortInt, Sliced)
{
  using T = int;
  // clang-format off
  column_wrapper<T>        col1{{8, 9, 2, 3, 2, 2, 4, 1, 7, 5, 6}};
  // sliced                                  2, 2, 4, 1, 7, 5, 6
  column_wrapper<int> segments1{{0,    2,       5}};
  column_wrapper<int> segments2{{-4,   0,      2,       5}};
  column_wrapper<int> segments3{{                 7}};
  column_wrapper<int> expected1{{0, 1, 3, 2, 4, 5, 6}};
  column_wrapper<int> expected2{{0, 1, 3, 2, 4, 5, 6}};
  column_wrapper<int> expected3{{0, 1, 2, 3, 4, 5, 6}};
  // clang-format on
  auto slice = cudf::slice(col1, {4, 11})[0];  // 7 elements
  cudf::table_view input{{slice}};
  auto seg_slice = cudf::slice(segments2, {2, 4})[0];  // 2 elements

  // sliced input
  auto results = cudf::segmented_sorted_order(input, segments1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected1);
  results = cudf::stable_segmented_sorted_order(input, segments1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected1);
  // sliced input and sliced segment
  results = cudf::segmented_sorted_order(input, seg_slice);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);
  results = cudf::stable_segmented_sorted_order(input, seg_slice);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected2);
  // sliced input, segment end.
  results = cudf::segmented_sorted_order(input, segments3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected3);
  results = cudf::stable_segmented_sorted_order(input, segments3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected3);
}

TEST_F(SegmentedSortInt, ErrorsMismatchArgSizes)
{
  using T = int;
  column_wrapper<T> col1{{5, 6, 7, 8, 9}};
  column_wrapper<T> segments{{1, 2, 3, 4}};
  cudf::table_view input1{{col1}};
  std::vector<cudf::order> order{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_order{cudf::null_order::AFTER, cudf::null_order::AFTER};

  // Mismatch order sizes
  EXPECT_THROW(cudf::segmented_sort_by_key(input1, input1, segments, order, {}), cudf::logic_error);
  EXPECT_THROW(cudf::stable_segmented_sorted_order(input1, segments, order, {}), cudf::logic_error);
  // Mismatch null precedence sizes
  EXPECT_THROW(cudf::segmented_sorted_order(input1, segments, {}, null_order), cudf::logic_error);
  EXPECT_THROW(cudf::stable_segmented_sort_by_key(input1, input1, segments, {}, null_order),
               cudf::logic_error);
  // Both
  EXPECT_THROW(cudf::segmented_sort_by_key(input1, input1, segments, order, null_order),
               cudf::logic_error);
  EXPECT_THROW(cudf::stable_segmented_sort_by_key(input1, input1, segments, order, null_order),
               cudf::logic_error);
  // segmented_offsets beyond num_rows - undefined behavior, no throw.
  CUDF_EXPECT_NO_THROW(cudf::segmented_sort_by_key(input1, input1, segments));
  CUDF_EXPECT_NO_THROW(cudf::stable_segmented_sort_by_key(input1, input1, segments));
}

// Test specifically verifies the patch added in https://github.com/rapidsai/cudf/pull/12234
// This test will fail if the CUB bug fix is not available or the patch has not been applied.
TEST_F(SegmentedSortInt, Bool)
{
  cudf::test::fixed_width_column_wrapper<bool> col1{
    {true,  false, false, true, true,  true,  true, true, true,  true, true,  true, true, false,
     false, false, false, true, false, false, true, true, true,  true, true,  true, true, false,
     true,  false, true,  true, true,  true,  true, true, false, true, false, false}};

  cudf::test::fixed_width_column_wrapper<int> segments{{0, 5, 10, 15, 20, 25, 30, 40}};

  cudf::test::fixed_width_column_wrapper<int> expected(
    {1,  2,  0,  3,  4,  5,  6,  7,  8,  9,  13, 14, 10, 11, 12, 15, 16, 18, 19, 17,
     20, 21, 22, 23, 24, 27, 29, 25, 26, 28, 36, 38, 39, 30, 31, 32, 33, 34, 35, 37});

  auto test_col = cudf::column_view{col1};
  auto result   = cudf::segmented_sorted_order(cudf::table_view({test_col}), segments);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  result = cudf::stable_segmented_sorted_order(cudf::table_view({test_col}), segments);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}
