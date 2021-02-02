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

#include <cudf/sorting.hpp>

#include <type_traits>
#include <vector>

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

namespace cudf {
namespace test {

template <typename T>
struct SortLists : public BaseFixture {
};

// using NumericTypesNotBool = Concat<IntegralTypesNotBool, FloatingPointTypes>;
TYPED_TEST_CASE(SortLists, NumericTypes);

using SortListsInt = SortLists<int>;
TEST_F(SortListsInt, ErrorsTableSizes)
{
  LCW<int> col1{{3, 1, 2}, {1}, {2}, {0}, {10, 9, 9}, {6, 7}};
  fixed_width_column_wrapper<int> col2{{5, 4, 3, 5, 8, 5}, {1, 1, 0, 1, 1, 1}};
  strings_column_wrapper col3({"d", "e", "a", "d", "k", "d"}, {1, 1, 0, 1, 1, 1});
  LCW<int> col4{{3, 1, 2}, {1}, {2}, {0}, {10, 9, 9, 4}, {6, 7}};
  table_view input1{{col1}};
  table_view input2{{col1, col2}};
  table_view input3{{col2, col3}};
  table_view input4{{col4}};
  table_view input5{{col1, col4}};
  // Valid
  CUDF_EXPECT_NO_THROW(cudf::sort_lists(input1, input1, {}, {}));
  // Non-List keys
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input2, input1, {}, {}),
                            "segmented_sort_by_key only supports lists columns");
  // Non-List values
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input1, input2, {}, {}),
                            "segmented_sort_by_key only supports lists columns");
  // Both
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input2, input2, {}, {}),
                            "segmented_sort_by_key only supports lists columns");
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input2, input3, {}, {}),
                            "segmented_sort_by_key only supports lists columns");
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input3, input3, {}, {}),
                            "segmented_sort_by_key only supports lists columns");
  // List sizes mismatch key
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input5, input4, {}, {}),
                            "size of each list in a row of table should be same");
  // List sizes mismatch value
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input1, input5, {}, {}),
                            "size of each list in a row of table should be same");
  // List sizes mismatch between key-value
  CUDF_EXPECT_THROW_MESSAGE(cudf::sort_lists(input1, input4, {}, {}),
                            "size of each list in a row of table should be same");
}

TEST_F(SortListsInt, ErrorsMismatchArgSizes)
{
  LCW<int> col1{{3, 1, 2}, {1}, {2}, {0}, {10, 9, 9}, {6, 7}};
  table_view input1{{col1}};

  // Mismatch order sizes
  EXPECT_THROW(cudf::sort_lists(input1, input1, {order::ASCENDING, order::ASCENDING}, {}),
               logic_error);
  // Mismatch null precedence sizes
  EXPECT_THROW(cudf::sort_lists(input1, input1, {}, {null_order::AFTER, null_order::AFTER}),
               logic_error);
  // Both
  EXPECT_THROW(
    cudf::sort_lists(
      input1, input1, {order::ASCENDING, order::ASCENDING}, {null_order::AFTER, null_order::AFTER}),
    logic_error);
}

TYPED_TEST(SortLists, NoNull)
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
  auto results = cudf::sort_lists(
    input, input, {order::ASCENDING, order::ASCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1);

  results = cudf::sort_lists(
    input, input, {order::ASCENDING, order::ASCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1);

  // Descending
  // LCW<int>  order{{3, 5, 4, 0, 1, 2}, {0}, {0, 2, 1},  {1, 0}};
  LCW<T> expected3{{4, 4, 4, 3, 2, 1}, {5}, {9, 9, 8}, {7, 6}};
  LCW<T> expected4{{3, 2, 1, 3, 1, 2}, {0}, {10, 9, 9}, {7, 6}};
  table_view expected_table2{{expected3, expected4}};
  results = cudf::sort_lists(
    input, input, {order::DESCENDING, order::DESCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2);

  results = cudf::sort_lists(
    input, input, {order::DESCENDING, order::DESCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2);
}

TYPED_TEST(SortLists, Nulls)
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
  auto results = cudf::sort_lists(
    input, input, {order::ASCENDING, order::ASCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1a);

  // LCW<int>  order{{3, 2, 1, 0, 4, 5}, {0}, {2, 1, 0},  {0, 1}};
  LCW<T> expected1b{{{4, 1, 2, 3, 4, 4}, valids1a.rbegin()}, {5}, {8, 9, 9}, {6, 7}};
  LCW<T> expected2b{{2, 2, 1, 3, 1, 3}, {0}, {{9, 9, 10}, valids2b.begin()}, {6, 7}};
  table_view expected_table1b{{expected1b, expected2b}};
  results = cudf::sort_lists(
    input, input, {order::ASCENDING, order::ASCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1b);

  // Descending
  LCW<T> expected3a{{{4, 4, 4, 3, 2, 1}, valids1a.rbegin()}, {5}, {9, 9, 8}, {7, 6}};
  LCW<T> expected4a{{2, 3, 1, 3, 1, 2}, {0}, {{9, 10, 9}, valids2.rbegin()}, {7, 6}};
  table_view expected_table2a{{expected3a, expected4a}};
  results = cudf::sort_lists(
    input, input, {order::DESCENDING, order::DESCENDING}, {null_order::AFTER, null_order::AFTER});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2a);

  LCW<T> expected3b{{{4, 4, 3, 2, 1, 4}, valids1a.begin()}, {5}, {9, 9, 8}, {7, 6}};
  LCW<T> expected4b{{3, 1, 3, 1, 2, 2}, {0}, {{10, 9, 9}, valids2b.begin()}, {7, 6}};
  table_view expected_table2b{{expected3b, expected4b}};
  results = cudf::sort_lists(
    input, input, {order::DESCENDING, order::DESCENDING}, {null_order::BEFORE, null_order::BEFORE});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2b);
}

TEST_F(SortListsInt, KeyValues)
{
  using T      = int;
  using LCWstr = cudf::test::lists_column_wrapper<cudf::string_view>;

  // List<T>
  LCW<T> a{{21, 22, 23, 22}, {22, 21, 23, 22}};
  LCW<T> b{{13, 14, 12, 11}, {14, 13, 12, 11}};
  LCWstr c{{"a", "b", "c", "d"}, {"a", "b", "c", "d"}};

  // Ascending {a}
  // LCW<T> order{{0, 1, 3, 2}, {1, 0, 3, 2}};
  LCW<T> sorted_a1{{21, 22, 22, 23}, {21, 22, 22, 23}};
  LCW<T> sorted_b1{{13, 14, 11, 12}, {13, 14, 11, 12}};
  LCWstr sorted_c1{{"a", "b", "d", "c"}, {"b", "a", "d", "c"}};
  auto results = cudf::sort_lists(table_view{{a, b, c}}, table_view{{a}}, {}, {});
  table_view expected_table1{{sorted_a1, sorted_b1, sorted_c1}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table1);

  // Ascending {a,b}
  // LCW<int>  order{{0, 3, 1, 2}, {1, 3, 0, 2}};
  LCW<T> sorted_a2{{21, 22, 22, 23}, {21, 22, 22, 23}};
  LCW<T> sorted_b2{{13, 11, 14, 12}, {13, 11, 14, 12}};
  LCWstr sorted_c2{{"a", "d", "b", "c"}, {"b", "d", "a", "c"}};
  table_view expected_table2{{sorted_a2, sorted_b2, sorted_c2}};
  table_view keys{{a, b}};
  table_view values{{a, b, c}};
  results = cudf::sort_lists(values, keys, {}, {});
  CUDF_TEST_EXPECT_TABLES_EQUAL(results->view(), expected_table2);
}

}  // namespace test
}  // namespace cudf
