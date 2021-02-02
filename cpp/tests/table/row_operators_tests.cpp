/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <vector>

struct RowOperatorTestForNAN : public cudf::test::BaseFixture {
};

TEST_F(RowOperatorTestForNAN, NANEquality)
{
  cudf::test::fixed_width_column_wrapper<double> col1{{1., double(NAN), 3., 4.}, {1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<double> col2{{1., double(NAN), 3., 4.}, {1, 1, 0, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col1, col2);
}

TEST_F(RowOperatorTestForNAN, NANSorting)
{
  // NULL Before
  cudf::test::fixed_width_column_wrapper<double> input{
    {0.,
     double(NAN),
     -1.,
     7.,
     std::numeric_limits<double>::infinity(),
     1.,
     -1 * std::numeric_limits<double>::infinity()},
    {1, 1, 1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected1{{3, 6, 2, 0, 5, 4, 1}};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence_1{cudf::null_order::BEFORE};
  cudf::table_view input_table{{input}};

  auto got1 = cudf::sorted_order(input_table, column_order, null_precedence_1);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, got1->view());

  // NULL After

  std::vector<cudf::null_order> null_precedence_2{cudf::null_order::AFTER};
  cudf::test::fixed_width_column_wrapper<int32_t> expected2{{6, 2, 0, 5, 4, 1, 3}};

  auto got2 = cudf::sorted_order(input_table, column_order, null_precedence_2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, got2->view());
}

TEST_F(RowOperatorTestForNAN, NANSortingNonNull)
{
  cudf::test::fixed_width_column_wrapper<double> input{
    {0.,
     double(NAN),
     -1.,
     7.,
     std::numeric_limits<double>::infinity(),
     1.,
     -1 * std::numeric_limits<double>::infinity()}};

  cudf::table_view input_table{{input}};

  auto result = cudf::sorted_order(input_table, {cudf::order::ASCENDING});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_asc{{6, 2, 0, 5, 3, 4, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_asc, result->view());
  auto sorted_result = cudf::sort(input_table, {cudf::order::ASCENDING});
  auto gather_result = cudf::gather(input_table, result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_result->view().column(0),
                                      gather_result->view().column(0));

  result = cudf::sorted_order(input_table, {cudf::order::DESCENDING});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_desc{{1, 4, 3, 5, 0, 2, 6}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_desc, result->view());
  sorted_result = cudf::sort(input_table, {cudf::order::DESCENDING});
  gather_result = cudf::gather(input_table, result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_result->view().column(0),
                                      gather_result->view().column(0));
}
