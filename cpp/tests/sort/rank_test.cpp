/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <tuple>
#include <vector>

namespace {
void run_rank_test(cudf::table_view input,
                   cudf::table_view expected,
                   cudf::rank_method method,
                   cudf::order column_order,
                   cudf::null_policy null_handling,
                   cudf::null_order null_precedence,
                   bool percentage)
{
  int i = 0;
  for (auto&& input_column : input) {
    // Rank
    auto got_rank_column =
      cudf::rank(input_column, method, column_order, null_handling, null_precedence, percentage);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected.column(i), got_rank_column->view());
    i++;
  }
}

using input_arg_t = std::tuple<cudf::order, cudf::null_policy, cudf::null_order>;
input_arg_t asce_keep{cudf::order::ASCENDING, cudf::null_policy::EXCLUDE, cudf::null_order::AFTER};
input_arg_t asce_top{cudf::order::ASCENDING, cudf::null_policy::INCLUDE, cudf::null_order::BEFORE};
input_arg_t asce_bottom{
  cudf::order::ASCENDING, cudf::null_policy::INCLUDE, cudf::null_order::AFTER};

input_arg_t desc_keep{
  cudf::order::DESCENDING, cudf::null_policy::EXCLUDE, cudf::null_order::BEFORE};
input_arg_t desc_top{cudf::order::DESCENDING, cudf::null_policy::INCLUDE, cudf::null_order::AFTER};
input_arg_t desc_bottom{
  cudf::order::DESCENDING, cudf::null_policy::INCLUDE, cudf::null_order::BEFORE};
using test_case_t = std::tuple<cudf::table_view, cudf::table_view>;
}  // namespace

template <typename T>
struct Rank : public cudf::test::BaseFixture {
  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<T> col2{{5, 4, 3, 5, 8, 5}, {1, 1, 0, 1, 1, 1}};
  cudf::test::strings_column_wrapper col3{{"d", "e", "a", "d", "k", "d"}, {1, 1, 1, 1, 1, 1}};

  void run_all_tests(cudf::rank_method method,
                     input_arg_t input_arg,
                     cudf::column_view const col1_rank,
                     cudf::column_view const col2_rank,
                     cudf::column_view const col3_rank,
                     bool percentage = false)
  {
    if (std::is_same_v<T, bool>) return;
    for (auto const& test_case : {
           // Non-null column
           test_case_t{cudf::table_view{{col1}}, cudf::table_view{{col1_rank}}},
           // Null column
           test_case_t{cudf::table_view{{col2}}, cudf::table_view{{col2_rank}}},
           // Table
           test_case_t{cudf::table_view{{col1, col2}}, cudf::table_view{{col1_rank, col2_rank}}},
           // Table with String column
           test_case_t{cudf::table_view{{col1, col2, col3}},
                       cudf::table_view{{col1_rank, col2_rank, col3_rank}}},
         }) {
      auto [input, output] = test_case;

      run_rank_test(input,
                    output,
                    method,
                    std::get<0>(input_arg),
                    std::get<1>(input_arg),
                    std::get<2>(input_arg),
                    percentage);
    }
  }
};

TYPED_TEST_SUITE(Rank, cudf::test::NumericTypes);

// fixed_width_column_wrapper<T>   col1{{  5,   4,   3,   5,   8,   5}};
//                                        3,   2,   1,   4,   6,   5
TYPED_TEST(Rank, first_asce_keep)
{
  // ASCENDING
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 4, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 1, -1, 3, 5, 4},
                                                                    {1, 1, 0, 1, 1, 1}};  // KEEP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 3, 6, 4},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::FIRST, asce_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_asce_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 4, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {3, 2, 1, 4, 6, 5}};  // BEFORE = TOP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 3, 6, 4}};
  this->run_all_tests(cudf::rank_method::FIRST, asce_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_asce_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 4, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 1, 6, 3, 5, 4}};  // AFTER  = BOTTOM
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 3, 6, 4}};
  this->run_all_tests(cudf::rank_method::FIRST, asce_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_desc_keep)
{
  // DESCENDING
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 3, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 5, -1, 3, 1, 4},
                                                                    {1, 1, 0, 1, 1, 1}};  // KEEP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 4, 1, 5},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::FIRST, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 3, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {3, 6, 1, 4, 2, 5}};  // AFTER  = TOP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 4, 1, 5}};
  this->run_all_tests(cudf::rank_method::FIRST, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 3, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 5, 6, 3, 1, 4}};  // BEFORE = BOTTOM
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 4, 1, 5}};
  this->run_all_tests(cudf::rank_method::FIRST, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_asce_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 1, -1, 2, 3, 2},
                                                                    {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 3, 1, 2, 4, 2},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::DENSE, asce_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_asce_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 3, 1, 2, 4, 2}};
  this->run_all_tests(cudf::rank_method::DENSE, asce_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_asce_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 1, 4, 2, 3, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 3, 1, 2, 4, 2}};
  this->run_all_tests(cudf::rank_method::DENSE, asce_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 3, -1, 2, 1, 2},
                                                                    {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 4, 3, 1, 3},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::DENSE, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 4, 1, 3, 2, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 4, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::DENSE, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 4, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::DENSE, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_asce_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 1, -1, 2, 5, 2},
                                                                    {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 2, 6, 2},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::MIN, asce_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_asce_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 2, 6, 2}};
  this->run_all_tests(cudf::rank_method::MIN, asce_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_asce_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 1, 6, 2, 5, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 2, 6, 2}};
  this->run_all_tests(cudf::rank_method::MIN, asce_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 5, -1, 2, 1, 2},
                                                                    {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 3, 1, 3},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::MIN, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 6, 1, 3, 2, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::MIN, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::MIN, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_asce_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{4, 1, -1, 4, 5, 4},
                                                                    {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{4, 5, 1, 4, 6, 4},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::MAX, asce_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_asce_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{4, 5, 1, 4, 6, 4}};
  this->run_all_tests(cudf::rank_method::MAX, asce_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_asce_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{4, 1, 6, 4, 5, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{4, 5, 1, 4, 6, 4}};
  this->run_all_tests(cudf::rank_method::MAX, asce_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{4, 5, -1, 4, 1, 4},
                                                                    {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{5, 2, 6, 5, 1, 5},
                                                                    {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::MAX, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{5, 6, 1, 5, 2, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{5, 2, 6, 5, 1, 5}};
  this->run_all_tests(cudf::rank_method::MAX, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{5, 2, 6, 5, 1, 5}};
  this->run_all_tests(cudf::rank_method::MAX, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_asce_keep)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 1, -1, 3, 5, 3}, {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{3, 5, 1, 3, 6, 3}, {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::AVERAGE, asce_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_asce_top)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{3, 5, 1, 3, 6, 3}};
  this->run_all_tests(cudf::rank_method::AVERAGE, asce_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_asce_bottom)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 1, 6, 3, 5, 3}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{3, 5, 1, 3, 6, 3}};
  this->run_all_tests(cudf::rank_method::AVERAGE, asce_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 5, -1, 3, 1, 3}, {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{4, 2, 6, 4, 1, 4}, {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_desc_top)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{4, 6, 1, 4, 2, 4}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{4, 2, 6, 4, 1, 4}};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{4, 2, 6, 4, 1, 4}};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_bottom, col1_rank, col2_rank, col3_rank);
}

// percentage==true (dense, not-dense)
TYPED_TEST(Rank, dense_asce_keep_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{
    {2.0 / 3.0, 1.0 / 3.0, -1., 2.0 / 3.0, 1., 2.0 / 3.0}, {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{0.5, 0.75, 0.25, 0.5, 1., 0.5},
                                                           {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::DENSE, asce_keep, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, dense_asce_top_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{0.5, 0.75, 0.25, 0.5, 1., 0.5}};
  this->run_all_tests(cudf::rank_method::DENSE, asce_top, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, dense_asce_bottom_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{0.5, 0.25, 1., 0.5, 0.75, 0.5}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{0.5, 0.75, 0.25, 0.5, 1., 0.5}};
  this->run_all_tests(cudf::rank_method::DENSE, asce_bottom, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, min_desc_keep_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{0.4, 1., -1., 0.4, 0.2, 0.4},
                                                           {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{
    {0.5, 1.0 / 3.0, 1., 0.5, 1.0 / 6.0, 0.5}, {1, 1, 1, 1, 1, 1}};
  this->run_all_tests(cudf::rank_method::MIN, desc_keep, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, min_desc_top_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{
    {0.5, 1., 1.0 / 6.0, 0.5, 1.0 / 3.0, 0.5}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{
    {0.5, 1.0 / 3.0, 1., 0.5, 1.0 / 6.0, 0.5}};
  this->run_all_tests(cudf::rank_method::MIN, desc_top, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, min_desc_bottom_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{
    {0.5, 1.0 / 3.0, 1., 0.5, 1.0 / 6.0, 0.5}};
  this->run_all_tests(cudf::rank_method::MIN, desc_bottom, col1_rank, col2_rank, col3_rank, true);
}

struct RankLarge : public cudf::test::BaseFixture {
};

TEST_F(RankLarge, average_large)
{
  // testcase of https://github.com/rapidsai/cudf/issues/9703
  auto iter = thrust::counting_iterator<int64_t>(0);
  cudf::test::fixed_width_column_wrapper<int64_t> col1(iter, iter + 10558);
  auto result = cudf::rank(col1,
                           cudf::rank_method::AVERAGE,
                           {},
                           cudf::null_policy::EXCLUDE,
                           cudf::null_order::AFTER,
                           false);
  cudf::test::fixed_width_column_wrapper<double, int> expected(iter + 1, iter + 10559);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}
