/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <algorithm>
#include <cmath>

using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;

struct Distinct : public cudf::test::BaseFixture {
};

TEST_F(Distinct, StringKeyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 4, 5, 5, 8, 1}, {1, 0, 0, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper key_col{{"all", "new", "new", "all", "new", "the", "strings"},
                                             {1, 1, 1, 1, 0, 1, 1}};
  cudf::table_view input{{col, key_col}};
  std::vector<cudf::size_type> keys{1};

  cudf::test::fixed_width_column_wrapper<int32_t> exp_sort_col{{5, 5, 4, 1, 8}, {1, 1, 0, 1, 1}};
  cudf::test::strings_column_wrapper exp_sort_key_col{{"new", "all", "new", "strings", "the"},
                                                      {0, 1, 1, 1, 1}};
  cudf::table_view expected_sort{{exp_sort_col, exp_sort_key_col}};

  auto got_unordered = distinct(input, keys);
  auto key_view      = got_unordered->select(keys.begin(), keys.end());
  auto sorted_result = cudf::sort_by_key(got_unordered->view(), key_view);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, sorted_result->view());
}

TEST_F(Distinct, EmptyInputTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col(std::initializer_list<int32_t>{});
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = distinct(input, keys, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Distinct, NoColumnInputTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = distinct(input, keys, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Distinct, EmptyKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{};

  auto got = distinct(input, keys, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{empty_col}}, got->view());
}

TEST_F(Distinct, NonNullTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{6, 6, 3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<float> col2{{6, 6, 3, 4, 9, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> col1_key{{20, 20, 20, 19, 21, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_key{{19, 19, 20, 20, 9, 21}};

  cudf::table_view input{{col1, col2, col1_key, col2_key}};
  std::vector<cudf::size_type> keys{2, 3};

  // The expected table would be sorted in ascending order with respect to keys
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1{{5, 5, 6, 3, 8}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2{{4, 4, 6, 3, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key{{9, 19, 20, 20, 21}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key{{21, 20, 19, 20, 9}};
  cudf::table_view expected{{exp_col1, exp_col2, exp_col1_key, exp_col2_key}};

  auto result        = distinct(input, keys);
  auto key_view      = result->select(keys.begin(), keys.end());
  auto sorted_result = cudf::sort_by_key(result->view(), key_view);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, sorted_result->view());
}

TEST_F(Distinct, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 4, 1, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> key{{20, 20, 20, 19, 21, 19}, {1, 0, 0, 1, 1, 1}};
  cudf::table_view input{{col, key}};
  std::vector<cudf::size_type> keys{1};

  // nulls are equal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_equal_col{{4, 1, 5, 8}, {0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_equal_key_col{{20, 19, 20, 21}, {0, 1, 1, 1}};
  cudf::table_view expected_equal{{exp_equal_col, exp_equal_key_col}};
  auto res_equal    = distinct(input, keys, null_equality::EQUAL);
  auto equal_keys   = res_equal->select(keys.begin(), keys.end());
  auto sorted_equal = cudf::sort_by_key(res_equal->view(), equal_keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_equal, sorted_equal->view());

  // nulls are unequal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_unequal_col{{4, 1, 4, 5, 8}, {0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_unequal_key_col{{20, 19, 20, 20, 21},
                                                                      {0, 1, 0, 1, 1}};
  cudf::table_view expected_unequal{{exp_unequal_col, exp_unequal_key_col}};
  auto res_unequal    = distinct(input, keys, null_equality::UNEQUAL);
  auto sorted_unequal = cudf::sort(res_unequal->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_unequal, sorted_unequal->view());
}
