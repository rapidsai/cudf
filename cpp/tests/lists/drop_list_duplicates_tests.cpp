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

//#include <tests/strings/utilities.h>
#include <cudf/lists/drop_list_duplicates.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

//#include <thrust/iterator/constant_iterator.h>
//#include <vector>

struct DropListDuplicatesTest : public cudf::test::BaseFixture {
};

TEST_F(DropListDuplicatesTest, NonNullTable)
{
  using LCW = cudf::test::lists_column_wrapper<int32_t, int32_t>;

  LCW l1{{1, 2, 3}, {3, 2, 1, 4, 1}, {5}, {10, 8, 9}, {6, 7}};

  auto sliced_list = cudf::slice(l1, {1, 5})[0];

  // Ascending

  LCW expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  auto results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{sliced_list},
                                                   cudf::null_equality::EQUAL);
  printf("line %d\n", __LINE__);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, true);
}

TEST_F(DropListDuplicatesTest, NonNullStringTable)
{
  //  using LCW = cudf::test::lists_column_wrapper<std::string, std::string>;

  using LCW = cudf::test::lists_column_wrapper<int32_t, int32_t>;
  //  LCW l1{{"hello", "apple"}, "world", "world", "my"};

  LCW l1{{1, 2, 3}, {3, 2, 1, 4, 1}, {5}, {10, 8, 9}, {6, 7}};

  auto sliced_list = cudf::slice(l1, {1, 5})[0];

  // Ascending

  LCW expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  auto results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{sliced_list},
                                                   cudf::null_equality::EQUAL);
  printf("line %d\n", __LINE__);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, true);
}

#if 0
TEST_F(DropListDuplicatesTest, NonNullTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{5, 4, 3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<float> col2{{4, 5, 3, 4, 9, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> col1_key{{20, 20, 20, 19, 21, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_key{{19, 19, 20, 20, 9, 21}};

  cudf::table_view input{{col1, col2, col1_key, col2_key}};
  std::vector<cudf::size_type> keys{2, 3};

  // Keep first of duplicate
  // The expected table would be sorted in ascending order with respect to keys
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_first{{5, 5, 5, 3, 8}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_first{{4, 4, 4, 3, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_first{{9, 19, 20, 20, 21}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_first{{21, 20, 19, 20, 9}};
  cudf::table_view expected_first{
    {exp_col1_first, exp_col2_first, exp_col1_key_first, exp_col2_key_first}};

  auto got_first = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_FIRST);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_first, got_first->view());

  // keep last of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_last{{5, 5, 4, 3, 8}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_last{{4, 4, 5, 3, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_last{{9, 19, 20, 20, 21}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_last{{21, 20, 19, 20, 9}};
  cudf::table_view expected_last{
    {exp_col1_last, exp_col2_last, exp_col1_key_last, exp_col2_key_last}};

  auto got_last = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_LAST);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_last, got_last->view());

  // Keep unique
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_unique{{5, 5, 3, 8}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_unique{{4, 4, 3, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_unique{{9, 19, 20, 21}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_unique{{21, 20, 20, 9}};
  cudf::table_view expected_unique{
    {exp_col1_unique, exp_col2_unique, exp_col1_key_unique, exp_col2_key_unique}};

  auto got_unique = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_NONE);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_unique, got_unique->view());
}

TEST_F(DropListDuplicatesTest, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> key{{20, 20, 20, 19, 21, 19}, {1, 0, 0, 1, 1, 1}};
  cudf::table_view input{{col, key}};
  std::vector<cudf::size_type> keys{1};

  // Keep first of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_first{{4, 5, 5, 8}, {0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_first{{20, 19, 20, 21}, {0, 1, 1, 1}};
  cudf::table_view expected_first{{exp_col_first, exp_key_col_first}};
  auto got_first = drop_duplicates(
    input, keys, cudf::duplicate_keep_option::KEEP_FIRST, cudf::null_equality::EQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_first, got_first->view());

  // Keep last of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_last{{3, 1, 5, 8}, {1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_last{{20, 19, 20, 21}, {0, 1, 1, 1}};
  cudf::table_view expected_last{{exp_col_last, exp_key_col_last}};
  auto got_last = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_LAST);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_last, got_last->view());

  // Keep unique of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_unique{{5, 8}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_unique{{20, 21}, {1, 1}};
  cudf::table_view expected_unique{{exp_col_unique, exp_key_col_unique}};
  auto got_unique = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_NONE);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_unique, got_unique->view());
}

TEST_F(DropListDuplicatesTest, StringKeyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper key_col{{"all", "new", "all", "new", "the", "strings"},
                                             {1, 1, 1, 0, 1, 1}};
  cudf::table_view input{{col, key_col}};
  std::vector<cudf::size_type> keys{1};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_last{{5, 3, 4, 1, 8}, {1, 1, 0, 1, 1}};
  cudf::test::strings_column_wrapper exp_key_col_last{{"new", "all", "new", "strings", "the"},
                                                      {0, 1, 1, 1, 1}};
  cudf::table_view expected_last{{exp_col_last, exp_key_col_last}};

  auto got_last = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_LAST);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_last, got_last->view());
}

TEST_F(DropListDuplicatesTest, EmptyInputTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col(std::initializer_list<int32_t>{});
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = drop_duplicates(
    input, keys, cudf::duplicate_keep_option::KEEP_FIRST, cudf::null_equality::EQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(DropListDuplicatesTest, NoColumnInputTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = drop_duplicates(
    input, keys, cudf::duplicate_keep_option::KEEP_FIRST, cudf::null_equality::EQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(DropListDuplicatesTest, EmptyKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{};

  auto got = drop_duplicates(
    input, keys, cudf::duplicate_keep_option::KEEP_FIRST, cudf::null_equality::EQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{empty_col}}, got->view());
}

#endif
