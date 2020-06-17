/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cmath>
#include <ctgmath>
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;
template <typename T>
struct UniqueCountCommon : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(UniqueCountCommon, cudf::test::NumericTypes);

TYPED_TEST(UniqueCountCommon, NoNull)
{
  using T = TypeParam;

  std::vector<T> input = cudf::test::make_type_param_vector<T>(
    {1, 3, 3, 4, 31, 1, 8, 2, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4});

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end());

  cudf::size_type expected = std::set<double>(input.begin(), input.end()).size();
  EXPECT_EQ(expected,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

struct UniqueCount : public cudf::test::BaseFixture {
};

TEST_F(UniqueCount, WithNull)
{
  using T = int32_t;

  // Considering 70 as null
  std::vector<T> input = {1, 3, 3, 70, 31, 1, 8, 2, 0, 70, 1, 70, 10, 40, 31, 42, 0, 42, 8, 5, 70};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end(), valid.begin());

  cudf::size_type expected = std::set<double>(input.begin(), input.end()).size();
  EXPECT_EQ(expected,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, IgnoringNull)
{
  using T = int32_t;

  // Considering 70 and 3 as null
  std::vector<T> input = {1, 3, 3, 70, 31, 1, 8, 2, 0, 70, 1, 70, 10, 40, 31, 42, 0, 42, 8, 5, 70};
  std::vector<cudf::size_type> valid = {1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end(), valid.begin());

  cudf::size_type expected = std::set<T>(input.begin(), input.end()).size();
  // Removing 2 from expected to remove count for 70 and 3
  EXPECT_EQ(expected - 2,
            cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, WithNansAndNull)
{
  using T = float;

  std::vector<T> input               = {1,  3,  NAN, 70, 31,  1, 8,   2, 0, 70, 1,
                          70, 10, 40,  31, NAN, 0, NAN, 8, 5, 70};
  std::vector<cudf::size_type> valid = {1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  cudf::size_type expected = std::set<T>(input.begin(), input.end()).size();
  EXPECT_EQ(expected,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, WithNansOnly)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  cudf::size_type expected = 5;
  EXPECT_EQ(expected,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, NansAsNullWithNoNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  cudf::size_type expected = 5;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, NansAsNullWithNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  cudf::size_type expected = 4;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, NansAsNullWithIgnoreNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  cudf::size_type expected = 3;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, EmptyColumn)
{
  using T = float;

  cudf::test::fixed_width_column_wrapper<T> input_col{};

  cudf::size_type expected = 0;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, StringColumnWithNull)
{
  cudf::test::strings_column_wrapper input_col{
    {"", "this", "is", "this", "This", "a", "column", "of", "the", "strings"},
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 1}};

  cudf::size_type expected =
    (std::vector<std::string>{"", "this", "is", "This", "a", "column", "of", "strings"}).size();
  EXPECT_EQ(expected,
            cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID));
}

struct DropDuplicate : public cudf::test::BaseFixture {
};

TEST_F(DropDuplicate, NonNullTable)
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

  cudf::test::expect_tables_equal(expected_first, got_first->view());

  // keep last of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_last{{5, 5, 4, 3, 8}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_last{{4, 4, 5, 3, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_last{{9, 19, 20, 20, 21}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_last{{21, 20, 19, 20, 9}};
  cudf::table_view expected_last{
    {exp_col1_last, exp_col2_last, exp_col1_key_last, exp_col2_key_last}};

  auto got_last = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_LAST);

  cudf::test::expect_tables_equal(expected_last, got_last->view());

  // Keep unique
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_unique{{5, 5, 3, 8}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_unique{{4, 4, 3, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_unique{{9, 19, 20, 21}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_unique{{21, 20, 20, 9}};
  cudf::table_view expected_unique{
    {exp_col1_unique, exp_col2_unique, exp_col1_key_unique, exp_col2_key_unique}};

  auto got_unique = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_NONE);

  cudf::test::expect_tables_equal(expected_unique, got_unique->view());
}

TEST_F(DropDuplicate, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> key{{20, 20, 20, 19, 21, 19}, {1, 0, 0, 1, 1, 1}};
  cudf::table_view input{{col, key}};
  std::vector<cudf::size_type> keys{1};

  // Keep first of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_first{{4, 5, 5, 8}, {0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_first{{20, 19, 20, 21}, {0, 1, 1, 1}};
  cudf::table_view expected_first{{exp_col_first, exp_key_col_first}};
  auto got_first =
    drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);

  cudf::test::expect_tables_equal(expected_first, got_first->view());

  // Keep last of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_last{{3, 1, 5, 8}, {1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_last{{20, 19, 20, 21}, {0, 1, 1, 1}};
  cudf::table_view expected_last{{exp_col_last, exp_key_col_last}};
  auto got_last = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_LAST);

  cudf::test::expect_tables_equal(expected_last, got_last->view());

  // Keep unique of duplicate
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_unique{{5, 8}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_unique{{20, 21}, {1, 1}};
  cudf::table_view expected_unique{{exp_col_unique, exp_key_col_unique}};
  auto got_unique = drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_NONE);

  cudf::test::expect_tables_equal(expected_unique, got_unique->view());
}

TEST_F(DropDuplicate, StringKeyColumn)
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

  cudf::test::expect_tables_equal(expected_last, got_last->view());
}

TEST_F(DropDuplicate, EmptyInputTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col(std::initializer_list<int32_t>{});
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{1, 2};

  auto got =
    drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);

  cudf::test::expect_tables_equal(input, got->view());
}

TEST_F(DropDuplicate, NoColumnInputTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{1, 2};

  auto got =
    drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);

  cudf::test::expect_tables_equal(input, got->view());
}

TEST_F(DropDuplicate, EmptyKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{};

  auto got =
    drop_duplicates(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);

  cudf::test::expect_tables_equal(cudf::table_view{{empty_col}}, got->view());
}
