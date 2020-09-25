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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>
#include <string>

template <typename T>
struct EmptyLikeTest : public cudf::test::BaseFixture {
};

using numeric_types = cudf::test::NumericTypes;

TYPED_TEST_CASE(EmptyLikeTest, numeric_types);

TYPED_TEST(EmptyLikeTest, ColumnNumericTests)
{
  cudf::size_type size   = 10;
  cudf::mask_state state = cudf::mask_state::ALL_VALID;
  auto input    = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, size, state);
  auto expected = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, 0);
  auto got      = cudf::empty_like(input->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *got);
}

struct EmptyLikeStringTest : public EmptyLikeTest<std::string> {
};

void check_empty_string_columns(cudf::column_view lhs, cudf::column_view rhs)
{
  EXPECT_EQ(lhs.type(), rhs.type());
  EXPECT_EQ(lhs.size(), 0);
  EXPECT_EQ(lhs.null_count(), 0);
  EXPECT_EQ(lhs.nullable(), false);
  EXPECT_EQ(lhs.has_nulls(), false);
  // An empty column is not required to have children
}

TEST_F(EmptyLikeStringTest, ColumnStringTest)
{
  std::vector<const char*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr,
                                     "absent stop words"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto got = cudf::empty_like(strings);
  check_empty_string_columns(got->view(), strings);
}

std::unique_ptr<cudf::table> create_table(cudf::size_type size, cudf::mask_state state)
{
  auto num_column_1 = make_numeric_column(cudf::data_type{cudf::type_id::INT64}, size, state);
  auto num_column_2 = make_numeric_column(cudf::data_type{cudf::type_id::INT32}, size, state);
  auto num_column_3 = make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, size, state);
  auto num_column_4 = make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32}, size, state);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(num_column_1));
  columns.push_back(std::move(num_column_2));
  columns.push_back(std::move(num_column_3));
  columns.push_back(std::move(num_column_4));

  return std::make_unique<cudf::table>(std::move(columns));
}

void expect_tables_prop_equal(cudf::table_view lhs, cudf::table_view rhs)
{
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
  for (cudf::size_type index = 0; index < lhs.num_columns(); index++)
    CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(lhs.column(index), rhs.column(index));
}

struct EmptyLikeTableTest : public cudf::test::BaseFixture {
};

TEST_F(EmptyLikeTableTest, TableTest)
{
  cudf::mask_state state = cudf::mask_state::ALL_VALID;
  cudf::size_type size   = 10;
  auto input             = create_table(size, state);
  auto expected          = create_table(0, cudf::mask_state::UNINITIALIZED);
  auto got               = cudf::empty_like(input->view());

  expect_tables_prop_equal(got->view(), expected->view());
}

template <typename T>
struct AllocateLikeTest : public cudf::test::BaseFixture {
};
;

TYPED_TEST_CASE(AllocateLikeTest, numeric_types);

TYPED_TEST(AllocateLikeTest, ColumnNumericTestSameSize)
{
  // For same size as input
  cudf::size_type size   = 10;
  cudf::mask_state state = cudf::mask_state::ALL_VALID;
  auto input    = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, size, state);
  auto expected = make_numeric_column(
    cudf::data_type{cudf::type_to_id<TypeParam>()}, size, cudf::mask_state::UNINITIALIZED);
  auto got = cudf::allocate_like(input->view());
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(*expected, *got);
}

TYPED_TEST(AllocateLikeTest, ColumnNumericTestSpecifiedSize)
{
  // For same size as input
  cudf::size_type size           = 10;
  cudf::size_type specified_size = 5;
  cudf::mask_state state         = cudf::mask_state::ALL_VALID;
  auto input    = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, size, state);
  auto expected = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                      specified_size,
                                      cudf::mask_state::UNINITIALIZED);
  auto got      = cudf::allocate_like(input->view(), specified_size);
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(*expected, *got);
}

CUDF_TEST_PROGRAM_MAIN()
