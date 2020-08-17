/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
class EncodeNumericTests : public cudf::test::BaseFixture {
};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_CASE(EncodeNumericTests, NumericTypesNotBool);

TYPED_TEST(EncodeNumericTests, SingleNullEncode)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input({1}, {0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect({0});
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, EmptyEncode)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input({});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect({});
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, SimpleNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 2, 3, 2, 3, 2, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{{0, 1, 2, 1, 2, 1, 0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_keys{{1, 2, 3}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, SimpleWithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 2, 3, 2, 3, 2, 1},
                                                          {1, 1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{{0, 1, 2, 3, 2, 1, 0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_keys{{1, 2, 3, 0}, {1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  cudf::test::print(result.first->view().column(0));
  cudf::test::print(expect_keys);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, UnorderedWithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{2, 1, 5, 1, 1, 3, 2},
                                                          {0, 1, 1, 1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{{4, 0, 3, 0, 4, 2, 1}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_keys{{1, 2, 3, 5, 0}, {1, 1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

struct EncodeStringTest : public cudf::test::BaseFixture {
};

TEST_F(EncodeStringTest, SimpleNoNulls)
{
  cudf::test::strings_column_wrapper input{"a", "b", "c", "d", "a"};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{0, 1, 2, 3, 0};
  cudf::test::strings_column_wrapper expect_keys{"a", "b", "c", "d"};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TEST_F(EncodeStringTest, SimpleWithNulls)
{
  cudf::test::strings_column_wrapper input{{"a", "b", "c", "d", "a"}, {1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{0, 3, 1, 2, 3};
  cudf::test::strings_column_wrapper expect_keys{{"a", "c", "d", "0"}, {1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TEST_F(EncodeStringTest, UnorderedWithNulls)
{
  cudf::test::strings_column_wrapper input{{"ef", "a", "c", "d", "ef", "a"}, {1, 0, 1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{3, 4, 1, 2, 4, 0};
  cudf::test::strings_column_wrapper expect_keys{{"a", "c", "d", "ef", "0"}, {1, 1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, TableEncodeWithNulls)
{
  auto col_1 = cudf::test::fixed_width_column_wrapper<TypeParam>({1, 0, 2, 0, 1}, {1, 0, 1, 0, 1});
  auto col_2 = cudf::test::fixed_width_column_wrapper<TypeParam>({1, 3, 2, 0, 1}, {1, 1, 1, 0, 1});
  auto input = cudf::table_view({col_1, col_2});

  auto expect_keys_col1 =
    cudf::test::fixed_width_column_wrapper<TypeParam>({1, 2, 0, 0}, {1, 1, 0, 0});
  auto expect_keys_col2 =
    cudf::test::fixed_width_column_wrapper<TypeParam>({1, 2, 3, 0}, {1, 1, 1, 0});
  auto expect_keys = cudf::table_view({expect_keys_col1, expect_keys_col2});
  auto expect      = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 2, 1, 3, 0});

  auto const result = cudf::encode(input);

  cudf::test::expect_tables_equivalent(result.first->view(), expect_keys);
  cudf::test::expect_columns_equal(result.second->view(), expect);
}

CUDF_TEST_PROGRAM_MAIN()
