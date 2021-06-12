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

#include <cudf/utilities/type_checks.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

namespace cudf {
namespace test {

template <typename T>
struct ColumnTypeCheckTestFixedWidth : public cudf::test::BaseFixture {
};

struct ColumnTypeCheckTestString : public cudf::test::BaseFixture {
};

template <typename T>
struct ColumnTypeCheckTestDictionary : public cudf::test::BaseFixture {
};

struct ColumnTypeCheckTestList : public cudf::test::BaseFixture {
};

struct ColumnTypeCheckTestStruct : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ColumnTypeCheckTestFixedWidth, cudf::test::FixedWidthTypes);
TYPED_TEST_CASE(ColumnTypeCheckTestDictionary, cudf::test::FixedWidthTypes);

TYPED_TEST(ColumnTypeCheckTestFixedWidth, SameType)
{
  fixed_width_column_wrapper<TypeParam> lhs{1, 1}, rhs{2};
  EXPECT_TRUE(column_types_equal(lhs, rhs));
}

TEST_F(ColumnTypeCheckTestString, SameType)
{
  strings_column_wrapper lhs{{'a', 'a'}}, rhs{{'b'}};
  EXPECT_TRUE(column_types_equal(lhs, rhs));

  strings_column_wrapper lhs2{}, rhs2{{'b'}};
  EXPECT_TRUE(column_types_equal(lhs2, rhs2));

  strings_column_wrapper lhs3{}, rhs3{};
  EXPECT_TRUE(column_types_equal(lhs3, rhs3));
}

TEST_F(ColumnTypeCheckTestList, SameType)
{
  using LCW = lists_column_wrapper<int32_t>;

  LCW lhs{}, rhs{};
  EXPECT_TRUE(column_types_equal(lhs, rhs));

  LCW lhs2{{1, 2, 3}}, rhs2{{4, 5}};
  EXPECT_TRUE(column_types_equal(lhs2, rhs2));

  LCW lhs3{{LCW{1}, LCW{2, 3}}}, rhs3{{LCW{4, 5}}};
  EXPECT_TRUE(column_types_equal(lhs3, rhs3));

  LCW lhs4{{LCW{1}, LCW{}, LCW{2, 3}}}, rhs4{{LCW{4, 5}, LCW{}}};
  EXPECT_TRUE(column_types_equal(lhs4, rhs4));
}

TYPED_TEST(ColumnTypeCheckTestDictionary, SameType)
{
  using DCW = dictionary_column_wrapper<TypeParam>;
  DCW lhs{1, 1, 2, 3}, rhs{5, 5};
  EXPECT_TRUE(column_types_equal(lhs, rhs));

  DCW lhs2{1, 1, 2, 3}, rhs2{};
  EXPECT_TRUE(column_types_equal(lhs2, rhs2));

  DCW lhs3{}, rhs3{};
  EXPECT_TRUE(column_types_equal(lhs3, rhs3));
}

TEST_F(ColumnTypeCheckTestStruct, SameType)
{
  using SCW      = structs_column_wrapper;
  using FCW      = fixed_width_column_wrapper<int32_t>;
  using StringCW = strings_column_wrapper;
  using LCW      = lists_column_wrapper<int32_t>;
  using DCW      = dictionary_column_wrapper<int32_t>;

  FCW lf1{1, 2, 3}, rf1{0, 1};
  StringCW lf2{"a", "bb", ""}, rf2{"cc", "d"};
  LCW lf3{LCW{1, 2}, LCW{}, LCW{4}}, rf3{LCW{1}, LCW{2}};
  DCW lf4{5, 5, 5}, rf4{9, 9};

  SCW lhs{lf1, lf2, lf3, lf4}, rhs{rf1, rf2, rf3, rf4};
  EXPECT_TRUE(column_types_equal(lhs, rhs));
}

}  // namespace test
}  // namespace cudf
