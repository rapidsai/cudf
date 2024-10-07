/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/scalar/scalar.hpp>

template <typename T>
struct TypedScalarTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedScalarTestWithoutFixedPoint : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedScalarTest, cudf::test::FixedWidthTypes);
TYPED_TEST_SUITE(TypedScalarTestWithoutFixedPoint, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(TypedScalarTest, DefaultValidity)
{
  using Type = cudf::device_storage_type_t<TypeParam>;
  Type value = static_cast<Type>(cudf::test::make_type_param_scalar<TypeParam>(7));
  cudf::scalar_type_t<TypeParam> s(value);

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.value());
}

TYPED_TEST(TypedScalarTest, ConstructNull)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(5);
  cudf::scalar_type_t<TypeParam> s(value, false);

  EXPECT_FALSE(s.is_valid());
}

TYPED_TEST(TypedScalarTestWithoutFixedPoint, SetValue)
{
  TypeParam init  = cudf::test::make_type_param_scalar<TypeParam>(0);
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(9);
  cudf::scalar_type_t<TypeParam> s(init, true);
  s.set_value(value);

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.value());
}

TYPED_TEST(TypedScalarTestWithoutFixedPoint, SetNull)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(6);
  cudf::scalar_type_t<TypeParam> s(value, true);
  s.set_valid_async(false);

  EXPECT_FALSE(s.is_valid());
}

TYPED_TEST(TypedScalarTest, CopyConstructor)
{
  using Type = cudf::device_storage_type_t<TypeParam>;
  Type value = static_cast<Type>(cudf::test::make_type_param_scalar<TypeParam>(8));
  cudf::scalar_type_t<TypeParam> s(value);
  auto s2 = s;

  EXPECT_TRUE(s2.is_valid());
  EXPECT_EQ(value, s2.value());
}

TYPED_TEST(TypedScalarTest, MoveConstructor)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(8);
  cudf::scalar_type_t<TypeParam> s(value);
  auto data_ptr = s.data();
  auto mask_ptr = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(data_ptr, s2.data());
}

struct StringScalarTest : public cudf::test::BaseFixture {};

TEST_F(StringScalarTest, DefaultValidity)
{
  std::string value = "test string";
  auto s            = cudf::string_scalar(value);

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.to_string());
}

TEST_F(StringScalarTest, CopyConstructor)
{
  std::string value = "test_string";
  auto s            = cudf::string_scalar(value);
  auto s2           = s;

  EXPECT_TRUE(s2.is_valid());
  EXPECT_EQ(value, s2.to_string());
}

TEST_F(StringScalarTest, MoveConstructor)
{
  std::string value = "another test string";
  auto s            = cudf::string_scalar(value);
  auto data_ptr     = s.data();
  auto mask_ptr     = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(data_ptr, s2.data());
}

struct ListScalarTest : public cudf::test::BaseFixture {};

TEST_F(ListScalarTest, DefaultValidityNonNested)
{
  auto data = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto s    = cudf::list_scalar(data);

  EXPECT_TRUE(s.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s.view());
}

TEST_F(ListScalarTest, DefaultValidityNested)
{
  auto data = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {2}, {}, {4, 5}};
  auto s    = cudf::list_scalar(data);

  EXPECT_TRUE(s.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s.view());
}

TEST_F(ListScalarTest, MoveColumnConstructor)
{
  auto data = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto col  = cudf::column(data);
  auto ptr  = col.view().data<int32_t>();
  auto s    = cudf::list_scalar(std::move(col));

  EXPECT_TRUE(s.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s.view());
  EXPECT_EQ(ptr, s.view().data<int32_t>());
}

TEST_F(ListScalarTest, CopyConstructorNonNested)
{
  auto data = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto s    = cudf::list_scalar(data);
  auto s2   = s;

  EXPECT_TRUE(s2.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s2.view());
  EXPECT_NE(s.view().data<int32_t>(), s2.view().data<int32_t>());
}

TEST_F(ListScalarTest, CopyConstructorNested)
{
  auto data = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {2}, {}, {4, 5}};
  auto s    = cudf::list_scalar(data);
  auto s2   = s;

  EXPECT_TRUE(s2.is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(data, s2.view());
  EXPECT_NE(s.view().child(0).data<int32_t>(), s2.view().child(0).data<int32_t>());
  EXPECT_NE(s.view().child(1).data<int32_t>(), s2.view().child(1).data<int32_t>());
}

TEST_F(ListScalarTest, MoveConstructorNonNested)
{
  auto data     = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto s        = cudf::list_scalar(data);
  auto data_ptr = s.view().data<int32_t>();
  auto mask_ptr = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(data_ptr, s2.view().data<int32_t>());
  EXPECT_EQ(s.view().data<int32_t>(), nullptr);  // NOLINT
}

TEST_F(ListScalarTest, MoveConstructorNested)
{
  auto data       = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {2}, {}, {4, 5}};
  auto s          = cudf::list_scalar(data);
  auto offset_ptr = s.view().child(0).data<cudf::size_type>();
  auto data_ptr   = s.view().child(1).data<int32_t>();
  auto mask_ptr   = s.validity_data();
  decltype(s) s2(std::move(s));

  EXPECT_EQ(mask_ptr, s2.validity_data());
  EXPECT_EQ(offset_ptr, s2.view().child(0).data<cudf::size_type>());
  EXPECT_EQ(data_ptr, s2.view().child(1).data<int32_t>());
  EXPECT_EQ(s.view().data<int32_t>(), nullptr);  // NOLINT
  EXPECT_EQ(s.view().num_children(), 0);         // NOLINT
}

struct StructScalarTest : public cudf::test::BaseFixture {};

TEST_F(StructScalarTest, Basic)
{
  cudf::test::fixed_width_column_wrapper<int> col0{1};
  cudf::test::strings_column_wrapper col1{"abc"};
  cudf::test::lists_column_wrapper<int> col2{{1, 2, 3}};
  cudf::test::structs_column_wrapper struct_col({col0, col1, col2});
  cudf::column_view cv = static_cast<cudf::column_view>(struct_col);
  std::vector<cudf::column_view> children(cv.child_begin(), cv.child_end());

  // table_view constructor
  {
    auto s = cudf::struct_scalar(children, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{children}, s.view());
  }

  // host_span constructor
  {
    auto s = cudf::struct_scalar(cudf::host_span<cudf::column_view const>{children}, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{children}, s.view());
  }
}

TEST_F(StructScalarTest, BasicNulls)
{
  cudf::test::fixed_width_column_wrapper<int> col0{1};
  cudf::test::strings_column_wrapper col1{"abc"};
  cudf::test::lists_column_wrapper<int> col2{{1, 2, 3}};
  std::vector<cudf::column_view> src_children({col0, col1, col2});

  std::vector<std::unique_ptr<cudf::column>> src_columns;

  // structs_column_wrapper takes ownership of the incoming columns, so make a copy
  src_columns.push_back(std::make_unique<cudf::column>(src_children[0]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[1]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[2]));
  cudf::test::structs_column_wrapper valid_struct_col(std::move(src_columns), {true});
  cudf::column_view vcv = static_cast<cudf::column_view>(valid_struct_col);
  std::vector<cudf::column_view> valid_children(vcv.child_begin(), vcv.child_end());

  // structs_column_wrapper takes ownership of the incoming columns, so make a copy
  src_columns.push_back(std::make_unique<cudf::column>(src_children[0]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[1]));
  src_columns.push_back(std::make_unique<cudf::column>(src_children[2]));
  cudf::test::structs_column_wrapper invalid_struct_col(std::move(src_columns), {false});
  cudf::column_view icv = static_cast<cudf::column_view>(invalid_struct_col);
  std::vector<cudf::column_view> invalid_children(icv.child_begin(), icv.child_end());

  // table_view constructor
  {
    auto s = cudf::struct_scalar(cudf::table_view{src_children}, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{valid_children}, s.view());
  }
  // host_span constructor
  {
    auto s = cudf::struct_scalar(cudf::host_span<cudf::column_view const>{src_children}, true);
    EXPECT_TRUE(s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{valid_children}, s.view());
  }

  // with nulls, we expect the incoming children to get nullified by passing false to
  // the scalar constructor itself. so we use the unmodified `children` as the input, but
  // we compare against the modified `invalid_children` produced by the source column as
  // proof that the scalar did the validity pushdown.

  // table_view constructor
  {
    auto s = cudf::struct_scalar(cudf::table_view{src_children}, false);
    EXPECT_TRUE(!s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{invalid_children}, s.view());
  }

  // host_span constructor
  {
    auto s = cudf::struct_scalar(cudf::host_span<cudf::column_view const>{src_children}, false);
    EXPECT_TRUE(!s.is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{invalid_children}, s.view());
  }
}

CUDF_TEST_PROGRAM_MAIN()
