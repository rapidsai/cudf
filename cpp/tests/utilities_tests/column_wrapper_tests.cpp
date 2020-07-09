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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
struct FixedWidthColumnWrapperTest : public cudf::test::BaseFixture,
                                     cudf::test::UniformRandomGenerator<cudf::size_type> {
  FixedWidthColumnWrapperTest() : cudf::test::UniformRandomGenerator<cudf::size_type>{1000, 5000} {}

  auto size() { return this->generate(); }

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_CASE(FixedWidthColumnWrapperTest, cudf::test::FixedWidthTypes);

TYPED_TEST(FixedWidthColumnWrapperTest, EmptyIterator)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  cudf::test::fixed_width_column_wrapper<TypeParam> col(sequence, sequence);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}
TYPED_TEST(FixedWidthColumnWrapperTest, EmptyList)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col{};
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableIteratorConstructor)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam> col(sequence, sequence + size);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableListConstructor)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4, 5});

  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllValid)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });

  auto all_valid = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam> col(sequence, sequence + size, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllValid)
{
  auto all_valid = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4, 5}, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllNull)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });

  auto all_null = cudf::test::make_counting_transform_iterator(0, [](auto i) { return false; });

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam> col(sequence, sequence + size, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), size);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllNull)
{
  auto all_null = cudf::test::make_counting_transform_iterator(0, [](auto i) { return false; });

  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4, 5}, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllValid)
{
  auto all_valid = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4, 5}, all_valid);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllNull)
{
  auto all_null = cudf::test::make_counting_transform_iterator(0, [](auto i) { return false; });

  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4, 5}, all_null);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}
