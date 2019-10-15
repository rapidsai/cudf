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
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
struct FixedWidthColumnWrapperTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(FixedWidthColumnWrapperTest, cudf::test::FixedWidthTypes);

TYPED_TEST(FixedWidthColumnWrapperTest, First) { ASSERT_TRUE(true); }

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableConstructor) {
  auto iter = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });

  cudf::test::fixed_width_column_wrapper<TypeParam> col(iter, iter + 100);

  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 100);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableConstructor) {

  auto iter = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });

  auto valid_iter = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  cudf::test::fixed_width_column_wrapper<TypeParam> col(iter, iter + 100,
                                                        valid_iter);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 100);
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
}
