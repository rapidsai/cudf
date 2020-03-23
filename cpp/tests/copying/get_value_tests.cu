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

#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/copying.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>


namespace cudf {
namespace test {


template <typename T>
struct FixedWidthGetValueTest : public BaseFixture {};

TYPED_TEST_CASE(FixedWidthGetValueTest, FixedWidthTypes);


TYPED_TEST(FixedWidthGetValueTest, BasicGet) {
  fixed_width_column_wrapper<TypeParam> col{9, 8, 7, 6};
  auto s = experimental::get_element(col, 2);

  using ScalarType = experimental::scalar_type_t<TypeParam>;
  auto typed_s = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(TypeParam(7), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetFromNullable) {
  fixed_width_column_wrapper<TypeParam> col({9, 8, 7, 6},
                                            {0, 1, 0, 1});
  auto s = experimental::get_element(col, 1);

  using ScalarType = experimental::scalar_type_t<TypeParam>;
  auto typed_s = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(TypeParam(8), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetNull) {
  fixed_width_column_wrapper<TypeParam> col({9, 8, 7, 6},
                                            {0, 1, 0, 1});
  auto s = experimental::get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

} // namespace test
} // namespace cudf
