/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf_test/default_stream.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/scalar/scalar.hpp>

template <typename T>
struct TypedScalarTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedScalarTest, cudf::test::FixedWidthTypes);

TYPED_TEST(TypedScalarTest, DefaultValidity)
{
  using Type = cudf::device_storage_type_t<TypeParam>;
  Type value = static_cast<Type>(cudf::test::make_type_param_scalar<TypeParam>(7));
  cudf::scalar_type_t<TypeParam> s(value, true, cudf::test::get_default_stream());
  EXPECT_EQ(value, s.value(cudf::test::get_default_stream()));
}

struct StringScalarTest : public cudf::test::BaseFixture {};

TEST_F(StringScalarTest, DefaultValidity)
{
  std::string value = "test string";
  auto s            = cudf::string_scalar(value, true, cudf::test::get_default_stream());
  EXPECT_EQ(value, s.to_string(cudf::test::get_default_stream()));
}
