/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>
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

CUDF_TEST_PROGRAM_MAIN()
