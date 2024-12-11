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

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

class ScalarFactoryTest : public cudf::test::BaseFixture {};

template <typename T>
struct NumericScalarFactory : public ScalarFactoryTest {};

TYPED_TEST_SUITE(NumericScalarFactory, cudf::test::NumericTypes);

TYPED_TEST(NumericScalarFactory, FactoryDefault)
{
  std::unique_ptr<cudf::scalar> s =
    cudf::make_numeric_scalar(cudf::data_type{cudf::type_to_id<TypeParam>()});

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(NumericScalarFactory, TypeCast)
{
  std::unique_ptr<cudf::scalar> s =
    cudf::make_numeric_scalar(cudf::data_type{cudf::type_to_id<TypeParam>()});

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  TypeParam value(37);
  numeric_s->set_value(value);
  EXPECT_EQ(numeric_s->value(), value);
  EXPECT_TRUE(numeric_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

template <typename T>
struct TimestampScalarFactory : public ScalarFactoryTest {};

TYPED_TEST_SUITE(TimestampScalarFactory, cudf::test::TimestampTypes);

TYPED_TEST(TimestampScalarFactory, FactoryDefault)
{
  std::unique_ptr<cudf::scalar> s =
    cudf::make_timestamp_scalar(cudf::data_type{cudf::type_to_id<TypeParam>()});

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(TimestampScalarFactory, TypeCast)
{
  std::unique_ptr<cudf::scalar> s =
    cudf::make_timestamp_scalar(cudf::data_type{cudf::type_to_id<TypeParam>()});

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  TypeParam value(typename TypeParam::duration{37});
  numeric_s->set_value(value);
  EXPECT_EQ(numeric_s->value(), value);
  EXPECT_TRUE(numeric_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

template <typename T>
struct DefaultScalarFactory : public ScalarFactoryTest {};

using MixedTypes = cudf::test::Concat<cudf::test::AllTypes, cudf::test::StringTypes>;
TYPED_TEST_SUITE(DefaultScalarFactory, MixedTypes);

TYPED_TEST(DefaultScalarFactory, FactoryDefault)
{
  std::unique_ptr<cudf::scalar> s =
    cudf::make_default_constructed_scalar(cudf::data_type{cudf::type_to_id<TypeParam>()});

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(DefaultScalarFactory, TypeCast)
{
  std::unique_ptr<cudf::scalar> s =
    cudf::make_default_constructed_scalar(cudf::data_type{cudf::type_to_id<TypeParam>()});

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  EXPECT_NO_THROW((void)numeric_s->value());
  EXPECT_FALSE(numeric_s->is_valid());
  EXPECT_FALSE(s->is_valid());
}

template <typename T>
struct FixedWidthScalarFactory : public ScalarFactoryTest {};

TYPED_TEST_SUITE(FixedWidthScalarFactory, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(FixedWidthScalarFactory, ValueProvided)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(54);

  std::unique_ptr<cudf::scalar> s = cudf::make_fixed_width_scalar<TypeParam>(value);

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(numeric_s->value(), value);
  EXPECT_TRUE(numeric_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

template <typename T>
struct FixedPointScalarFactory : public ScalarFactoryTest {};

TYPED_TEST_SUITE(FixedPointScalarFactory, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointScalarFactory, ValueProvided)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const rep_value      = static_cast<typename decimalXX::rep>(123);
  auto const s              = cudf::make_fixed_point_scalar<decimalXX>(123, scale_type{-2});
  auto const fp_s           = static_cast<cudf::scalar_type_t<decimalXX>*>(s.get());
  auto const expected_dtype = cudf::data_type{cudf::type_to_id<decimalXX>(), -2};

  EXPECT_EQ(s->type(), expected_dtype);
  EXPECT_EQ(fp_s->value(), rep_value);
  EXPECT_TRUE(fp_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

struct StructScalarFactory : public ScalarFactoryTest {};

TEST_F(StructScalarFactory, Basic)
{
  cudf::test::fixed_width_column_wrapper<int> col0{1};
  cudf::test::strings_column_wrapper col1{"abc"};
  cudf::test::lists_column_wrapper<int> col2{{1, 2, 3}};
  cudf::test::structs_column_wrapper struct_col({col0, col1, col2});
  cudf::column_view cv = static_cast<cudf::column_view>(struct_col);
  std::vector<cudf::column_view> children(cv.child_begin(), cv.child_end());

  // table_view constructor
  {
    auto sc = cudf::make_struct_scalar(cudf::table_view{children});
    auto s  = static_cast<cudf::scalar_type_t<cudf::struct_view>*>(sc.get());
    EXPECT_TRUE(s->is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{children}, s->view());
  }

  // host_span constructor
  {
    auto sc = cudf::make_struct_scalar(cudf::host_span<cudf::column_view const>{children});
    auto s  = static_cast<cudf::scalar_type_t<cudf::struct_view>*>(sc.get());
    EXPECT_TRUE(s->is_valid());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{children}, s->view());
  }
}

CUDF_TEST_PROGRAM_MAIN()
