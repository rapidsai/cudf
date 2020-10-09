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

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

class ScalarFactoryTest : public cudf::test::BaseFixture {
  cudaStream_t _stream{0};

 public:
  cudaStream_t stream() { return _stream; }
};

template <typename T>
struct NumericScalarFactory : public ScalarFactoryTest {
  static constexpr auto factory = cudf::make_numeric_scalar;
};

TYPED_TEST_CASE(NumericScalarFactory, cudf::test::NumericTypes);

TYPED_TEST(NumericScalarFactory, FactoryDefault)
{
  std::unique_ptr<cudf::scalar> s =
    this->factory(cudf::data_type{cudf::type_to_id<TypeParam>()}, this->stream(), this->mr());

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(NumericScalarFactory, TypeCast)
{
  std::unique_ptr<cudf::scalar> s =
    this->factory(cudf::data_type{cudf::type_to_id<TypeParam>()}, this->stream(), this->mr());

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  TypeParam value(37);
  numeric_s->set_value(value);
  EXPECT_EQ(numeric_s->value(), value);
  EXPECT_TRUE(numeric_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

template <typename T>
struct TimestampScalarFactory : public ScalarFactoryTest {
  static constexpr auto factory = cudf::make_timestamp_scalar;
};

TYPED_TEST_CASE(TimestampScalarFactory, cudf::test::TimestampTypes);

TYPED_TEST(TimestampScalarFactory, FactoryDefault)
{
  std::unique_ptr<cudf::scalar> s =
    this->factory(cudf::data_type{cudf::type_to_id<TypeParam>()}, this->stream(), this->mr());

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(TimestampScalarFactory, TypeCast)
{
  std::unique_ptr<cudf::scalar> s =
    this->factory(cudf::data_type{cudf::type_to_id<TypeParam>()}, this->stream(), this->mr());

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  TypeParam value(typename TypeParam::duration{37});
  numeric_s->set_value(value);
  EXPECT_EQ(numeric_s->value(), value);
  EXPECT_TRUE(numeric_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

template <typename T>
struct DefaultScalarFactory : public cudf::test::BaseFixture {
  static constexpr auto factory = cudf::make_default_constructed_scalar;
};

using MixedTypes = cudf::test::Concat<cudf::test::AllTypes, cudf::test::StringTypes>;
TYPED_TEST_CASE(DefaultScalarFactory, MixedTypes);

TYPED_TEST(DefaultScalarFactory, FactoryDefault)
{
  std::unique_ptr<cudf::scalar> s = this->factory(cudf::data_type{cudf::type_to_id<TypeParam>()});

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(DefaultScalarFactory, TypeCast)
{
  std::unique_ptr<cudf::scalar> s = this->factory(cudf::data_type{cudf::type_to_id<TypeParam>()});

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  EXPECT_NO_THROW(numeric_s->value());
  EXPECT_FALSE(numeric_s->is_valid());
  EXPECT_FALSE(s->is_valid());
}

template <typename T>
struct FixedWidthScalarFactory : public ScalarFactoryTest {
};

TYPED_TEST_CASE(FixedWidthScalarFactory, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(FixedWidthScalarFactory, ValueProvided)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(54);

  std::unique_ptr<cudf::scalar> s =
    cudf::make_fixed_width_scalar<TypeParam>(value, this->stream(), this->mr());

  auto numeric_s = static_cast<cudf::scalar_type_t<TypeParam>*>(s.get());

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(numeric_s->value(), value);
  EXPECT_TRUE(numeric_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

template <typename T>
struct FixedPointScalarFactory : public ScalarFactoryTest {
};

TYPED_TEST_CASE(FixedPointScalarFactory, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointScalarFactory, ValueProvided)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const rep_value = static_cast<typename decimalXX::rep>(123);
  auto const s =
    cudf::make_fixed_point_scalar<decimalXX>(123, scale_type{-2}, this->stream(), this->mr());
  auto const fp_s = static_cast<cudf::scalar_type_t<decimalXX>*>(s.get());

  EXPECT_EQ(s->type(), cudf::data_type{cudf::type_to_id<decimalXX>()});
  EXPECT_EQ(fp_s->value(), rep_value);
  EXPECT_TRUE(fp_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

CUDF_TEST_PROGRAM_MAIN()
