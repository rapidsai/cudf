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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/typed_tests.hpp>

#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <gmock/gmock.h>

class ColumnFactoryTest : public cudf::test::BaseFixture {
  cudf::size_type _size{1000};
  cudaStream_t _stream{0};

 public:
  cudf::size_type size() { return _size; }
  cudaStream_t stream() { return _stream; }
};

template <typename T>
class NumericFactoryTest : public ColumnFactoryTest {};

TYPED_TEST_CASE(NumericFactoryTest, cudf::test::NumericTypes);

TYPED_TEST(NumericFactoryTest, EmptyNoMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, 0,
      cudf::mask_state::UNALLOCATED, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, EmptyAllValidMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, 0,
      cudf::mask_state::ALL_VALID, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, EmptyAllNullMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, 0,
      cudf::mask_state::ALL_NULL, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, NoMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, this->size(),
      cudf::mask_state::UNALLOCATED, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, UnitializedMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, this->size(),
      cudf::mask_state::UNINITIALIZED, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_TRUE(column->nullable());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, AllValidMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, this->size(),
      cudf::mask_state::ALL_VALID, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, AllNullMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, this->size(),
      cudf::mask_state::ALL_NULL, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(this->size(), column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

class NonNumericFactoryTest
    : public ColumnFactoryTest,
      public testing::WithParamInterface<cudf::type_id> {};

// All non-numeric types should throw
TEST_P(NonNumericFactoryTest, NonNumericThrow) {
  auto construct = [this]() {
    auto column = cudf::make_numeric_column(
        cudf::data_type{GetParam()}, this->size(),
        cudf::mask_state::UNALLOCATED, this->stream(), this->mr());
  };
  EXPECT_THROW(construct(), cudf::logic_error);
}

INSTANTIATE_TEST_CASE_P(NonNumeric, NonNumericFactoryTest,
                        testing::ValuesIn(cudf::test::non_numeric_type_ids));