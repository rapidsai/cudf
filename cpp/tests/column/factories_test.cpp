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
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

class ColumnFactoryTest : public cudf::test::BaseFixture {
  cudf::size_type _size{1000};
  cudaStream_t _stream{0};

 public:
  cudf::size_type size() { return _size; }
  cudaStream_t stream() { return _stream; }
};

template <typename T>
class NumericFactoryTest : public ColumnFactoryTest {
};

TYPED_TEST_CASE(NumericFactoryTest, cudf::test::NumericTypes);

TYPED_TEST(NumericFactoryTest, EmptyNoMask)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          0,
                                          cudf::mask_state::UNALLOCATED,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, EmptyAllValidMask)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          0,
                                          cudf::mask_state::ALL_VALID,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, EmptyAllNullMask)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          0,
                                          cudf::mask_state::ALL_NULL,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, NoMask)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          this->size(),
                                          cudf::mask_state::UNALLOCATED,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, UnitializedMask)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          this->size(),
                                          cudf::mask_state::UNINITIALIZED,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_TRUE(column->nullable());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, AllValidMask)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          this->size(),
                                          cudf::mask_state::ALL_VALID,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, AllNullMask)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          this->size(),
                                          cudf::mask_state::ALL_NULL,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(this->size(), column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, NullMaskAsParm)
{
  rmm::device_buffer null_mask{create_null_mask(this->size(), cudf::mask_state::ALL_NULL)};
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          this->size(),
                                          null_mask,
                                          this->size(),
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(this->size(), column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, NullMaskAsEmptyParm)
{
  rmm::device_buffer null_mask{};
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          this->size(),
                                          null_mask,
                                          0,
                                          this->stream(),
                                          this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

class NonNumericFactoryTest : public ColumnFactoryTest,
                              public testing::WithParamInterface<cudf::type_id> {
};

// All non-numeric types should throw
TEST_P(NonNumericFactoryTest, NonNumericThrow)
{
  auto construct = [this]() {
    auto column = cudf::make_numeric_column(cudf::data_type{GetParam()},
                                            this->size(),
                                            cudf::mask_state::UNALLOCATED,
                                            this->stream(),
                                            this->mr());
  };
  EXPECT_THROW(construct(), cudf::logic_error);
}

INSTANTIATE_TEST_CASE_P(NonNumeric,
                        NonNumericFactoryTest,
                        testing::ValuesIn(cudf::test::non_numeric_type_ids));

template <typename T>
class FixedWidthFactoryTest : public ColumnFactoryTest {
};

TYPED_TEST_CASE(FixedWidthFactoryTest, cudf::test::FixedWidthTypes);

TYPED_TEST(FixedWidthFactoryTest, EmptyNoMask)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              0,
                                              cudf::mask_state::UNALLOCATED,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
}

template <typename T>
class EmptyFactoryTest : public ColumnFactoryTest {
};

TYPED_TEST_CASE(EmptyFactoryTest, cudf::test::AllTypes);

TYPED_TEST(EmptyFactoryTest, Empty)
{
  auto type   = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto column = cudf::make_empty_column(type);
  EXPECT_EQ(type, column->type());
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, EmptyAllValidMask)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              0,
                                              cudf::mask_state::ALL_VALID,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, EmptyAllNullMask)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              0,
                                              cudf::mask_state::ALL_NULL,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, NoMask)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              this->size(),
                                              cudf::mask_state::UNALLOCATED,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, UnitializedMask)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              this->size(),
                                              cudf::mask_state::UNINITIALIZED,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_TRUE(column->nullable());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, AllValidMask)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              this->size(),
                                              cudf::mask_state::ALL_VALID,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, AllNullMask)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              this->size(),
                                              cudf::mask_state::ALL_NULL,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(this->size(), column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, NullMaskAsParm)
{
  rmm::device_buffer null_mask{create_null_mask(this->size(), cudf::mask_state::ALL_NULL)};
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              this->size(),
                                              null_mask,
                                              this->size(),
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(this->size(), column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(FixedWidthFactoryTest, NullMaskAsEmptyParm)
{
  rmm::device_buffer null_mask{};
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              this->size(),
                                              null_mask,
                                              0,
                                              this->stream(),
                                              this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

class NonFixedWidthFactoryTest : public ColumnFactoryTest,
                                 public testing::WithParamInterface<cudf::type_id> {
};

// All non-fixed types should throw
TEST_P(NonFixedWidthFactoryTest, NonFixedWidthThrow)
{
  auto construct = [this]() {
    auto column = cudf::make_fixed_width_column(cudf::data_type{GetParam()},
                                                this->size(),
                                                cudf::mask_state::UNALLOCATED,
                                                this->stream(),
                                                this->mr());
  };
  EXPECT_THROW(construct(), cudf::logic_error);
}

INSTANTIATE_TEST_CASE_P(NonFixedWidth,
                        NonFixedWidthFactoryTest,
                        testing::ValuesIn(cudf::test::non_fixed_width_type_ids));

TYPED_TEST(NumericFactoryTest, FromScalar)
{
  cudf::numeric_scalar<TypeParam> value(12);
  auto column = cudf::make_column_from_scalar(value, 10);
  EXPECT_EQ(column->type(), value.type());
  EXPECT_EQ(10, column->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, FromNullScalar)
{
  cudf::numeric_scalar<TypeParam> value(0, false);
  auto column = cudf::make_column_from_scalar(value, 10);
  EXPECT_EQ(column->type(), value.type());
  EXPECT_EQ(10, column->size());
  EXPECT_EQ(10, column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(NumericFactoryTest, FromScalarWithZeroSize)
{
  cudf::numeric_scalar<TypeParam> value(7);
  auto column = cudf::make_column_from_scalar(value, 0);
  EXPECT_EQ(column->type(), value.type());
  EXPECT_EQ(0, column->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TEST_F(ColumnFactoryTest, FromStringScalar)
{
  cudf::string_scalar value("hello");
  auto column = cudf::make_column_from_scalar(value, 1);
  EXPECT_EQ(1, column->size());
  EXPECT_EQ(column->type(), value.type());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
}

TEST_F(ColumnFactoryTest, FromNullStringScalar)
{
  cudf::string_scalar value("", false);
  auto column = cudf::make_column_from_scalar(value, 2);
  EXPECT_EQ(2, column->size());
  EXPECT_EQ(column->type(), value.type());
  EXPECT_EQ(2, column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
}

TEST_F(ColumnFactoryTest, FromStringScalarWithZeroSize)
{
  cudf::string_scalar value("hello");
  auto column = cudf::make_column_from_scalar(value, 0);
  EXPECT_EQ(0, column->size());
  EXPECT_EQ(column->type(), value.type());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
}

TEST_F(ColumnFactoryTest, DictionaryFromStringScalar)
{
  cudf::string_scalar value("hello");
  auto column = cudf::make_dictionary_from_scalar(value, 1);
  EXPECT_EQ(1, column->size());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_id::DICTIONARY32});
  EXPECT_EQ(0, column->null_count());
  EXPECT_EQ(2, column->num_children());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
}

TEST_F(ColumnFactoryTest, DictionaryFromStringScalarError)
{
  cudf::string_scalar value("hello", false);
  EXPECT_THROW(cudf::make_dictionary_from_scalar(value, 1), cudf::logic_error);
}
