/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

class ColumnFactoryTest : public cudf::test::BaseFixture {
  cudf::size_type _size{1000};

 public:
  cudf::size_type size() { return _size; }
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
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
                                          std::move(null_mask),
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
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                          this->size(),
                                          rmm::device_buffer{},
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
                                              std::move(null_mask),
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
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                              this->size(),
                                              rmm::device_buffer{},
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

template <typename T>
class ListsFixedWidthLeafTest : public ColumnFactoryTest {
};

TYPED_TEST_CASE(ListsFixedWidthLeafTest, cudf::test::FixedWidthTypes);

TYPED_TEST(ListsFixedWidthLeafTest, FromNonNested)
{
  using FCW     = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using LCW     = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using valid_t = std::vector<cudf::valid_type>;

  auto s   = cudf::make_list_scalar(FCW({1, -1, 3}, {1, 0, 1}));
  auto col = cudf::make_column_from_scalar(*s, 3);

  auto expected = LCW{LCW({1, 2, 3}, valid_t{1, 0, 1}.begin()),
                      LCW({1, 2, 3}, valid_t{1, 0, 1}.begin()),
                      LCW({1, 2, 3}, valid_t{1, 0, 1}.begin())};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, expected);
}

TYPED_TEST(ListsFixedWidthLeafTest, FromNested)
{
  using LCW     = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using valid_t = std::vector<cudf::valid_type>;

#define row_data \
  LCW({LCW({-1, -1, 3}, valid_t{0, 0, 1}.begin()), LCW{}, LCW{}}, valid_t{1, 0, 1}.begin())

  auto s   = cudf::make_list_scalar(row_data);
  auto col = cudf::make_column_from_scalar(*s, 5);

  auto expected = LCW{row_data, row_data, row_data, row_data, row_data};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, expected);

#undef row_data
}

template <typename T>
class ListsDictionaryLeafTest : public ColumnFactoryTest {
};

TYPED_TEST_CASE(ListsDictionaryLeafTest, cudf::test::FixedWidthTypes);

TYPED_TEST(ListsDictionaryLeafTest, FromNonNested)
{
  using DCW      = cudf::test::dictionary_column_wrapper<TypeParam>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::offset_type>;

  auto s   = cudf::make_list_scalar(DCW({1, 3, -1, 1, 3}, {1, 1, 0, 1, 1}));
  auto col = cudf::make_column_from_scalar(*s, 2);

  DCW leaf({1, 3, -1, 1, 3, 1, 3, -1, 1, 3}, {1, 1, 0, 1, 1, 1, 1, 0, 1, 1});
  offset_t offsets{0, 5, 10};
  auto mask = cudf::create_null_mask(2, cudf::mask_state::UNALLOCATED);

  auto expected = cudf::make_lists_column(2, offsets.release(), leaf.release(), 0, std::move(mask));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, *expected);
}

TYPED_TEST(ListsDictionaryLeafTest, FromNested)
{
  using DCW      = cudf::test::dictionary_column_wrapper<TypeParam>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::offset_type>;

  DCW leaf({1, 3, -1, 1, 3, 1, 3, -1, 1, 3}, {1, 1, 0, 1, 1, 1, 1, 0, 1, 1});
  offset_t offsets{0, 3, 3, 6, 6, 10};
  auto mask = cudf::create_null_mask(5, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 1, 2, false);
  auto data = cudf::make_lists_column(5, offsets.release(), leaf.release(), 0, std::move(mask));

  auto s   = cudf::make_list_scalar(*data);
  auto col = cudf::make_column_from_scalar(*s, 3);

  DCW leaf2(
    {1, 3, -1, 1, 3, 1, 3, -1, 1, 3, 1, 3, -1, 1, 3,
     1, 3, -1, 1, 3, 1, 3, -1, 1, 3, 1, 3, -1, 1, 3},
    {1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1});
  offset_t offsets2{0, 3, 3, 6, 6, 10, 13, 13, 16, 16, 20, 23, 23, 26, 26, 30};
  auto mask2 = cudf::create_null_mask(15, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask2.data()), 1, 2, false);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask2.data()), 6, 7, false);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask2.data()), 11, 12, false);
  auto nested =
    cudf::make_lists_column(15, offsets2.release(), leaf2.release(), 3, std::move(mask2));

  offset_t offsets3{0, 5, 10, 15};
  auto mask3 = cudf::create_null_mask(3, cudf::mask_state::UNALLOCATED);
  auto expected =
    cudf::make_lists_column(3, offsets3.release(), std::move(nested), 0, std::move(mask3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, *expected);
}

class ListsStringLeafTest : public ColumnFactoryTest {
};

TEST_F(ListsStringLeafTest, FromNonNested)
{
  using SCW     = cudf::test::strings_column_wrapper;
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  using valid_t = std::vector<cudf::valid_type>;

  auto s   = cudf::make_list_scalar(SCW({"xx", "", "z"}, {true, false, true}));
  auto col = cudf::make_column_from_scalar(*s, 4);

  auto expected = LCW{LCW({"xx", "", "z"}, valid_t{1, 0, 1}.begin()),
                      LCW({"xx", "", "z"}, valid_t{1, 0, 1}.begin()),
                      LCW({"xx", "", "z"}, valid_t{1, 0, 1}.begin()),
                      LCW({"xx", "", "z"}, valid_t{1, 0, 1}.begin())};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, expected);
}

TEST_F(ListsStringLeafTest, FromNested)
{
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  using valid_t = std::vector<cudf::valid_type>;

#define row_data                                                              \
  LCW({LCW{},                                                                 \
       LCW({"@@", "rapids", "", "四", "ら"}, valid_t{1, 1, 0, 1, 1}.begin()), \
       LCW{},                                                                 \
       LCW({"hello", ""}, valid_t{1, 0}.begin())},                            \
      valid_t{0, 1, 1, 1}.begin())

  auto s = cudf::make_list_scalar(row_data);

  auto col = cudf::make_column_from_scalar(*s, 3);

  auto expected = LCW{row_data, row_data, row_data};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, expected);
#undef row_data
}

template <typename T>
class ListsStructsLeafTest : public ColumnFactoryTest {
 protected:
  using SCW = cudf::test::structs_column_wrapper;
  /**
   * @brief Create a structs column that contains 3 fields: int, string, List<int>
   */
  template <typename MaskIterator>
  SCW make_test_structs_column(cudf::test::fixed_width_column_wrapper<T> field1,
                               cudf::test::strings_column_wrapper field2,
                               cudf::test::lists_column_wrapper<T, int32_t> field3,
                               MaskIterator mask)
  {
    return SCW{{field1, field2, field3}, mask};
  }
};

TYPED_TEST_CASE(ListsStructsLeafTest, cudf::test::FixedWidthTypes);

TYPED_TEST(ListsStructsLeafTest, FromNonNested)
{
  using LCWinner_t = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using StringCW   = cudf::test::strings_column_wrapper;
  using offset_t   = cudf::test::fixed_width_column_wrapper<cudf::offset_type>;
  using valid_t    = std::vector<cudf::valid_type>;

  auto data = this->make_test_structs_column(
    {{1, 3, 5, 2, 4}, {1, 0, 1, 0, 1}},
    StringCW({"fleur", "flower", "", "花", "はな"}, {true, true, false, true, true}),
    LCWinner_t({{1, 2}, {}, {4, 5}, {-1}, {}}, valid_t{1, 1, 1, 1, 0}.begin()),
    valid_t{1, 1, 1, 0, 1}.begin());
  auto s   = cudf::make_list_scalar(data);
  auto col = cudf::make_column_from_scalar(*s, 2);

  auto leaf = this->make_test_structs_column(
    {{1, 3, 5, 2, 4, 1, 3, 5, 2, 4}, {1, 0, 1, 0, 1, 1, 0, 1, 0, 1}},
    StringCW({"fleur", "flower", "", "花", "はな", "fleur", "flower", "", "花", "はな"},
             {true, true, false, true, true, true, true, false, true, true}),
    LCWinner_t({{1, 2}, {}, {4, 5}, {-1}, {}, {1, 2}, {}, {4, 5}, {-1}, {}},
               valid_t{1, 1, 1, 1, 0, 1, 1, 1, 1, 0}.begin()),
    valid_t{1, 1, 1, 0, 1, 1, 1, 1, 0, 1}.begin());
  auto expected = cudf::make_lists_column(2,
                                          offset_t{0, 5, 10}.release(),
                                          leaf.release(),
                                          0,
                                          cudf::create_null_mask(2, cudf::mask_state::UNALLOCATED));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, *expected);
}

TYPED_TEST(ListsStructsLeafTest, FromNested)
{
  using LCWinner_t = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using StringCW   = cudf::test::strings_column_wrapper;
  using offset_t   = cudf::test::fixed_width_column_wrapper<cudf::offset_type>;
  using valid_t    = std::vector<cudf::valid_type>;
  auto leaf        = this->make_test_structs_column(
    {{1, 2}, {0, 1}},
    StringCW({"étoile", "星"}, {true, true}),
    LCWinner_t({LCWinner_t{}, LCWinner_t{42}}, valid_t{1, 1}.begin()),
    valid_t{0, 1}.begin());
  auto mask = cudf::create_null_mask(3, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, 1, false);
  auto data =
    cudf::make_lists_column(3, offset_t{0, 0, 1, 2}.release(), leaf.release(), 1, std::move(mask));
  auto s = cudf::make_list_scalar(*data);

  auto col = cudf::make_column_from_scalar(*s, 3);

  auto leaf2 = this->make_test_structs_column(
    {{1, 2, 1, 2, 1, 2}, {0, 1, 0, 1, 0, 1}},
    StringCW({"étoile", "星", "étoile", "星", "étoile", "星"},
             {true, true, true, true, true, true}),
    LCWinner_t(
      {LCWinner_t{}, LCWinner_t{42}, LCWinner_t{}, LCWinner_t{42}, LCWinner_t{}, LCWinner_t{42}},
      valid_t{1, 1, 1, 1, 1, 1}.begin()),
    valid_t{0, 1, 0, 1, 0, 1}.begin());
  auto mask2 = cudf::create_null_mask(9, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask2.data()), 0, 1, false);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask2.data()), 3, 4, false);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask2.data()), 6, 7, false);
  auto data2 = cudf::make_lists_column(
    9, offset_t{0, 0, 1, 2, 2, 3, 4, 4, 5, 6}.release(), leaf2.release(), 3, std::move(mask2));
  auto expected = cudf::make_lists_column(3,
                                          offset_t{0, 3, 6, 9}.release(),
                                          std::move(data2),
                                          0,
                                          cudf::create_null_mask(3, cudf::mask_state::UNALLOCATED));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col, *expected);
}

class ListsZeroLengthColumnTest : public ColumnFactoryTest {
 protected:
  using StructsCW = cudf::test::structs_column_wrapper;
  StructsCW make_test_structs_column(cudf::test::fixed_width_column_wrapper<int32_t> field1,
                                     cudf::test::strings_column_wrapper field2,
                                     cudf::test::lists_column_wrapper<int32_t> field3)
  {
    return StructsCW{field1, field2, field3};
  }
};

TEST_F(ListsZeroLengthColumnTest, MixedTypes)
{
  using FCW      = cudf::test::fixed_width_column_wrapper<int32_t>;
  using StringCW = cudf::test::strings_column_wrapper;
  using LCW      = cudf::test::lists_column_wrapper<int32_t>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::offset_type>;
  {
    auto s   = cudf::make_list_scalar(FCW{1, 2, 3});
    auto got = cudf::make_column_from_scalar(*s, 0);
    auto expected =
      cudf::make_lists_column(0,
                              offset_t{}.release(),
                              FCW{}.release(),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, *expected);
  }

  {
    auto s      = cudf::make_list_scalar(LCW{LCW{1, 2, 3}, LCW{}, LCW{5, 6}});
    auto got    = cudf::make_column_from_scalar(*s, 0);
    auto nested = cudf::make_lists_column(0,
                                          offset_t{}.release(),
                                          FCW{}.release(),
                                          0,
                                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    auto expected =
      cudf::make_lists_column(0,
                              offset_t{}.release(),
                              std::move(nested),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, *expected);
  }

  {
    auto s = cudf::make_list_scalar(
      this->make_test_structs_column({1, 2, 3}, StringCW({"x", "", "y"}), LCW{{5, 6}, {}, {7}}));
    auto got = cudf::make_column_from_scalar(*s, 0);

    std::vector<std::unique_ptr<cudf::column>> children;
    children.emplace_back(FCW{}.release());
    children.emplace_back(StringCW{}.release());
    children.emplace_back(LCW{}.release());
    auto nested = cudf::make_structs_column(
      0, std::move(children), 0, cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    auto expected =
      cudf::make_lists_column(0,
                              offset_t{}.release(),
                              std::move(nested),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, *expected);
  }
}

void struct_from_scalar(bool is_valid)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  cudf::test::fixed_width_column_wrapper<int> col0{1};
  cudf::test::strings_column_wrapper col1{"abc"};
  cudf::test::lists_column_wrapper<int> col2{{1, 2, 3}};
  cudf::test::lists_column_wrapper<int> col3{LCW{}};

  std::vector<cudf::column_view> src_children({col0, col1, col2, col3});
  auto value = cudf::struct_scalar(src_children, is_valid);
  cudf::test::structs_column_wrapper struct_col({col0, col1, col2, col3}, {is_valid});

  auto const num_rows = 32;
  auto result         = cudf::make_column_from_scalar(value, num_rows);

  // generate a column of size num_rows
  std::vector<cudf::column_view> cols;
  auto iter = thrust::make_counting_iterator(0);
  std::transform(iter, iter + num_rows, std::back_inserter(cols), [&](int i) {
    return static_cast<cudf::column_view>(struct_col);
  });
  auto expected = cudf::concatenate(cols);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TEST_F(ColumnFactoryTest, FromStructScalar) { struct_from_scalar(true); }

TEST_F(ColumnFactoryTest, FromStructScalarNull) { struct_from_scalar(false); }
