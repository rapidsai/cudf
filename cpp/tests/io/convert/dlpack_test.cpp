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
#include <cudf/column/dlpack.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <dlpack/dlpack.h>

using namespace cudf::test;

template <typename T>
void validate_dtype(DLDataType const& dtype)
{
  switch (dtype.code) {
    case kDLInt:
      EXPECT_TRUE(std::is_integral<T>::value && std::is_signed<T>::value);
      break;
    case kDLUInt:
      EXPECT_TRUE(std::is_integral<T>::value && std::is_unsigned<T>::value);
      break;
    case kDLFloat:
      EXPECT_TRUE(std::is_floating_point<T>::value);
      break;
    default: FAIL();
  }
  EXPECT_EQ(1, dtype.lanes);
  EXPECT_EQ(sizeof(T) * 8, dtype.bits);
}

class DLPackUntypedTests : public BaseFixture {};

TEST_F(DLPackUntypedTests, EmptyTableToDlpack)
{
  cudf::table_view empty(std::vector<cudf::column_view>{});
  EXPECT_EQ(nullptr, cudf::to_dlpack(empty));
}

TEST_F(DLPackUntypedTests, MultipleTypesToDlpack)
{
  fixed_width_column_wrapper<int16_t> col1({1, 2, 3, 4});
  fixed_width_column_wrapper<int32_t> col2({1, 2, 3, 4});
  cudf::table_view input({col1, col2});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, StringTypeToDlpack)
{
  strings_column_wrapper col({"foo", "bar", "baz"});
  cudf::table_view input({col});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

template <typename T>
class DLPackTimestampTests : public BaseFixture {};

TYPED_TEST_CASE(DLPackTimestampTests, TimestampTypes);

TYPED_TEST(DLPackTimestampTests, TimestampTypesToDlpack)
{
  fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4});
  cudf::table_view input({col});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

template <typename T>
class DLPackNumericTests : public BaseFixture {};

TYPED_TEST_CASE(DLPackNumericTests, NumericTypes);

TYPED_TEST(DLPackNumericTests, ToDlpack1D)
{
  fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4});
  auto const col_view = static_cast<cudf::column_view>(col);
  cudf::table_view input({col});
  auto result = cudf::to_dlpack(input);

  auto const& tensor = result->dl_tensor;
  validate_dtype<TypeParam>(tensor.dtype);
  EXPECT_EQ(kDLGPU, tensor.ctx.device_type);
  EXPECT_EQ(1, tensor.ndim);
  EXPECT_EQ(0, tensor.byte_offset);
  EXPECT_EQ(nullptr, tensor.strides);

  EXPECT_NE(nullptr, tensor.data);
  EXPECT_NE(nullptr, tensor.shape);

  // Verify that data matches input column
  constexpr cudf::data_type type{cudf::experimental::type_to_id<TypeParam>()};
  cudf::column_view const result_view(type, tensor.shape[0], tensor.data);
  expect_columns_equal(col_view, result_view);

  // Free the managed tensor
  result->deleter(result);
}

TYPED_TEST(DLPackNumericTests, ToDlpack2D)
{
  std::vector<fixed_width_column_wrapper<TypeParam>> cols;
  cols.push_back(fixed_width_column_wrapper<TypeParam>{1, 2, 3, 4});
  cols.push_back(fixed_width_column_wrapper<TypeParam>{4, 5, 6, 7});

  std::vector<cudf::column_view> col_views;
  std::transform(cols.begin(), cols.end(), std::back_inserter(col_views),
    [](auto const& col) { return static_cast<cudf::column_view>(col); });

  cudf::table_view input(col_views);
  auto result = cudf::to_dlpack(input);

  auto const& tensor = result->dl_tensor;
  validate_dtype<TypeParam>(tensor.dtype);
  EXPECT_EQ(kDLGPU, tensor.ctx.device_type);
  EXPECT_EQ(2, tensor.ndim);
  EXPECT_EQ(0, tensor.byte_offset);

  EXPECT_NE(nullptr, tensor.data);
  EXPECT_NE(nullptr, tensor.shape);
  EXPECT_NE(nullptr, tensor.strides);

  EXPECT_EQ(1, tensor.strides[0]);
  EXPECT_EQ(tensor.shape[0], tensor.strides[1]);

  // Verify that data matches input columns
  cudf::size_type offset{0};
  for (auto const& col : input) {
    constexpr cudf::data_type type{cudf::experimental::type_to_id<TypeParam>()};
    cudf::column_view const result_view(type, tensor.shape[0], tensor.data, nullptr, 0, offset);
    expect_columns_equal(col, result_view);
    offset += tensor.strides[1];
  }

  // Free the managed tensor
  result->deleter(result);
}
