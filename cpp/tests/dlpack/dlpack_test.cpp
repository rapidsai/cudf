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
#include <cudf/dlpack.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>

#include <dlpack/dlpack.h>

using namespace cudf::test;

struct dlpack_deleter {
  void operator()(DLManagedTensor* tensor) { tensor->deleter(tensor); }
};

using unique_managed_tensor = std::unique_ptr<DLManagedTensor, dlpack_deleter>;

template <typename T>
DLDataType get_dtype()
{
  uint8_t const bits{sizeof(T) * 8};
  uint16_t const lanes{1};
  if (std::is_floating_point<T>::value) {
    return DLDataType{kDLFloat, bits, lanes};
  } else if (std::is_signed<T>::value) {
    return DLDataType{kDLInt, bits, lanes};
  } else if (std::is_unsigned<T>::value) {
    return DLDataType{kDLUInt, bits, lanes};
  } else {
    static_assert(true, "unsupported type");
  }
}

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

TEST_F(DLPackUntypedTests, EmptyColsToDlpack)
{
  fixed_width_column_wrapper<int32_t> col1({});
  fixed_width_column_wrapper<int32_t> col2({});
  cudf::table_view input({col1, col2});
  EXPECT_EQ(nullptr, cudf::to_dlpack(input));
}

TEST_F(DLPackUntypedTests, NullTensorFromDlpack)
{
  EXPECT_THROW(cudf::from_dlpack(nullptr), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, MultipleTypesToDlpack)
{
  fixed_width_column_wrapper<int16_t> col1({1, 2, 3, 4});
  fixed_width_column_wrapper<int32_t> col2({1, 2, 3, 4});
  cudf::table_view input({col1, col2});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, InvalidNullsToDlpack)
{
  fixed_width_column_wrapper<int32_t> col1({1, 2, 3, 4});
  fixed_width_column_wrapper<int32_t> col2({1, 2, 3, 4}, {1, 0, 1, 1});
  cudf::table_view input({col1, col2});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, StringTypeToDlpack)
{
  strings_column_wrapper col({"foo", "bar", "baz"});
  cudf::table_view input({col});
  EXPECT_THROW(cudf::to_dlpack(input), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedDeviceTypeFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported device type
  tensor->dl_tensor.ctx.device_type = kDLOpenCL;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, InvalidDeviceIdFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof the wrong device ID
  tensor->dl_tensor.ctx.device_id += 1;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedDimsFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported number of dims
  tensor->dl_tensor.ndim = 3;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, TooManyRowsFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof too many rows
  constexpr int64_t max_size_type{std::numeric_limits<int32_t>::max()};
  tensor->dl_tensor.shape[0] = max_size_type + 1;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, TooManyColsFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col1({1, 2, 3, 4});
  fixed_width_column_wrapper<int32_t> col2({5, 6, 7, 8});
  cudf::table_view input({col1, col2});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof too many cols
  constexpr int64_t max_size_type{std::numeric_limits<int32_t>::max()};
  tensor->dl_tensor.shape[1] = max_size_type + 1;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, InvalidTypeFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an invalid data type
  tensor->dl_tensor.dtype.code = 3;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedIntBitsizeFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported bitsize
  tensor->dl_tensor.dtype.bits = 7;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedFloatBitsizeFromDlpack)
{
  fixed_width_column_wrapper<float> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported bitsize
  tensor->dl_tensor.dtype.bits = 7;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
}

TEST_F(DLPackUntypedTests, UnsupportedLanesFromDlpack)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Spoof an unsupported number of lanes
  tensor->dl_tensor.dtype.lanes = 2;
  EXPECT_THROW(cudf::from_dlpack(tensor.get()), cudf::logic_error);
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
  // Test nullable column with no nulls
  fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4}, {1, 1, 1, 1});
  auto const col_view = static_cast<cudf::column_view>(col);
  EXPECT_FALSE(col_view.has_nulls());
  EXPECT_TRUE(col_view.nullable());

  cudf::table_view input({col});
  unique_managed_tensor result(cudf::to_dlpack(input));

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
  cudf::column_view const result_view(type, tensor.shape[0], tensor.data,
    col_view.null_mask());
  expect_columns_equal(col_view, result_view);
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
  unique_managed_tensor result(cudf::to_dlpack(input));

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
}

TYPED_TEST(DLPackNumericTests, FromDlpack1D)
{
  // Use to_dlpack to generate an input tensor
  fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4});
  cudf::table_view input({col});
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Verify that from_dlpack(to_dlpack(input)) == input
  auto result = cudf::from_dlpack(tensor.get());
  expect_tables_equal(input, result->view());
}

TYPED_TEST(DLPackNumericTests, FromDlpack2D)
{
  // Use to_dlpack to generate an input tensor
  std::vector<fixed_width_column_wrapper<TypeParam>> cols;
  cols.push_back(fixed_width_column_wrapper<TypeParam>{1, 2, 3, 4});
  cols.push_back(fixed_width_column_wrapper<TypeParam>{4, 5, 6, 7});

  std::vector<cudf::column_view> col_views;
  std::transform(cols.begin(), cols.end(), std::back_inserter(col_views),
    [](auto const& col) { return static_cast<cudf::column_view>(col); });

  cudf::table_view input(col_views);
  unique_managed_tensor tensor(cudf::to_dlpack(input));

  // Verify that from_dlpack(to_dlpack(input)) == input
  auto result = cudf::from_dlpack(tensor.get());
  expect_tables_equal(input, result->view());
}

TYPED_TEST(DLPackNumericTests, FromDlpackCpu)
{
  // Host buffer with stride > rows and byte_offset > 0
  std::vector<TypeParam> data{0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0};
  uint64_t const offset{sizeof(TypeParam)};
  int64_t shape[2] = {4, 2};
  int64_t strides[2] = {1, 5};

  DLManagedTensor tensor{};
  tensor.dl_tensor.ctx.device_type = kDLCPU;
  tensor.dl_tensor.dtype = get_dtype<TypeParam>();
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.byte_offset = offset;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = strides;
  tensor.dl_tensor.data = data.data();

  fixed_width_column_wrapper<TypeParam> col1({1, 2, 3, 4});
  fixed_width_column_wrapper<TypeParam> col2({5, 6, 7, 8});
  cudf::table_view expected({col1, col2});

  auto result = cudf::from_dlpack(&tensor);
  expect_tables_equal(expected, result->view());
}
