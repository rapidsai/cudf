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

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <thrust/sequence.h>
#include <random>

#include <gmock/gmock.h>

template <typename T>
struct TypedScalarDeviceViewTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(TypedScalarDeviceViewTest, cudf::test::NumericTypes);


template <typename T,
  typename ScalarDeviceViewType>
__global__ void test_value(ScalarDeviceViewType s, T value, bool* result) {
  *result = (s.value() == value);
}

TYPED_TEST(TypedScalarDeviceViewTest, Value) {
  TypeParam value{7};
  cudf::experimental::type_to_scalar_type<TypeParam> s(value);
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  rmm::device_scalar<bool> result;

  test_value<<<1, 1>>>(scalar_device_view, value, result.get());
  CUDA_CHECK_LAST();

  EXPECT_TRUE(result.value());
}

template <typename ScalarDeviceViewType>
__global__ void test_null(ScalarDeviceViewType s, bool* result) {
  *result = s.is_valid();
}

TYPED_TEST(TypedScalarDeviceViewTest, ConstructNull) {
  TypeParam value = 5;
  cudf::experimental::type_to_scalar_type<TypeParam> s(value, false);
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  rmm::device_scalar<bool> result;

  test_null<<<1, 1>>>(scalar_device_view, result.get());
  CUDA_CHECK_LAST();

  EXPECT_FALSE(result.value());
}

template <typename T,
  typename ScalarDeviceViewType>
__global__ void test_setvalue(ScalarDeviceViewType s, T value) {
  s.set_value(value);
  s.set_valid();
}

TYPED_TEST(TypedScalarDeviceViewTest, SetValue) {
  TypeParam value = 9;
  cudf::experimental::type_to_scalar_type<TypeParam> s;
  auto scalar_device_view = cudf::get_scalar_device_view(s);

  test_setvalue<<<1, 1>>>(scalar_device_view, value);
  CUDA_CHECK_LAST();

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.value());
}

template <typename ScalarDeviceViewType>
__global__ void test_setnull(ScalarDeviceViewType s) {
  s.set_null();
}

TYPED_TEST(TypedScalarDeviceViewTest, SetNull) {
  cudf::experimental::type_to_scalar_type<TypeParam> s;
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  s.set_valid();
  EXPECT_TRUE(s.is_valid());

  test_setnull<<<1, 1>>>(scalar_device_view);
  CUDA_CHECK_LAST();

  EXPECT_FALSE(s.is_valid());
}


struct StringScalarDeviceViewTest : public cudf::test::BaseFixture {};


__global__ void test_string_value(cudf::string_scalar_device_view s, 
                                  const char* value, cudf::size_type size,
                                  bool* result)
{
  *result = (s.value() == cudf::string_view(value, size));
}

TEST_F(StringScalarDeviceViewTest, Value) {
  std::string value("test string");
  cudf::string_scalar s(value);
  
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  rmm::device_scalar<bool> result;
  rmm::device_vector<char> value_v(value.begin(), value.end());

  test_string_value<<<1, 1>>>(scalar_device_view, value_v.data().get(),
                              value.size(), result.get());
  CUDA_CHECK_LAST();

  EXPECT_TRUE(result.value());  
}
