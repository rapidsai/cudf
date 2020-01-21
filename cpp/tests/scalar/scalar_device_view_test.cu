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

TYPED_TEST_CASE(TypedScalarDeviceViewTest, cudf::test::FixedWidthTypes);


template <typename ScalarDeviceViewType>
__global__ void test_set_value(ScalarDeviceViewType s, 
                               ScalarDeviceViewType s1) {
  s1.set_value(s.value());
  s1.set_valid(true);
}

template <typename ScalarDeviceViewType>
__global__ void test_value(ScalarDeviceViewType s, 
                           ScalarDeviceViewType s1, bool *result) {
  *result = (s.value() == s1.value());
}

TYPED_TEST(TypedScalarDeviceViewTest, Value) {
  TypeParam value{7};
  cudf::experimental::scalar_type_t<TypeParam> s(value);
  cudf::experimental::scalar_type_t<TypeParam> s1;

  auto scalar_device_view = cudf::get_scalar_device_view(s);
  auto scalar_device_view1 = cudf::get_scalar_device_view(s1);
  rmm::device_scalar<bool> result;

  test_set_value<<<1, 1>>>(scalar_device_view, scalar_device_view1);
  CHECK_CUDA(0);

  EXPECT_EQ(s1.value(), value);
  EXPECT_TRUE(s1.is_valid());

  test_value<<<1, 1>>>(scalar_device_view, scalar_device_view1, result.data());
  CHECK_CUDA(0);

  EXPECT_TRUE(result.value());
}

template <typename ScalarDeviceViewType>
__global__ void test_null(ScalarDeviceViewType s, bool* result) {
  *result = s.is_valid();
}

TYPED_TEST(TypedScalarDeviceViewTest, ConstructNull) {
  TypeParam value = 5;
  cudf::experimental::scalar_type_t<TypeParam> s(value, false);
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  rmm::device_scalar<bool> result;

  test_null<<<1, 1>>>(scalar_device_view, result.data());
  CHECK_CUDA(0);

  EXPECT_FALSE(result.value());
}

template <typename ScalarDeviceViewType>
__global__ void test_setnull(ScalarDeviceViewType s) {
  s.set_valid(false);
}

TYPED_TEST(TypedScalarDeviceViewTest, SetNull) {
  cudf::experimental::scalar_type_t<TypeParam> s;
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  s.set_valid(true);
  EXPECT_TRUE(s.is_valid());

  test_setnull<<<1, 1>>>(scalar_device_view);
  CHECK_CUDA(0);

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
                              value.size(), result.data());
  CHECK_CUDA(0);

  EXPECT_TRUE(result.value());  
}
