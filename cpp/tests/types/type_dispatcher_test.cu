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

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list.hpp>
#include <tests/utilities/typed_tests.hpp>

#include <thrust/device_vector.h>

#include <gmock/gmock.h>

struct DispatcherTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedDispatcherTest : public DispatcherTest {};

TYPED_TEST_CASE(TypedDispatcherTest, cudf::test::AllTypes);

namespace {
template <typename Expected>
struct type_tester {
  template <typename Dispatched>
  bool operator()() {
    return std::is_same<Expected, Dispatched>::value;
  }
};
}  // namespace

TYPED_TEST(TypedDispatcherTest, TypeToId) {
  EXPECT_TRUE(cudf::exp::type_dispatcher(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()},
      type_tester<TypeParam>{}));
}

namespace {
struct verify_dispatched_type {
  template <typename T>
  __host__ __device__ bool operator()(cudf::type_id id) {
    return id == cudf::exp::type_to_id<T>();
  }
};

__global__ void dispatch_test_kernel(cudf::type_id id, bool* d_result) {
  if (0 == threadIdx.x + blockIdx.x * blockDim.x)
    *d_result = cudf::exp::type_dispatcher(cudf::data_type{id},
                                           verify_dispatched_type{}, id);
}
}  // namespace

TYPED_TEST(TypedDispatcherTest, DeviceDispatch) {
  thrust::device_vector<bool> result(1, false);
  dispatch_test_kernel<<<1, 1>>>(cudf::exp::type_to_id<TypeParam>(),
                                 result.data().get());
  cudaDeviceSynchronize();
  EXPECT_EQ(true, result[0]);
}

struct IdDispatcherTest : public DispatcherTest,
                          public testing::WithParamInterface<cudf::type_id> {};

INSTANTIATE_TEST_CASE_P(TestAllIds, IdDispatcherTest,
                        testing::ValuesIn(cudf::test::all_type_ids));

TEST_P(IdDispatcherTest, IdToType) {
  auto t = GetParam();
  EXPECT_TRUE(cudf::exp::type_dispatcher(cudf::data_type{t},
                                         verify_dispatched_type{}, t));
}
