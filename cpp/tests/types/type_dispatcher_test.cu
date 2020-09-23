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
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/device_vector.h>

struct DispatcherTest : public cudf::test::BaseFixture {
};

template <typename T>
struct TypedDispatcherTest : public DispatcherTest {
};

TYPED_TEST_CASE(TypedDispatcherTest, cudf::test::AllTypes);

namespace {
template <typename Expected>
struct type_tester {
  template <typename Dispatched>
  bool operator()()
  {
    return std::is_same<Expected, Dispatched>::value;
  }
};
}  // namespace

TYPED_TEST(TypedDispatcherTest, TypeToId)
{
  EXPECT_TRUE(cudf::type_dispatcher(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                    type_tester<TypeParam>{}));
}

namespace {
struct verify_dispatched_type {
  template <typename T>
  __host__ __device__ bool operator()(cudf::type_id id)
  {
    return id == cudf::type_to_id<T>();
  }
};

__global__ void dispatch_test_kernel(cudf::type_id id, bool* d_result)
{
  if (0 == threadIdx.x + blockIdx.x * blockDim.x)
    *d_result = cudf::type_dispatcher(cudf::data_type{id}, verify_dispatched_type{}, id);
}
}  // namespace

TYPED_TEST(TypedDispatcherTest, DeviceDispatch)
{
  thrust::device_vector<bool> result(1, false);
  dispatch_test_kernel<<<1, 1>>>(cudf::type_to_id<TypeParam>(), result.data().get());
  CUDA_TRY(cudaDeviceSynchronize());
  EXPECT_EQ(true, result[0]);
}

struct IdDispatcherTest : public DispatcherTest, public testing::WithParamInterface<cudf::type_id> {
};

INSTANTIATE_TEST_CASE_P(TestAllIds, IdDispatcherTest, testing::ValuesIn(cudf::test::all_type_ids));

TEST_P(IdDispatcherTest, IdToType)
{
  auto t = GetParam();
  EXPECT_TRUE(cudf::type_dispatcher(cudf::data_type{t}, verify_dispatched_type{}, t));
}

template <typename T>
struct TypedDoubleDispatcherTest : public DispatcherTest {
};

TYPED_TEST_CASE(TypedDoubleDispatcherTest, cudf::test::AllTypes);

namespace {
template <typename Expected1, typename Expected2>
struct two_type_tester {
  template <typename Dispatched1, typename Dispatched2>
  bool operator()()
  {
    return std::is_same<Expected1, Dispatched1>::value &&
           std::is_same<Expected2, Dispatched2>::value;
  }
};
}  // namespace

TYPED_TEST(TypedDoubleDispatcherTest, TypeToId)
{
  EXPECT_TRUE(cudf::double_type_dispatcher(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                                           two_type_tester<TypeParam, TypeParam>{}));
}

namespace {
struct verify_double_dispatched_type {
  template <typename T1, typename T2>
  __host__ __device__ bool operator()(cudf::type_id id1, cudf::type_id id2)
  {
    return id1 == cudf::type_to_id<T1>() && id2 == cudf::type_to_id<T2>();
  }
};

__global__ void double_dispatch_test_kernel(cudf::type_id id1, cudf::type_id id2, bool* d_result)
{
  if (0 == threadIdx.x + blockIdx.x * blockDim.x)
    *d_result = cudf::double_type_dispatcher(
      cudf::data_type{id1}, cudf::data_type{id2}, verify_double_dispatched_type{}, id1, id2);
}
}  // namespace

TYPED_TEST(TypedDoubleDispatcherTest, DeviceDoubleDispatch)
{
  thrust::device_vector<bool> result(1, false);
  double_dispatch_test_kernel<<<1, 1>>>(
    cudf::type_to_id<TypeParam>(), cudf::type_to_id<TypeParam>(), result.data().get());
  CUDA_TRY(cudaDeviceSynchronize());
  EXPECT_EQ(true, result[0]);
}

struct IdDoubleDispatcherTest : public DispatcherTest,
                                public testing::WithParamInterface<cudf::type_id> {
};

INSTANTIATE_TEST_CASE_P(TestAllIds,
                        IdDoubleDispatcherTest,
                        testing::ValuesIn(cudf::test::all_type_ids));

TEST_P(IdDoubleDispatcherTest, IdToType)
{
  // Test double-dispatch of all types using the same type for both dispatches
  auto t = GetParam();
  EXPECT_TRUE(cudf::double_type_dispatcher(
    cudf::data_type{t}, cudf::data_type{t}, verify_double_dispatched_type{}, t, t));
}

struct IdFixedDoubleDispatcherTest : public DispatcherTest,
                                     public testing::WithParamInterface<cudf::type_id> {
};

INSTANTIATE_TEST_CASE_P(TestAllIds,
                        IdFixedDoubleDispatcherTest,
                        testing::ValuesIn(cudf::test::all_type_ids));

TEST_P(IdFixedDoubleDispatcherTest, IdToType)
{
  // Test double-dispatch of all types against one fixed type, in each direction
  auto t = GetParam();
  EXPECT_TRUE(cudf::double_type_dispatcher(cudf::data_type{t},
                                           cudf::data_type{cudf::type_to_id<float>()},
                                           verify_double_dispatched_type{},
                                           t,
                                           cudf::type_to_id<float>()));
  EXPECT_TRUE(cudf::double_type_dispatcher(cudf::data_type{cudf::type_to_id<float>()},
                                           cudf::data_type{t},
                                           verify_double_dispatched_type{},
                                           cudf::type_to_id<float>(),
                                           t));
}

CUDF_TEST_PROGRAM_MAIN()
