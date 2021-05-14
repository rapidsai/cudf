/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <limits>
#include <vector>

using namespace numeric;

struct FixedPointTest : public cudf::test::BaseFixture {
};

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

using RepresentationTypes = ::testing::Types<int32_t, int64_t>;

TYPED_TEST_CASE(FixedPointTestBothReps, RepresentationTypes);

TYPED_TEST(FixedPointTestBothReps, DecimalXXThrust)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  std::vector<decimalXX> vec1(1000);
  std::vector<int32_t> vec2(1000);

  std::iota(std::begin(vec1), std::end(vec1), decimalXX{0, scale_type{-2}});
  std::iota(std::begin(vec2), std::end(vec2), 0);

  auto const res1 =
    thrust::reduce(std::cbegin(vec1), std::cend(vec1), decimalXX{0, scale_type{-2}});

  auto const res2 = std::accumulate(std::cbegin(vec2), std::cend(vec2), 0);

  EXPECT_EQ(static_cast<int32_t>(res1), res2);

  std::vector<int32_t> vec3(vec1.size());

  thrust::transform(std::cbegin(vec1), std::cend(vec1), std::begin(vec3), [](auto const& e) {
    return static_cast<int32_t>(e);
  });

  EXPECT_EQ(vec2, vec3);
}

struct cast_to_int32_fn {
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;
  int32_t __host__ __device__ operator()(decimal32 fp) { return static_cast<int32_t>(fp); }
};

TEST_F(FixedPointTest, DecimalXXThrustOnDevice)
{
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

  std::vector<decimal32> vec1(1000, decimal32{1, scale_type{-2}});
  auto d_vec1 = cudf::detail::make_device_uvector_sync(vec1);

  auto const sum = thrust::reduce(
    rmm::exec_policy(), std::cbegin(d_vec1), std::cend(d_vec1), decimal32{0, scale_type{-2}});

  EXPECT_EQ(static_cast<int32_t>(sum), 1000);

  // TODO: Once nvbugs/1990211 is fixed (ExclusiveSum initial_value = 0 bug)
  //       change inclusive scan to run on device (avoid copying to host)
  thrust::inclusive_scan(std::cbegin(vec1), std::cend(vec1), std::begin(vec1));

  d_vec1 = cudf::detail::make_device_uvector_sync(vec1);

  std::vector<int32_t> vec2(1000);
  std::iota(std::begin(vec2), std::end(vec2), 1);

  auto const res1 = thrust::reduce(
    rmm::exec_policy(), std::cbegin(d_vec1), std::cend(d_vec1), decimal32{0, scale_type{-2}});

  auto const res2 = std::accumulate(std::cbegin(vec2), std::cend(vec2), 0);

  EXPECT_EQ(static_cast<int32_t>(res1), res2);

  rmm::device_uvector<int32_t> d_vec3(1000, rmm::cuda_stream_default);

  thrust::transform(rmm::exec_policy(),
                    std::cbegin(d_vec1),
                    std::cend(d_vec1),
                    std::begin(d_vec3),
                    cast_to_int32_fn{});

  auto vec3 = cudf::detail::make_std_vector_sync(d_vec3);

  EXPECT_EQ(vec2, vec3);
}
