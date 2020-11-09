/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <cudf/utilities/error.hpp>

#include <cudf_test/base_fixture.hpp>

#include <cstring>

TEST(ExpectsTest, FalseCondition)
{
  EXPECT_THROW(CUDF_EXPECTS(false, "condition is false"), cudf::logic_error);
}

TEST(ExpectsTest, TrueCondition) { EXPECT_NO_THROW(CUDF_EXPECTS(true, "condition is true")); }

TEST(ExpectsTest, TryCatch)
{
  CUDF_EXPECT_THROW_MESSAGE(CUDF_EXPECTS(false, "test reason"), "test reason");
}

TEST(CudaTryTest, Error)
{
  CUDA_EXPECT_THROW_MESSAGE(CUDA_TRY(cudaErrorLaunchFailure),
                            "cudaErrorLaunchFailure unspecified launch failure");
}
TEST(CudaTryTest, Success) { EXPECT_NO_THROW(CUDA_TRY(cudaSuccess)); }

TEST(CudaTryTest, TryCatch)
{
  CUDA_EXPECT_THROW_MESSAGE(CUDA_TRY(cudaErrorMemoryAllocation),
                            "cudaErrorMemoryAllocation out of memory");
}

TEST(StreamCheck, success) { EXPECT_NO_THROW(CHECK_CUDA(0)); }

namespace {
// Some silly kernel that will cause an error
void __global__ test_kernel(int* data) { data[threadIdx.x] = threadIdx.x; }
}  // namespace

// In a release build and without explicit synchronization, CHECK_CUDA may
// or may not fail on erroneous asynchronous CUDA calls. Invoke
// cudaStreamSynchronize to guarantee failure on error. In a non-release build,
// CHECK_CUDA deterministically fails on erroneous asynchronous CUDA
// calls.
TEST(StreamCheck, FailedKernel)
{
  cudaStream_t stream;
  CUDA_TRY(cudaStreamCreate(&stream));
  int a;
  test_kernel<<<0, 0, 0, stream>>>(&a);
#ifdef NDEBUG
  CUDA_TRY(cudaStreamSynchronize(stream));
#endif
  EXPECT_THROW(CHECK_CUDA(stream), cudf::cuda_error);
  CUDA_TRY(cudaStreamDestroy(stream));
}

TEST(StreamCheck, CatchFailedKernel)
{
  cudaStream_t stream;
  CUDA_TRY(cudaStreamCreate(&stream));
  int a;
  test_kernel<<<0, 0, 0, stream>>>(&a);
#ifndef NDEBUG
  CUDA_TRY(cudaStreamSynchronize(stream));
#endif
  CUDA_EXPECT_THROW_MESSAGE(CHECK_CUDA(stream),
                            "cudaErrorInvalidConfiguration "
                            "invalid configuration argument");
  CUDA_TRY(cudaStreamDestroy(stream));
}

__global__ void assert_false_kernel() { release_assert(false && "this kernel should die"); }

__global__ void assert_true_kernel() { release_assert(true && "this kernel should live"); }

TEST(ReleaseAssertDeathTest, release_assert_false)
{
  testing::FLAGS_gtest_death_test_style = "threadsafe";

  auto call_kernel = []() {
    assert_false_kernel<<<1, 1>>>();

    // Kernel should fail with `cudaErrorAssert`
    // This error invalidates the current device context, so we need to kill
    // the current process. Running with EXPECT_DEATH spawns a new process for
    // each attempted kernel launch
    if (cudaErrorAssert == cudaDeviceSynchronize()) { std::abort(); }

    // If we reach this point, the release_assert didn't work so we exit normally, which will cause
    // EXPECT_DEATH to fail.
  };

  EXPECT_DEATH(call_kernel(), "this kernel should die");
}

TEST(ReleaseAssert, release_assert_true)
{
  assert_true_kernel<<<1, 1>>>();
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

// These tests don't use CUDF_TEST_PROGRAM_MAIN because :
// 1.) They don't need the RMM Pool
// 2.) The RMM Pool interferes with the death test
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
