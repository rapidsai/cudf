#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "gtest/gtest.h"
#include "scan_select_if.cuh"

#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/constant_iterator.h>

class InclusiveCopyIfTest : public cudf::test::BaseFixture {
};

struct simple_op {
  inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs) { return rhs + lhs; }
  inline __device__ bool operator()(uint32_t value)
  {
    printf("pred_op(%i)\n", value);
    return true;
  }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto op = simple_op{};

  auto d_result = scan_select_if(input, input + 256, op, op);

  auto h_result = thrust::host_vector<uint32_t>(d_result.size());

  cudaMemcpy(h_result.data(), d_result.data(), d_result.size(), cudaMemcpyDeviceToHost);

  for (auto value : h_result) { EXPECT_EQ(static_cast<uint32_t>(-1), value); }

  FAIL();
}

CUDF_TEST_PROGRAM_MAIN()
