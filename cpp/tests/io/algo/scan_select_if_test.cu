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
  inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs)
  {
    // printf("b(%i) t(%i) %i + %i\n", blockIdx.x, threadIdx.x, lhs, rhs);
    return lhs + rhs;
  }
  inline __device__ bool operator()(uint32_t value)
  {
    // printf("b(%i) t(%i) pred(%i)\n", blockIdx.x, threadIdx.x, value);
    return true;
  }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto op = simple_op{};

  const uint32_t size = 4096;

  thrust::host_vector<uint32_t> h_result = scan_select_if(input, input + size, op, op);

  // cudaMemcpy(h_result.data(), d_result.data(), d_result.size(), cudaMemcpyDeviceToHost);

  ASSERT_EQ(static_cast<uint32_t>(size), h_result.size());

  for (uint32_t i = 0; i < h_result.size(); i++) {
    ASSERT_EQ(static_cast<uint32_t>(i + 1), h_result[i]);
  }

  // FAIL();
}

CUDF_TEST_PROGRAM_MAIN()
