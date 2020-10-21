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
  inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs) { return lhs + rhs; }
  inline __device__ bool operator()(uint32_t value) { return value % 3 == 0; }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto op = simple_op{};

  const uint32_t size = 128;

  thrust::host_vector<uint32_t> h_result = scan_select_if(input, input + size, op, op);

  // cudaMemcpy(h_result.data(), d_result.data(), d_result.size(), cudaMemcpyDeviceToHost);

  // 4096 / 3 = 1365.333...
  // ASSERT_EQ(static_cast<uint32_t>(1365), h_result.size());

  // for (uint32_t i = 0; i < 1365; i++) {  //
  //   EXPECT_EQ(static_cast<uint32_t>(i * 3 + 3), h_result[i]);
  // }

  FAIL();
}

// struct csv_row_start_op {
//   inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs) { return lhs + rhs; }
//   inline __device__ bool operator()(uint32_t value) { return true; }
// };

// TEST_F(InclusiveCopyIfTest, CanDetectCsvRowStart) {}

CUDF_TEST_PROGRAM_MAIN()
