#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "gtest/gtest.h"
#include "scan_select_if.cuh"

#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/counting_iterator.h>

class InclusiveCopyIfTest : public cudf::test::BaseFixture {
};

struct simple_op {
  static inline constexpr uint32_t scan(uint32_t lhs, uint32_t rhs) { return rhs - lhs / 2; }
  static inline constexpr bool predicate(uint32_t a) { return a % 3 == 0; }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_counting_iterator<uint32_t>(0);

  auto d_result = scan_select_if(input,  //
                                 input + 10,
                                 simple_op::scan,
                                 simple_op::predicate);

  auto h_result = thrust::host_vector<uint32_t>(d_result.size());

  cudaMemcpy(h_result.data(), d_result.data(), d_result.size(), cudaMemcpyDeviceToHost);

  for (auto value : h_result) { EXPECT_EQ(static_cast<uint32_t>(-1), value); }

  FAIL();
}

CUDF_TEST_PROGRAM_MAIN()
