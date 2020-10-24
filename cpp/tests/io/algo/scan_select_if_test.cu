#include <algorithm>
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
  inline constexpr uint32_t operator()(uint32_t lhs, uint32_t rhs) { return lhs + rhs; }
  inline constexpr bool operator()(uint32_t value) { return false; }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto op = simple_op{};

  // const uint32_t size = 1 << 24;
  const uint32_t size = 1 << 17;

  auto d_result = scan_select_if(input, input + size, op, op);

  thrust::host_vector<uint32_t> h_result(d_result.size());

  cudaMemcpy(
    h_result.data(), d_result.data(), sizeof(uint32_t) * d_result.size(), cudaMemcpyDeviceToHost);

  // 4096 / 3 = 1365.333...
  ASSERT_EQ(static_cast<uint32_t>(0), h_result.size());

  for (uint32_t i = 0; i < h_result.size(); i++) {  //
    // ASSERT_EQ(static_cast<uint32_t>(i * 3 + 3), h_result[i]);
    // EXPECT_EQ(static_cast<uint32_t>(-1), h_result[i]);
    // EXPECT_EQ(static_cast<uint32_t>(i + 1), h_result[i]);
    // EXPECT_EQ(static_cast<uint32_t>(i * 2 + 2), h_result[i]);
  }

  FAIL();
}

struct successive_capitalization_state {
  char curr;
  char prev;
};

struct successive_capitalization_op {
  inline constexpr successive_capitalization_state operator()(  //
    successive_capitalization_state lhs,
    successive_capitalization_state rhs)
  {
    return {rhs.curr, lhs.curr};
  }

  inline constexpr bool is_capital(char value)
  {                          //
    return value >= 'A' and  //
           value <= 'Z';
  }

  inline __device__ bool operator()(successive_capitalization_state value)
  {
    printf("p(%c) c(%c)\n", value.prev, value.curr);
    return is_capital(value.prev) and  //
           is_capital(value.curr);
  }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIfFloat)
{
  // auto input_str = std::string("AbcDeFGLiJKlMnoP");

  // auto input = rmm::device_vector<successive_capitalization_state>(input_str.size());

  // std::transform(input_str.begin(),  //
  //                input_str.end(),
  //                input.begin(),
  //                [](char value) { return successive_capitalization_state{value}; });

  // auto op = successive_capitalization_op{};

  // scan_select_if(  //
  //   input.begin(),
  //   input.end(),
  //   op,
  //   op);

  // thrust::host_vector<uint32_t> h_result = scan_select_if(  //
  //   input.begin(),
  //   input.end(),
  //   op,
  //   op);

  // ASSERT_EQ(static_cast<uint32_t>(2), h_result.size());

  // ASSERT_EQ(static_cast<uint32_t>(6), h_result[0]);
  // ASSERT_EQ(static_cast<uint32_t>(10), h_result[1]);
}

// struct csv_row_start_op {
//   inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs) { return lhs + rhs; }
//   inline __device__ bool operator()(uint32_t value) { return true; }
// };

// TEST_F(InclusiveCopyIfTest, CanDetectCsvRowStart) {}

CUDF_TEST_PROGRAM_MAIN()
