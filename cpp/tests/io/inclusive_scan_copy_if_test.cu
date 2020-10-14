#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "inclusive_scan_copy_if.cuh"

#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

class InclusiveScanCopyIfTest : public cudf::test::BaseFixture {
};

template <typename T>
struct pick_rhs_functor {
  inline constexpr T operator()(T const& lhs, T const& rhs) { return rhs; }
};

template <typename T>
struct find_functor {
  T needle;
  bool invert = false;
  inline constexpr bool operator()(T const& value)
  {
    return invert ? not(value == needle) : value == needle;
  }
};

TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy)
{
  auto input = std::string("00__1___0__0_1__01___0__0_1");
  rmm::device_vector<uint8_t> d_input(input.c_str(), input.c_str() + input.size());

  auto d_output = inclusive_scan_copy_if<uint8_t>(device_span<uint8_t>(d_input),  //
                                                  pick_rhs_functor<uint8_t>(),
                                                  find_functor<uint8_t>{'1', false},
                                                  0);

  thrust::host_vector<uint32_t> h_indices = d_output;

  ASSERT_EQ(static_cast<uint32_t>(4), h_indices.size());

  EXPECT_EQ(static_cast<uint32_t>(4), h_indices[0]);
  EXPECT_EQ(static_cast<uint32_t>(13), h_indices[1]);
  EXPECT_EQ(static_cast<uint32_t>(17), h_indices[2]);
  EXPECT_EQ(static_cast<uint32_t>(26), h_indices[3]);
}

TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy2)
{
  auto input = std::string("0100000100000010010001");
  rmm::device_vector<uint8_t> d_input(input.c_str(), input.c_str() + input.size());

  auto d_output = inclusive_scan_copy_if<uint8_t>(device_span<uint8_t>(d_input),  //
                                                  pick_rhs_functor<uint8_t>(),
                                                  find_functor<uint8_t>{'0', true},
                                                  0);

  thrust::host_vector<uint32_t> h_indices = d_output;

  ASSERT_EQ(static_cast<uint32_t>(5), h_indices.size());

  EXPECT_EQ(static_cast<uint32_t>(1), h_indices[0]);
  EXPECT_EQ(static_cast<uint32_t>(7), h_indices[1]);
  EXPECT_EQ(static_cast<uint32_t>(14), h_indices[2]);
  EXPECT_EQ(static_cast<uint32_t>(17), h_indices[3]);
  EXPECT_EQ(static_cast<uint32_t>(21), h_indices[4]);
}

struct ascend_state {
  int value = std::numeric_limits<int>::max();
  bool did_ascend;
};

struct ascend_reduce_functor {
  inline constexpr ascend_state operator()(ascend_state const& lhs, ascend_state const& rhs)
  {
    return {rhs.value, rhs.value > lhs.value};
  }
};

struct ascend_detect_functor {
  inline constexpr bool operator()(ascend_state const& state) { return state.did_ascend; }
};

TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy3)
{
  auto input       = std::vector<int>{1, 6, 9, 5, 4, 8, 3, 2, 8, 9};
  auto input_state = std::vector<ascend_state>(input.size());

  std::transform(  //
    input.begin(),
    input.end(),
    input_state.begin(),
    [](int value) -> ascend_state { return {value}; });

  rmm::device_vector<ascend_state> d_input_state(input_state.begin(), input_state.end());

  auto d_output = inclusive_scan_copy_if<ascend_state>(d_input_state,  //
                                                       ascend_reduce_functor{},
                                                       ascend_detect_functor{},
                                                       0);

  thrust::host_vector<uint32_t> h_indices = d_output;

  ASSERT_EQ(static_cast<uint32_t>(5), h_indices.size());

  EXPECT_EQ(static_cast<uint32_t>(1), h_indices[0]);
  EXPECT_EQ(static_cast<uint32_t>(2), h_indices[1]);
  EXPECT_EQ(static_cast<uint32_t>(5), h_indices[2]);
  EXPECT_EQ(static_cast<uint32_t>(8), h_indices[3]);
  EXPECT_EQ(static_cast<uint32_t>(9), h_indices[4]);
}

TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy4)
{
  auto input = std::vector<int>(8);
  std::iota(input.begin(), input.end(), 0);

  auto input_state = std::vector<ascend_state>(input.size());

  std::transform(  //
    input.begin(),
    input.end(),
    input_state.begin(),
    [](int value) -> ascend_state { return {value}; });

  rmm::device_vector<ascend_state> d_input_state(input_state.begin(), input_state.end());

  auto d_output = inclusive_scan_copy_if<ascend_state>(d_input_state,  //
                                                       ascend_reduce_functor{},
                                                       ascend_detect_functor{},
                                                       0);

  thrust::host_vector<uint32_t> h_indices = d_output;

  ASSERT_EQ(static_cast<uint32_t>(input.size() - 1), h_indices.size());

  for (uint64_t i = 0; i < input.size() - 1; i++) {
    EXPECT_EQ(static_cast<uint32_t>(i + 1), h_indices[i]);
  }
}

CUDF_TEST_PROGRAM_MAIN()
