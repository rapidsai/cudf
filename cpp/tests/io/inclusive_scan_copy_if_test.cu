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

// template <typename T>
// struct pick_rhs_functor {
//   inline constexpr T operator()(T const& lhs, T const& rhs) { return rhs; }
// };

// template <typename T>
// struct find_functor {
//   T needle;
//   bool invert = false;
//   inline constexpr bool operator()(T const& value)
//   {
//     return invert ? not(value == needle) : value == needle;
//   }
// };

// TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy)
// {
//   auto input = std::string("00__1___0__0_1__01___0__0_1");
//   rmm::device_vector<uint8_t> d_input(input.c_str(), input.c_str() + input.size());

//   auto d_output = inclusive_scan_copy_if<uint8_t>(device_span<uint8_t>(d_input),  //
//                                                   pick_rhs_functor<uint8_t>(),
//                                                   find_functor<uint8_t>{'1', false},
//                                                   0);

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   // ASSERT_EQ(static_cast<uint32_t>(4), h_indices.size());

//   // EXPECT_EQ(static_cast<uint32_t>(4), h_indices[0]);
//   // EXPECT_EQ(static_cast<uint32_t>(13), h_indices[1]);
//   // EXPECT_EQ(static_cast<uint32_t>(17), h_indices[2]);
//   // EXPECT_EQ(static_cast<uint32_t>(26), h_indices[3]);

//   for (uint64_t i = 0; i < h_indices.size(); i++) {
//     EXPECT_EQ(static_cast<uint32_t>(-1), h_indices[i]);
//   }
// }

// TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy2)
// {
//   auto input = std::string("0100000100000010010001");
//   rmm::device_vector<uint8_t> d_input(input.c_str(), input.c_str() + input.size());

//   auto d_output = inclusive_scan_copy_if<uint8_t>(device_span<uint8_t>(d_input),  //
//                                                   pick_rhs_functor<uint8_t>(),
//                                                   find_functor<uint8_t>{'0', true},
//                                                   0);

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   // ASSERT_EQ(static_cast<uint32_t>(5), h_indices.size());

//   // EXPECT_EQ(static_cast<uint32_t>(1), h_indices[0]);
//   // EXPECT_EQ(static_cast<uint32_t>(7), h_indices[1]);
//   // EXPECT_EQ(static_cast<uint32_t>(14), h_indices[2]);
//   // EXPECT_EQ(static_cast<uint32_t>(17), h_indices[3]);
//   // EXPECT_EQ(static_cast<uint32_t>(21), h_indices[4]);

//   for (uint64_t i = 0; i < h_indices.size(); i++) {
//     EXPECT_EQ(static_cast<uint32_t>(-1), h_indices[i]);
//   }
// }

struct ascend_state {
  uint8_t prev;
  uint8_t next;
  bool did_ascend;
  bool is_identity;

  inline constexpr ascend_state() : prev(0), next(0), did_ascend(false), is_identity(true) {}
  inline constexpr ascend_state(uint8_t value)
    : prev(value), next(value), did_ascend(false), is_identity(false)
  {
  }
  inline constexpr ascend_state(uint8_t prev, uint8_t next, bool did_ascend)
    : prev(prev), next(next), did_ascend(did_ascend), is_identity(false)
  {
  }

  inline __device__ ascend_state operator+(ascend_state const& rhs) const
  {
    auto const& lhs = *this;

    if (lhs.is_identity) { return rhs; }
    if (rhs.is_identity) { return lhs; }

    auto result = ascend_state(lhs.prev, rhs.next, lhs.next < rhs.prev);

    // printf("[bid(%i) tid(%i)]: (%i, %i, %i, %i) = (%i, %i, %i, %i) + (%i, %i, %i, %i)\n",  //
    //        blockIdx.x,
    //        threadIdx.x,
    //        result.prev,
    //        result.next,
    //        result.did_ascend,
    //        result.is_identity,
    //        lhs.prev,
    //        lhs.next,
    //        lhs.did_ascend,
    //        lhs.is_identity,
    //        rhs.prev,
    //        rhs.next,
    //        rhs.did_ascend,
    //        rhs.is_identity);

    return result;
  }
};

struct ascend_reduce_functor {
  inline __device__ ascend_state operator()(ascend_state const& lhs, ascend_state const& rhs)
  {
    return lhs + rhs;
  }
};

struct ascend_detect_functor {
  inline __device__ bool operator()(ascend_state const& state)
  {
    printf("[bid(%i) tid(%i)]: (%i, %i, %i, %i)\n",  //
           blockIdx.x,
           threadIdx.x,
           state.prev,
           state.next,
           state.did_ascend,
           state.is_identity);

    return state.did_ascend;
  }
};

// TEST_F(InclusiveScanCopyIfTest, AscendStateWorksProperly)
// {
//   auto id = ascend_state();
//   auto a  = ascend_state(4);
//   auto b  = ascend_state(5);
//   auto c  = ascend_state(3);

//   auto ab = a + b;
//   EXPECT_TRUE(ab.did_ascend);
//   EXPECT_EQ(a.prev, ab.prev);
//   EXPECT_EQ(b.next, ab.next);

//   auto bc = b + c;
//   EXPECT_FALSE(bc.did_ascend);
//   EXPECT_EQ(b.prev, bc.prev);
//   EXPECT_EQ(c.next, bc.next);

//   auto aid = a + id;
//   EXPECT_FALSE(aid.did_ascend);
//   EXPECT_EQ(a.prev, aid.prev);
//   EXPECT_EQ(a.next, aid.next);
// }

// TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy3)
// {
//   auto input       = std::vector<int>{1, 6, 9, 5, 4, 8, 3, 2, 8, 9};
//   auto input_state = std::vector<ascend_state>(input.size());

//   std::transform(  //
//     input.begin(),
//     input.end(),
//     input_state.begin(),
//     [](int value) { return ascend_state(value); });

//   rmm::device_vector<ascend_state> d_input_state(input_state.begin(), input_state.end());

//   auto d_output = inclusive_scan_copy_if<ascend_state>(d_input_state,  //
//                                                        ascend_reduce_functor{},
//                                                        ascend_detect_functor{},
//                                                        0);

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   ASSERT_EQ(static_cast<uint32_t>(5), h_indices.size());

//   EXPECT_EQ(static_cast<uint32_t>(1), h_indices[0]);
//   EXPECT_EQ(static_cast<uint32_t>(2), h_indices[1]);
//   EXPECT_EQ(static_cast<uint32_t>(5), h_indices[2]);
//   EXPECT_EQ(static_cast<uint32_t>(8), h_indices[3]);
//   EXPECT_EQ(static_cast<uint32_t>(9), h_indices[4]);
// }

TEST_F(InclusiveScanCopyIfTest, CanInclusiveScanCopy4)
{
  auto input = std::vector<int>(8);
  std::iota(input.begin(), input.end(), 0);

  auto input_state = std::vector<ascend_state>(input.size());

  std::transform(  //
    input.begin(),
    input.end(),
    input_state.begin(),
    [](int value) { return ascend_state(value); });

  rmm::device_vector<ascend_state> d_input_state(input_state.begin(), input_state.end());

  auto print_op = [](thrust::host_vector<ascend_state> states,
                     thrust::host_vector<uint32_t> counts) {
    for (uint32_t i = 0; i < states.size(); i++) {
      auto& state = states[i];
      auto& count = counts[i];
      printf("%i: (%i, %i, %i, %i) (%i)\n",  //
             i,
             state.prev,
             state.next,
             state.did_ascend,
             state.is_identity,
             count);
    }
  };

  auto d_output = inclusive_scan_copy_if<ascend_state>(d_input_state,  //
                                                       ascend_reduce_functor{},
                                                       ascend_detect_functor{},
                                                       print_op,
                                                       0);

  thrust::host_vector<uint32_t> h_indices = d_output;

  ASSERT_EQ(static_cast<uint32_t>(input.size() - 1), h_indices.size());

  // for (uint64_t i = 0; i < input.size() - 1; i++) {
  //   EXPECT_EQ(static_cast<uint32_t>(i + 1), h_indices[i]);
  // }

  // for (uint64_t i = 0; i < h_indices.size(); i++) {
  //   EXPECT_EQ(static_cast<uint32_t>(-1), h_indices[i]);
  // }
}

CUDF_TEST_PROGRAM_MAIN()
