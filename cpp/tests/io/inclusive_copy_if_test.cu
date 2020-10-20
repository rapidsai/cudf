#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

// #include "inclusive_copy_if.cuh"

// #include <cudf/utilities/span.hpp>

// #include <rmm/thrust_rmm_allocator.h>

// #include <thrust/functional.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/iterator/transform_iterator.h>
// #include <thrust/reduce.h>
// #include <thrust/transform_reduce.h>

// #include <algorithm>
// #include <limits>
// #include <numeric>
// #include <string>
// #include <vector>

// class InclusiveCopyIfTest : public cudf::test::BaseFixture {
// };

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

// TEST_F(InclusiveCopyIfTest, CanInclusiveScanCopy)
// {
//   auto input = std::string("00__1___0__0_1__01___0__0_1");
//   rmm::device_vector<uint8_t> d_input(input.c_str(), input.c_str() + input.size());

//   auto d_output = inclusive_copy_if<uint8_t>(device_span<uint8_t>(d_input),  //
//                                              pick_rhs_functor<uint8_t>(),
//                                              find_functor<uint8_t>{'1', false},
//                                              0);

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   ASSERT_EQ(static_cast<uint32_t>(4), h_indices.size());

//   EXPECT_EQ(static_cast<uint32_t>(4), h_indices[0]);
//   EXPECT_EQ(static_cast<uint32_t>(13), h_indices[1]);
//   EXPECT_EQ(static_cast<uint32_t>(17), h_indices[2]);
//   EXPECT_EQ(static_cast<uint32_t>(26), h_indices[3]);
// }

// TEST_F(InclusiveCopyIfTest, CanInclusiveScanCopy2)
// {
//   auto input = std::string("0100000100000010010001");
//   rmm::device_vector<uint8_t> d_input(input.c_str(), input.c_str() + input.size());

//   auto d_output = inclusive_copy_if<uint8_t>(device_span<uint8_t>(d_input),  //
//                                              pick_rhs_functor<uint8_t>(),
//                                              find_functor<uint8_t>{'0', true},
//                                              0);

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   ASSERT_EQ(static_cast<uint32_t>(5), h_indices.size());

//   EXPECT_EQ(static_cast<uint32_t>(1), h_indices[0]);
//   EXPECT_EQ(static_cast<uint32_t>(7), h_indices[1]);
//   EXPECT_EQ(static_cast<uint32_t>(14), h_indices[2]);
//   EXPECT_EQ(static_cast<uint32_t>(17), h_indices[3]);
//   EXPECT_EQ(static_cast<uint32_t>(21), h_indices[4]);
// }

// struct ascend_state {
//   uint8_t prev;
//   uint8_t next;
//   bool did_ascend;
//   bool is_identity;

//   inline constexpr ascend_state() : prev(0), next(0), did_ascend(false), is_identity(true) {}
//   inline constexpr ascend_state(uint8_t value)
//     : prev(value), next(value), did_ascend(false), is_identity(false)
//   {
//   }
//   inline constexpr ascend_state(uint8_t prev, uint8_t next, bool did_ascend)
//     : prev(prev), next(next), did_ascend(did_ascend), is_identity(false)
//   {
//   }

//   inline constexpr ascend_state operator+(ascend_state const& rhs) const
//   {
//     auto const& lhs = *this;

//     if (lhs.is_identity) { return rhs; }
//     if (rhs.is_identity) { return lhs; }

//     auto result = ascend_state(lhs.prev, rhs.next, lhs.next < rhs.prev);

//     return result;
//   }
// };

// struct ascend_reduce_functor {
//   inline constexpr ascend_state operator()(ascend_state const& lhs, ascend_state const& rhs)
//   {
//     return lhs + rhs;
//   }
// };

// struct ascend_detect_functor {
//   inline constexpr bool operator()(ascend_state const& state) { return state.did_ascend; }
// };

// TEST_F(InclusiveCopyIfTest, CanInclusiveScanCopy3)
// {
//   auto input       = std::vector<int>{1, 6, 9, 5, 4, 8, 3, 2, 8, 9};
//   auto input_state = std::vector<ascend_state>(input.size());

//   std::transform(  //
//     input.begin(),
//     input.end(),
//     input_state.begin(),
//     [](int value) { return ascend_state(value); });

//   rmm::device_vector<ascend_state> d_input_state(input_state.begin(), input_state.end());

//   auto d_output = inclusive_copy_if<ascend_state>(d_input_state,  //
//                                                   ascend_reduce_functor{},
//                                                   ascend_detect_functor{},
//                                                   0);

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   ASSERT_EQ(static_cast<uint32_t>(5), h_indices.size());

//   EXPECT_EQ(static_cast<uint32_t>(1), h_indices[0]);
//   EXPECT_EQ(static_cast<uint32_t>(2), h_indices[1]);
//   EXPECT_EQ(static_cast<uint32_t>(5), h_indices[2]);
//   EXPECT_EQ(static_cast<uint32_t>(8), h_indices[3]);
//   EXPECT_EQ(static_cast<uint32_t>(9), h_indices[4]);
// }

// TEST_F(InclusiveCopyIfTest, CanInclusiveScanCopy4)
// {
//   auto input = std::vector<int>(8);
//   std::iota(input.begin(), input.end(), 0);

//   auto input_state = std::vector<ascend_state>(input.size());

//   std::transform(  //
//     input.begin(),
//     input.end(),
//     input_state.begin(),
//     [](int value) { return ascend_state(value); });

//   rmm::device_vector<ascend_state> d_input_state(input_state.begin(), input_state.end());

//   auto d_output = inclusive_copy_if<ascend_state>(d_input_state,  //
//                                                   ascend_reduce_functor{},
//                                                   ascend_detect_functor{},
//                                                   0);

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   ASSERT_EQ(static_cast<uint32_t>(input.size() - 1), h_indices.size());

//   for (uint64_t i = 0; i < input.size() - 1; i++) {
//     EXPECT_EQ(static_cast<uint32_t>(i + 1), h_indices[i]);
//   }
// }

// template <typename T, size_t N>
// constexpr size_t array_size(T (&)[N])
// {
//   return N;
// }

// struct matcher_state {
//   uint8_t c;
//   uint32_t prev;
//   uint32_t next;
//   bool is_identity = true;

//   static constexpr uint8_t pattern[] = "hand";

//   static inline constexpr int find(uint8_t c)
//   {
//     for (uint64_t i = 0; i < array_size(pattern); i++) {
//       if (pattern[i] == c) { return i; }
//     }

//     return -1;
//   }

//   inline constexpr matcher_state()  //
//     : c(0),                         //
//       prev(0),
//       next(0),
//       is_identity(true)
//   {
//   }
//   inline constexpr matcher_state(uint8_t value)  //
//     : c(value),                                  //                    //
//       prev(0),
//       next(0),
//       is_identity(false)
//   {  //
//     auto m = find(c);
//     prev   = max(0, m);
//     next   = m + 1;
//   }

//   inline constexpr matcher_state(uint8_t value, uint32_t prev, uint32_t next)  //
//     : c(value),                                                                //
//       prev(prev),
//       next(next),
//       is_identity(false)
//   {
//   }

//   struct scan {
//     inline constexpr matcher_state operator()(  //
//       matcher_state const& lhs,
//       matcher_state const& rhs) const
//     {
//       if (lhs.is_identity) { return rhs; }
//       if (rhs.is_identity) { return lhs; }

//       if (lhs.next == rhs.prev) {
//         return matcher_state(lhs.c, lhs.prev, rhs.next);
//       } else {
//         return matcher_state(lhs.c, 0, 0);
//       }

//       return {};
//     }
//   };

//   struct predicate {
//     inline constexpr bool operator()(matcher_state const& state)
//     {
//       return state.prev == 0 and state.next == array_size(pattern);
//     }
//   };
// };

// TEST_F(InclusiveCopyIfTest, CanScanBytes)
// {
//   auto input = std::string("can you give me a hand with this?");

//   auto h_input = std::vector<matcher_state>(input.size());

//   std::transform(  //
//     input.begin(),
//     input.end(),
//     h_input.begin(),
//     [](uint8_t c) { return matcher_state(c); });

//   rmm::device_vector<matcher_state> d_input = h_input;

//   auto d_output = inclusive_copy_if<matcher_state>(  //
//     device_span<matcher_state>(d_input),
//     matcher_state::scan(),
//     matcher_state::predicate());

//   thrust::host_vector<uint32_t> h_indices = d_output;

//   ASSERT_EQ(static_cast<uint32_t>(1), h_indices.size());

//   EXPECT_EQ(static_cast<uint32_t>(18), h_indices[0]);
// }

CUDF_TEST_PROGRAM_MAIN()
