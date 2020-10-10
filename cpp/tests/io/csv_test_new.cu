#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "csv_test_new.hpp"

#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <numeric>
#include <vector>

using cudf::detail::device_span;

class JsonReaderTest : public cudf::test::BaseFixture {
};

struct csv_scan_state {
  uint8_t c;
  bool toggle;
  uint32_t num_commas_curr;
  uint32_t num_commas_swap;

  constexpr csv_scan_state operator+(csv_scan_state rhs)
  {
    auto lhs = *this;

    auto const next_is_comma = lhs.c != '\\' && rhs.c == ',';
    auto const next_is_quote = lhs.c != '\\' && rhs.c == '"';

    if (lhs.toggle != rhs.toggle) {
      // rectify
      auto temp           = lhs.num_commas_curr;
      lhs.num_commas_curr = lhs.num_commas_swap;
      lhs.num_commas_swap = temp;
    }

    if (next_is_comma) {
      // increment due to comma found
      rhs.num_commas_curr += 1;
    }

    return {rhs.c,
            next_is_quote ? not rhs.toggle : rhs.toggle,
            lhs.num_commas_curr + rhs.num_commas_curr,
            lhs.num_commas_swap + rhs.num_commas_swap};
  }

  constexpr uint32_t operator~() { return toggle ? num_commas_swap : num_commas_curr; }
};

__global__ void doot(device_span<uint8_t const> input, csv_scan_state* output)
{
  // // Specialize BlockReduce for a 1D block of 128 threads on type int
  // // Allocate shared memory for BlockReduce
  // typedef cub::BlockReduce<csv_scan_state, 128> BlockReduce;
  // __shared__ typename BlockReduce::TempStorage temp_storage;
  // // // Obtain a segment of consecutive items that are blocked across threads
  // csv_scan_state thread_data[1];
  // // // Compute the block-wide sum for thread0

  // thread_data[threadIdx.x] = {input[threadIdx.x]};

  // int aggregate =
  //   BlockReduce(temp_storage).Reduce(thread_data, csv_scan_state_reducer{}, int num_valid);
}

void expect_eq(csv_scan_state expected, csv_scan_state actual)
{
  EXPECT_EQ(expected.c, actual.c);
  EXPECT_EQ(expected.toggle, actual.toggle);
  EXPECT_EQ(expected.num_commas_curr, actual.num_commas_curr);
  EXPECT_EQ(expected.num_commas_swap, actual.num_commas_swap);
}

TEST_F(JsonReaderTest, CanCountCommas)
{
  using _     = csv_scan_state;
  auto result = _{'x'} + _{','};

  expect_eq({',', false, 1, 0}, result);
}

TEST_F(JsonReaderTest, CanIgnoreEscapedCommas)
{
  using _     = csv_scan_state;
  auto result = _{'\\'} + _{','};

  expect_eq({',', false, 0, 0}, result);
}

TEST_F(JsonReaderTest, CanIgnorePreviousCommas)
{
  using _     = csv_scan_state;
  auto result = _{','} + _{'x'};

  expect_eq({'x', false, 0, 0}, result);
}

TEST_F(JsonReaderTest, CanToggleOnDoubleQuote)
{
  using _     = csv_scan_state;
  auto result = _{','} + _{'"'};

  expect_eq({'"', true, 0, 0}, result);
}

csv_scan_state csv_scan_state_reduce(std::string input)
{
  using _     = csv_scan_state;
  auto result = _{static_cast<uint8_t>(input[0])};

  for (char c : input.substr(1)) {  //
    result = result + _{static_cast<uint8_t>(c)};
  }

  return result;
}

TEST_F(JsonReaderTest, CanCombineMultiple)
{
  auto result = csv_scan_state_reduce("a,\"b,c\",d");

  expect_eq({'d', false, 2, 1}, result);
}

TEST_F(JsonReaderTest, CanCombineMultiple2)
{
  auto result = csv_scan_state_reduce("Christopher, \"Hello, World\", Harris");

  expect_eq({'s', false, 2, 1}, result);
}

TEST_F(JsonReaderTest, CanCombineMultiple3)
{
  EXPECT_EQ(static_cast<uint32_t>(6),
            ~csv_scan_state_reduce("Christopher, \"Hello, World\", Harris,,,,"));
}

CUDF_TEST_PROGRAM_MAIN()
