#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "csv_test_new.cuh"

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

class CsvReaderTest : public cudf::test::BaseFixture {
};

void expect_eq(csv_dim const& expected, csv_dim const& actual)
{
  EXPECT_EQ(expected.num_columns, actual.num_columns);
  EXPECT_EQ(expected.num_rows, actual.num_rows);
}

void expect_eq(csv_dimensional_sum const& expected, csv_dimensional_sum const& actual)
{
  EXPECT_EQ(expected.c, actual.c);
  EXPECT_EQ(expected.toggle, actual.toggle);
  expect_eq(expected.dimensions[0], actual.dimensions[0]);
  expect_eq(expected.dimensions[1], actual.dimensions[1]);
}

TEST_F(CsvReaderTest, CanCountCommas)
{
  using _     = csv_dimensional_sum;
  auto result = _{'x'} + _{','};

  expect_eq({',', false, {{1}, {0}}}, result);
}

TEST_F(CsvReaderTest, CanIgnoreEscapedCommas)
{
  using _     = csv_dimensional_sum;
  auto result = _{'\\'} + _{','};

  expect_eq({',', false, {{0}, {0}}}, result);
}

TEST_F(CsvReaderTest, CanIgnorePreviousCommas)
{
  using _     = csv_dimensional_sum;
  auto result = _{','} + _{'x'};

  expect_eq({'x', false, {{0}, {0}}}, result);
}

TEST_F(CsvReaderTest, CanToggleOnDoubleQuote)
{
  using _     = csv_dimensional_sum;
  auto result = _{','} + _{'"'};

  expect_eq({'"', true, {{0}, {0}}}, result);
}

csv_dimensional_sum csv_scan_state_reduce(std::string input)
{
  using _     = csv_dimensional_sum;
  auto result = _{static_cast<uint8_t>(input[0])};

  for (char c : input.substr(1)) {  //
    result = result + _{static_cast<uint8_t>(c)};
  }

  return result;
}

TEST_F(CsvReaderTest, CanCombineMultiple)
{
  auto result = csv_scan_state_reduce("a,\"b,c\",d");

  expect_eq({'d', false, {{2}, {1}}}, result);
}

TEST_F(CsvReaderTest, CanCombineMultiple2)
{
  auto result = csv_scan_state_reduce("Christopher, \"Hello, World\", Harris");

  expect_eq({'s', false, {{2}, {1}}}, result);
}

TEST_F(CsvReaderTest, CanCombineMultiple3)
{
  auto input = std::string("Christopher,\n\n \"Hello,\\n\n World\", Harris,,,,");

  // can't reliably use reduce here, because it "does not support" non-commutative operators.
  auto result = thrust::transform_reduce(input.c_str(),
                                         input.c_str() + input.size(),
                                         csv_dimensional_sum_factory{},
                                         csv_dimensional_sum::identity(),
                                         thrust::plus<csv_dimensional_sum>());

  expect_eq({',', false, {{6, 2}, {1, 1}}}, result);
}

// ===== TYPE INFERENCE ============================================================================

TEST_F(CsvReaderTest, CanDeduceValueTypeInteger)
{
  auto input = std::string("12349851");

  // can't reliably use reduce here, because it "does not support" non-commutative operators.
  auto result = thrust::transform_reduce(input.c_str(),
                                         input.c_str() + input.size(),
                                         csv_type_deduction_sum_factory{},
                                         csv_type_deduction_sum::identity(),
                                         thrust::plus<csv_type_deduction_sum>());

  EXPECT_EQ(csv_column_type::integer, result.type);
}

TEST_F(CsvReaderTest, CanDeduceValueTypeString)
{
  auto input = std::string("123x49851");

  // can't reliably use reduce here, because it "does not support" non-commutative operators.
  auto result = thrust::transform_reduce(input.c_str(),
                                         input.c_str() + input.size(),
                                         csv_type_deduction_sum_factory{},
                                         csv_type_deduction_sum::identity(),
                                         thrust::plus<csv_type_deduction_sum>());

  EXPECT_EQ(csv_column_type::string, result.type);
}

TEST_F(CsvReaderTest, X)
{
  rmm::device_vector<uint32_t> d_input = std::vector<uint32_t>{0, 0, 0, 0, 5, 0, 0, 0, 0, 0};

  auto result = reduce(d_input);

  EXPECT_EQ(static_cast<uint32_t>(5), result);
}

TEST_F(CsvReaderTest, CanGatherPositions)
{
  auto input = std::string("00______0__0____0____0__0__");
  rmm::device_vector<uint8_t> d_input(input.c_str(), input.c_str() + input.size());

  auto d_output = find(d_input, '0');

  thrust::host_vector<uint32_t> h_indices = d_output;

  EXPECT_EQ(static_cast<uint32_t>(7), h_indices.size());

  EXPECT_EQ(static_cast<uint32_t>(0), h_indices[0]);
  EXPECT_EQ(static_cast<uint32_t>(1), h_indices[1]);
  EXPECT_EQ(static_cast<uint32_t>(8), h_indices[2]);
  EXPECT_EQ(static_cast<uint32_t>(11), h_indices[3]);
  EXPECT_EQ(static_cast<uint32_t>(16), h_indices[4]);
  EXPECT_EQ(static_cast<uint32_t>(21), h_indices[5]);
  EXPECT_EQ(static_cast<uint32_t>(24), h_indices[6]);
}

CUDF_TEST_PROGRAM_MAIN()
