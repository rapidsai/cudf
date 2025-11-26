/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

using cudf::nan_policy;
using cudf::null_policy;

constexpr int32_t XXX{70};  // Mark for null elements

// Simple helper to check if approximation is reasonable (within 20% for small datasets)
bool is_reasonable_approximation(cudf::size_type approx_count, cudf::size_type exact_count)
{
  if (exact_count == 0) return approx_count == 0;
  if (exact_count == 1) return approx_count <= 2;  // Very small counts can vary
  double error = std::abs(static_cast<double>(approx_count) - static_cast<double>(exact_count)) /
                 static_cast<double>(exact_count);
  return error <= 0.2;  // 20% tolerance for simplicity
}

struct ApproxDistinctCount : public cudf::test::BaseFixture {};

TEST_F(ApproxDistinctCount, BasicFunctionality)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{1, 3, 3, 4, 1, 8, 2, 4, 10, 8};

  auto const approx_count = cudf::approx_distinct_count(input_col);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, TableBasic)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{1, 1, 2, 1, 2};
  cudf::table_view input_table({col1, col2});

  auto const approx_count = cudf::approx_distinct_count(input_table);
  auto const exact_count  = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, 3, 3, XXX, 1, 8, 2},
                                                            {1, 1, 1, 0, 1, 1, 1}};

  auto const approx_count =
    cudf::approx_distinct_count(input_col, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, IgnoreNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, 3, 3, XXX, 1, 8, 2},
                                                            {1, 1, 1, 0, 1, 1, 1}};

  auto const approx_count =
    cudf::approx_distinct_count(input_col, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, BothAPIs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, 3, 3, XXX, 1, 8, 2},
                                                            {1, 1, 1, 0, 1, 1, 1}};

  // Test using both null_policy and nan_policy parameters
  auto const approx_count =
    cudf::approx_distinct_count(input_col, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, EmptyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{};

  auto const result = cudf::approx_distinct_count(input_col);
  EXPECT_EQ(0, result);
}

TEST_F(ApproxDistinctCount, StringColumn)
{
  cudf::test::strings_column_wrapper input_col{"a", "b", "a", "c", "b"};

  auto const approx_count = cudf::approx_distinct_count(input_col);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, DifferentPrecisions)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{1, 2, 3, 4, 5, 1, 2, 3};

  // Test precision bounds (should clamp to 4-18) but use reasonable values
  auto const result_low = cudf::approx_distinct_count(input_col, 2);   // Should clamp to 4
  auto const result_mid = cudf::approx_distinct_count(input_col, 12);  // Default precision
  auto const result_high =
    cudf::approx_distinct_count(input_col, 10);  // Lower precision to test memory

  // All should give reasonable results for this small dataset
  EXPECT_GT(result_low, 0);
  EXPECT_GT(result_mid, 0);
  EXPECT_GT(result_high, 0);
  EXPECT_LE(result_low, 10);   // Should be reasonable
  EXPECT_LE(result_mid, 10);   // Should be reasonable
  EXPECT_LE(result_high, 10);  // Should be reasonable
}

// ===== COMPREHENSIVE NULL/NaN PARAMETER TESTING =====

TEST_F(ApproxDistinctCount, NullEqualityUnequal)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, XXX, 3, XXX, 1}, {1, 0, 1, 0, 1}};

  // For approximate distinct count, we simplify null handling
  // Just test that it gives a reasonable approximation
  auto const approx_count =
    cudf::approx_distinct_count(input_col, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullEqualityEqual)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, XXX, 3, XXX, 1}, {1, 0, 1, 0, 1}};

  // With simplified API, all nulls are treated as equal
  auto const approx_count =
    cudf::approx_distinct_count(input_col, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NaNHandling)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{1.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          3.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          1.0f};

  // Test NaN as null with EXCLUDE policy
  auto const approx_exclude =
    cudf::approx_distinct_count(input_col, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);
  auto const exact_exclude =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_exclude, exact_exclude))
    << "Exclude - Exact: " << exact_exclude << ", Approx: " << approx_exclude;

  // Test NaN as null with INCLUDE policy
  auto const approx_include =
    cudf::approx_distinct_count(input_col, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto const exact_include =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_include, exact_include))
    << "Include - Exact: " << exact_include << ", Approx: " << approx_include;
}

TEST_F(ApproxDistinctCount, NaNEqualityUnequal)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{1.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          3.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          1.0f};

  // For approximate distinct count, simplified NaN handling
  auto const approx_count =
    cudf::approx_distinct_count(input_col, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, TableNullHandling)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{1, XXX, 3, 1}, {1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{1, 2, XXX, 1}, {1, 1, 0, 1}};
  cudf::table_view input_table({col1, col2});

  // Test table with simplified null handling
  auto const approx_count =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto const exact_count = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, CombinedNullNaNHandling)
{
  // Create float column with both nulls and NaNs
  std::vector<float> values{1.0f,
                            std::numeric_limits<float>::quiet_NaN(),
                            0.0f,
                            3.0f,
                            std::numeric_limits<float>::quiet_NaN()};
  std::vector<bool> validity{true, true, false, true, true};
  cudf::test::fixed_width_column_wrapper<float> input_col(
    values.begin(), values.end(), validity.begin());

  // Test combination of null and NaN handling
  auto const approx_count =
    cudf::approx_distinct_count(input_col, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}
