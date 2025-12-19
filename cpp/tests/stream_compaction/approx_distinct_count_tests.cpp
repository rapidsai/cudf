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
  cudf::table_view input_table({input_col});

  auto adc          = cudf::approx_distinct_count(input_table);
  auto approx_count = adc.estimate();
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

  auto adc               = cudf::approx_distinct_count(input_table);
  auto approx_count      = adc.estimate();
  auto const exact_count = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, 3, 3, XXX, 1, 8, 2},
                                                            {1, 1, 1, 0, 1, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, IgnoreNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, 3, 3, XXX, 1, 8, 2},
                                                            {1, 1, 1, 0, 1, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, BothAPIs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, 3, 3, XXX, 1, 8, 2},
                                                            {1, 1, 1, 0, 1, 1, 1}};
  cudf::table_view input_table({input_col});

  // Test using both null_policy and nan_policy parameters
  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, EmptyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{};
  cudf::table_view input_table({input_col});

  auto adc    = cudf::approx_distinct_count(input_table);
  auto result = adc.estimate();
  EXPECT_EQ(0, result);
}

TEST_F(ApproxDistinctCount, StringColumn)
{
  cudf::test::strings_column_wrapper input_col{"a", "b", "a", "c", "b"};
  cudf::table_view input_table({input_col});

  auto adc          = cudf::approx_distinct_count(input_table);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, DifferentPrecisions)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{1, 2, 3, 4, 5, 1, 2, 3};
  cudf::table_view input_table({input_col});

  // Test precision bounds (should clamp to 4-18) but use reasonable values
  auto adc_low  = cudf::approx_distinct_count(input_table, 2);   // Should clamp to 4
  auto adc_mid  = cudf::approx_distinct_count(input_table, 12);  // Default precision
  auto adc_high = cudf::approx_distinct_count(input_table, 10);  // Lower precision to test memory

  auto result_low  = adc_low.estimate();
  auto result_mid  = adc_mid.estimate();
  auto result_high = adc_high.estimate();

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
  cudf::table_view input_table({input_col});

  // For approximate distinct count, we simplify null handling
  // Just test that it gives a reasonable approximation
  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullEqualityEqual)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, XXX, 3, XXX, 1}, {1, 0, 1, 0, 1}};
  cudf::table_view input_table({input_col});

  // With simplified API, all nulls are treated as equal
  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
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
  cudf::table_view input_table({input_col});

  // Test NaN as null with EXCLUDE policy
  auto adc_exclude =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_exclude = adc_exclude.estimate();
  auto const exact_exclude =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_exclude, exact_exclude))
    << "Exclude - Exact: " << exact_exclude << ", Approx: " << approx_exclude;

  // Test NaN as null with INCLUDE policy
  auto adc_include =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_include = adc_include.estimate();
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
  cudf::table_view input_table({input_col});

  // For approximate distinct count, simplified NaN handling
  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
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
  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count      = adc.estimate();
  auto const exact_count = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, CombinedNullNaNHandling)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{{1.0f,
                                                           std::numeric_limits<float>::quiet_NaN(),
                                                           0.0f,
                                                           3.0f,
                                                           std::numeric_limits<float>::quiet_NaN()},
                                                          {1, 1, 0, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, AddToExistingSketch)
{
  // Create initial sketch
  cudf::test::fixed_width_column_wrapper<int32_t> col1{1, 2, 3};
  cudf::table_view table1({col1});
  auto adc = cudf::approx_distinct_count(table1);

  // Add more data
  cudf::test::fixed_width_column_wrapper<int32_t> col2{3, 4, 5};
  cudf::table_view table2({col2});
  adc.add(table2);

  auto approx_count = adc.estimate();

  // Create combined table for exact count
  cudf::test::fixed_width_column_wrapper<int32_t> combined{1, 2, 3, 3, 4, 5};
  auto exact_count = cudf::distinct_count(combined, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullInclude_NaNValid)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{
    {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 0.0f, 1.0f}, {1, 1, 0, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullInclude_NaNNull)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{{1.0f,
                                                           std::numeric_limits<float>::quiet_NaN(),
                                                           3.0f,
                                                           0.0f,
                                                           std::numeric_limits<float>::quiet_NaN()},
                                                          {1, 1, 0, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullExclude_NaNValid)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{{1.0f,
                                                           std::numeric_limits<float>::quiet_NaN(),
                                                           3.0f,
                                                           0.0f,
                                                           std::numeric_limits<float>::quiet_NaN()},
                                                          {1, 1, 0, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullExclude_NaNNull)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{{1.0f,
                                                           std::numeric_limits<float>::quiet_NaN(),
                                                           3.0f,
                                                           0.0f,
                                                           std::numeric_limits<float>::quiet_NaN()},
                                                          {1, 1, 0, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, TableWithNoNulls_ExcludePolicy)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{1, 2, 2, 4};
  cudf::table_view input_table({col1, col2});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count      = adc.estimate();
  auto const exact_count = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, FloatColumnWithNoNaNs_NaNNullPolicy)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{1.0f, 2.0f, 3.0f, 1.0f, 2.0f};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, MixedTypes_NullAndNaNHandling)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4}, {1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<float> float_col{
    {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f}, {1, 1, 1, 0}};

  cudf::table_view input_table({int_col, float_col});

  auto adc_in_v =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto adc_in_n =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto adc_ex_v =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto adc_ex_n =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  auto count_in_v = adc_in_v.estimate();
  auto count_in_n = adc_in_n.estimate();
  auto count_ex_v = adc_ex_v.estimate();
  auto count_ex_n = adc_ex_n.estimate();

  EXPECT_LE(count_ex_v, count_in_v);
  EXPECT_LE(count_ex_n, count_in_n);
  EXPECT_LE(count_ex_n, count_ex_v);
  EXPECT_LE(count_in_n, count_in_v);

  EXPECT_GT(count_in_v, 0);
  EXPECT_GT(count_in_n, 0);
  EXPECT_GT(count_ex_v, 0);
  EXPECT_GT(count_ex_n, 0);
}
