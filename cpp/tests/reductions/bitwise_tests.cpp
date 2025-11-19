/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>

template <typename T>
using column = cudf::test::fixed_width_column_wrapper<T>;

class BitwiseAggregationTest : public cudf::test::BaseFixture {};

template <typename T>
class BitwiseAggTypedTest : public BitwiseAggregationTest {};

// Test types for bitwise operations - only integer types
TYPED_TEST_SUITE(BitwiseAggTypedTest, cudf::test::IntegralTypes);

// Basic bitwise AND test
TYPED_TEST(BitwiseAggTypedTest, BitwiseAND)
{
  using T = TypeParam;

  // Test data
  auto const input_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, static_cast<T>(0xFF), 0x2F, 0x3F};
    }
  }();
  auto const input_col = column<T>(input_data.begin(), input_data.end());

  // Expected result - bitwise AND of all values
  auto const expected = static_cast<T>(input_data[0] & input_data[1] & input_data[2] &
                                       input_data[3] & input_data[4] & input_data[5]);

  // Create and execute aggregation
  auto const agg = cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::AND);
  auto const result = cudf::reduce(input_col, *agg, cudf::data_type{cudf::type_to_id<T>()});

  EXPECT_EQ(expected, static_cast<cudf::scalar_type_t<T>*>(result.get())->value());
  EXPECT_TRUE(result->is_valid());
}

// Basic bitwise OR test
TYPED_TEST(BitwiseAggTypedTest, BitwiseOR)
{
  using T = TypeParam;

  // Test data
  auto const input_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, static_cast<T>(0xFF), 0x2F, 0x3F};
    }
  }();
  auto const input_col = column<T>(input_data.begin(), input_data.end());

  // Expected result - bitwise OR of all values
  auto const expected = static_cast<T>(input_data[0] | input_data[1] | input_data[2] |
                                       input_data[3] | input_data[4] | input_data[5]);

  // Create and execute aggregation
  auto const agg = cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::OR);
  auto const result = cudf::reduce(input_col, *agg, cudf::data_type{cudf::type_to_id<T>()});

  EXPECT_EQ(expected, static_cast<cudf::scalar_type_t<T>*>(result.get())->value());
  EXPECT_TRUE(result->is_valid());
}

// Basic bitwise XOR test
TYPED_TEST(BitwiseAggTypedTest, BitwiseXOR)
{
  using T = TypeParam;

  // Test data
  auto const input_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, static_cast<T>(0xFF), 0x2F, 0x3F};
    }
  }();
  auto const input_col = column<T>(input_data.begin(), input_data.end());

  // Expected result - bitwise XOR of all values
  auto const expected = static_cast<T>(input_data[0] ^ input_data[1] ^ input_data[2] ^
                                       input_data[3] ^ input_data[4] ^ input_data[5]);

  // Create and execute aggregation
  auto const agg = cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::XOR);
  auto const result = cudf::reduce(input_col, *agg, cudf::data_type{cudf::type_to_id<T>()});

  EXPECT_EQ(expected, static_cast<cudf::scalar_type_t<T>*>(result.get())->value());
  EXPECT_TRUE(result->is_valid());
}

// Test with null values
TYPED_TEST(BitwiseAggTypedTest, WithNulls)
{
  using T = TypeParam;

  // Test data with nulls
  auto const input_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, true, false, true, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, static_cast<T>(0xFF), 0x2F, 0x3F};
    }
  }();
  // If T is bool, null is at the position of a false value.
  auto const validity  = std::vector<bool>{true, true, false, true, true, true};
  auto const input_col = column<T>(input_data.begin(), input_data.end(), validity.begin());

  // Expected results - null values should be excluded
  auto const expected_and =
    static_cast<T>(input_data[0] & input_data[1] & input_data[3] & input_data[4] & input_data[5]);
  auto const expected_or =
    static_cast<T>(input_data[0] | input_data[1] | input_data[3] | input_data[4] | input_data[5]);
  auto const expected_xor =
    static_cast<T>(input_data[0] ^ input_data[1] ^ input_data[3] ^ input_data[4] ^ input_data[5]);

  // Create and execute aggregations
  auto const agg_and =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::AND);
  auto const result_and = cudf::reduce(input_col, *agg_and, cudf::data_type{cudf::type_to_id<T>()});

  auto const agg_or =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::OR);
  auto const result_or = cudf::reduce(input_col, *agg_or, cudf::data_type{cudf::type_to_id<T>()});

  auto const agg_xor =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::XOR);
  auto const result_xor = cudf::reduce(input_col, *agg_xor, cudf::data_type{cudf::type_to_id<T>()});

  // Validate results
  EXPECT_EQ(expected_and, static_cast<cudf::scalar_type_t<T>*>(result_and.get())->value());
  EXPECT_TRUE(result_and->is_valid());

  EXPECT_EQ(expected_or, static_cast<cudf::scalar_type_t<T>*>(result_or.get())->value());
  EXPECT_TRUE(result_or->is_valid());

  EXPECT_EQ(expected_xor, static_cast<cudf::scalar_type_t<T>*>(result_xor.get())->value());
  EXPECT_TRUE(result_xor->is_valid());
}

// Test with empty column
TYPED_TEST(BitwiseAggTypedTest, EmptyColumn)
{
  using T = TypeParam;

  // Empty column
  auto const empty_col = column<T>{};

  // Create aggregations
  auto const agg_and =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::AND);
  auto const agg_or =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::OR);
  auto const agg_xor =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::XOR);

  // Get results
  auto const result_and = cudf::reduce(empty_col, *agg_and, cudf::data_type{cudf::type_to_id<T>()});
  auto const result_or  = cudf::reduce(empty_col, *agg_or, cudf::data_type{cudf::type_to_id<T>()});
  auto const result_xor = cudf::reduce(empty_col, *agg_xor, cudf::data_type{cudf::type_to_id<T>()});

  // Check that results are not valid for empty inputs
  EXPECT_FALSE(result_and->is_valid());
  EXPECT_FALSE(result_or->is_valid());
  EXPECT_FALSE(result_xor->is_valid());
}

// Test with invalid input type
TEST_F(BitwiseAggregationTest, InvalidInputType)
{
  // Create float column (not supported by bitwise operations)
  auto const float_col = cudf::test::fixed_width_column_wrapper<float>{1.1f, 2.2f, 3.3f};

  // Create string column (not supported by bitwise operations)
  auto const string_col = cudf::test::strings_column_wrapper{"a", "b", "c"};

  // Create aggregations
  auto const agg_and =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::AND);
  auto const agg_or =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::OR);
  auto const agg_xor =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::XOR);

  // Float input should throw exception
  EXPECT_THROW(cudf::reduce(float_col, *agg_and, cudf::data_type{cudf::type_to_id<float>()}),
               std::invalid_argument);
  EXPECT_THROW(cudf::reduce(float_col, *agg_or, cudf::data_type{cudf::type_to_id<float>()}),
               std::invalid_argument);
  EXPECT_THROW(cudf::reduce(float_col, *agg_xor, cudf::data_type{cudf::type_to_id<float>()}),
               std::invalid_argument);

  // String input should throw exception
  EXPECT_THROW(cudf::reduce(string_col, *agg_and, cudf::data_type{cudf::type_to_id<int32_t>()}),
               std::invalid_argument);
  EXPECT_THROW(cudf::reduce(string_col, *agg_or, cudf::data_type{cudf::type_to_id<int32_t>()}),
               std::invalid_argument);
  EXPECT_THROW(cudf::reduce(string_col, *agg_xor, cudf::data_type{cudf::type_to_id<int32_t>()}),
               std::invalid_argument);
}

// Test with all null values
TYPED_TEST(BitwiseAggTypedTest, AllNulls)
{
  using T = TypeParam;

  auto const input_data = std::vector<T>{0, 0, 0};
  auto const validity   = std::vector<bool>{false, false, false};
  auto const input_col  = column<T>(input_data.begin(), input_data.end(), validity.begin());

  // Create aggregations
  auto const agg_and =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::AND);
  auto const agg_or =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::OR);
  auto const agg_xor =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::XOR);

  // Get results
  auto const result_and = cudf::reduce(input_col, *agg_and, cudf::data_type{cudf::type_to_id<T>()});
  auto const result_or  = cudf::reduce(input_col, *agg_or, cudf::data_type{cudf::type_to_id<T>()});
  auto const result_xor = cudf::reduce(input_col, *agg_xor, cudf::data_type{cudf::type_to_id<T>()});

  // Results should be invalid
  EXPECT_FALSE(result_and->is_valid());
  EXPECT_FALSE(result_or->is_valid());
  EXPECT_FALSE(result_xor->is_valid());
}

// Test with a single row
TYPED_TEST(BitwiseAggTypedTest, SingleRow)
{
  using T = TypeParam;

  // Column with a single value
  auto const single_value = static_cast<T>(0x42);
  auto const single_col   = column<T>{single_value};

  // Create aggregations
  auto const agg_and =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::AND);
  auto const agg_or =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::OR);
  auto const agg_xor =
    cudf::make_bitwise_aggregation<cudf::reduce_aggregation>(cudf::bitwise_op::XOR);

  // Get results
  auto const result_and =
    cudf::reduce(single_col, *agg_and, cudf::data_type{cudf::type_to_id<T>()});
  auto const result_or = cudf::reduce(single_col, *agg_or, cudf::data_type{cudf::type_to_id<T>()});
  auto const result_xor =
    cudf::reduce(single_col, *agg_xor, cudf::data_type{cudf::type_to_id<T>()});

  // For a single value, all operations should return that value
  EXPECT_EQ(single_value, static_cast<cudf::scalar_type_t<T>*>(result_and.get())->value());
  EXPECT_TRUE(result_and->is_valid());

  EXPECT_EQ(single_value, static_cast<cudf::scalar_type_t<T>*>(result_or.get())->value());
  EXPECT_TRUE(result_or->is_valid());

  EXPECT_EQ(single_value, static_cast<cudf::scalar_type_t<T>*>(result_xor.get())->value());
  EXPECT_TRUE(result_xor->is_valid());
}
