/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

auto bitwise_aggregate(cudf::column_view const& keys,
                       cudf::column_view const& values,
                       cudf::bitwise_op bit_op)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = values;
  requests[0].aggregations.emplace_back(
    cudf::make_bitwise_aggregation<cudf::groupby_aggregation>(bit_op));
  auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));
  auto result = gb_obj.aggregate(requests);

  auto const sort_order = cudf::sorted_order(result.first->view(), {}, {});
  auto sorted_keys      = cudf::gather(result.first->view(), *sort_order);
  auto sorted_vals =
    cudf::gather(cudf::table_view({result.second.front().results.front()->view()}), *sort_order);

  return std::pair{std::move(sorted_keys->release().front()),
                   std::move(sorted_vals->release().front())};
}

template <typename T>
using column = cudf::test::fixed_width_column_wrapper<T>;

using keys_column = cudf::test::fixed_width_column_wrapper<int32_t>;

class GroupByBitwiseTest : public cudf::test::BaseFixture {};

template <typename T>
class GroupByBitwiseTypedTest : public GroupByBitwiseTest {};

// Test types for bitwise operations - only integer types
TYPED_TEST_SUITE(GroupByBitwiseTypedTest, cudf::test::IntegralTypes);

// Basic bitwise AND test
TYPED_TEST(GroupByBitwiseTypedTest, BitwiseAND)
{
  using T = TypeParam;

  // Keys and values
  auto const keys      = keys_column{1, 1, 2, 2, 2, 3, 3, 4};
  auto const vals_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, false, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, 0x55, 0x3F, 0x2F, 0x0F, 0x42};
    }
  }();
  auto const vals = column<T>(vals_data.begin(), vals_data.end());

  // Expected output
  auto const expected_keys = keys_column{1, 2, 3, 4};
  auto const expected_vals = column<T>{
    static_cast<T>(vals_data[0] & vals_data[1]),                 // key = 1
    static_cast<T>(vals_data[2] & vals_data[3] & vals_data[4]),  // key = 2
    static_cast<T>(vals_data[5] & vals_data[6]),                 // key = 3
    static_cast<T>(vals_data[7])                                 // key = 4
  };

  auto constexpr bit_op           = cudf::bitwise_op::AND;
  auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
}

// Basic bitwise OR test
TYPED_TEST(GroupByBitwiseTypedTest, BitwiseOR)
{
  using T = TypeParam;

  // Keys and values
  auto const keys      = keys_column{1, 1, 2, 2, 2, 3, 3, 4};
  auto const vals_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, false, true, true};
    } else {
      return std::vector<T>{0x10, 0x01, 0x20, 0x04, 0x08, 0x40, 0x4F, 0x42};
    }
  }();
  auto const vals = column<T>(vals_data.begin(), vals_data.end());

  // Expected output
  auto const expected_keys = keys_column{1, 2, 3, 4};
  auto const expected_vals = column<T>{
    static_cast<T>(vals_data[0] | vals_data[1]),                 // key = 1
    static_cast<T>(vals_data[2] | vals_data[3] | vals_data[4]),  // key = 2
    static_cast<T>(vals_data[5] | vals_data[6]),                 // key = 3
    static_cast<T>(vals_data[7])                                 // key = 4
  };

  auto constexpr bit_op           = cudf::bitwise_op::OR;
  auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
}

// Basic bitwise XOR test
TYPED_TEST(GroupByBitwiseTypedTest, BitwiseXOR)
{
  using T = TypeParam;

  // Keys and values
  auto const keys      = keys_column{1, 1, 2, 2, 2, 3, 3, 4};
  auto const vals_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, false, true, true};
    } else {
      return std::vector<T>{0x10, 0x01, 0x20, 0x04, 0x08, 0x40, 0x4F, 0x42};
    }
  }();
  auto const vals = column<T>(vals_data.begin(), vals_data.end());

  // Expected output
  auto const expected_keys = keys_column{1, 2, 3, 4};
  auto const expected_vals = column<T>{
    static_cast<T>(vals_data[0] ^ vals_data[1]),                 // key = 1
    static_cast<T>(vals_data[2] ^ vals_data[3] ^ vals_data[4]),  // key = 2
    static_cast<T>(vals_data[5] ^ vals_data[6]),                 // key = 3
    static_cast<T>(vals_data[7])                                 // key = 4
  };

  auto constexpr bit_op           = cudf::bitwise_op::XOR;
  auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
}

// Test with null values
TYPED_TEST(GroupByBitwiseTypedTest, NullValues)
{
  using T = TypeParam;

  // Keys and values
  auto const keys = keys_column{1, 1, 2, 2, 2, 3, 3, 4};

  // Create values with some nulls
  auto const vals_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, false, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, 0x55, 0x3F, 0x2F, 0x0F, 0x42};
    }
  }();
  auto const vals_valid = std::vector<bool>{true, true, false, true, true, true, false, true};
  auto const vals       = column<T>(vals_data.begin(), vals_data.end(), vals_valid.begin());

  // Expected output - null values are excluded from aggregation
  auto const expected_keys = keys_column{1, 2, 3, 4};

  {
    auto const expected_vals = column<T>{
      static_cast<T>(vals_data[0] & vals_data[1]),  // key = 1
      static_cast<T>(vals_data[3] & vals_data[4]),  // key = 2 (vals_data[2] is null and excluded)
      static_cast<T>(vals_data[5]),                 // key = 3 (vals_data[6] is null and excluded)
      static_cast<T>(vals_data[7])                  // key = 4
    };

    auto constexpr bit_op           = cudf::bitwise_op::AND;
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
  }

  {
    auto const expected_vals = column<T>{
      static_cast<T>(vals_data[0] | vals_data[1]),  // key = 1
      static_cast<T>(vals_data[3] | vals_data[4]),  // key = 2 (vals_data[2] is null and excluded)
      static_cast<T>(vals_data[5]),                 // key = 3 (vals_data[6] is null and excluded)
      static_cast<T>(vals_data[7])                  // key = 4
    };

    auto constexpr bit_op           = cudf::bitwise_op::OR;
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
  }

  {
    auto const expected_vals = column<T>{
      static_cast<T>(vals_data[0] ^ vals_data[1]),  // key = 1
      static_cast<T>(vals_data[3] ^ vals_data[4]),  // key = 2 (vals_data[2] is null and excluded)
      static_cast<T>(vals_data[5]),                 // key = 3 (vals_data[6] is null and excluded)
      static_cast<T>(vals_data[7])                  // key = 4
    };

    auto constexpr bit_op           = cudf::bitwise_op::XOR;
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
  }
}

// Test with all null values for a group
TYPED_TEST(GroupByBitwiseTypedTest, AllNullValuesInGroup)
{
  using T = TypeParam;

  // Keys and values
  auto const keys = keys_column{1, 1, 2, 2, 3, 3, 4};

  // Create values with group 3 having all nulls
  auto const vals_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, false, false, false, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, 0x55, 0x2F, 0x0F, 0x42};
    }
  }();
  auto const vals_valid = std::vector<bool>{true, true, true, true, false, false, true};
  auto const vals       = column<T>(vals_data.begin(), vals_data.end(), vals_valid.begin());

  // Expected output - groups with all null values get a null result
  auto const expected_keys = keys_column{1, 2, 3, 4};
  {
    auto const expected_vals_data = std::vector<T>{
      static_cast<T>(vals_data[0] & vals_data[1]),  // key = 1
      static_cast<T>(vals_data[2] & vals_data[3]),  // key = 2
      static_cast<T>(0),                            // key = 3 (all nulls)
      static_cast<T>(vals_data[6])                  // key = 4
    };
    auto const expected_vals_valid = std::vector<bool>{true, true, false, true};
    auto const expected_vals =
      column<T>(expected_vals_data.begin(), expected_vals_data.end(), expected_vals_valid.begin());

    auto constexpr bit_op           = cudf::bitwise_op::AND;
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
  }

  {
    auto const expected_vals_data = std::vector<T>{
      static_cast<T>(vals_data[0] | vals_data[1]),  // key = 1
      static_cast<T>(vals_data[2] | vals_data[3]),  // key = 2
      static_cast<T>(0),                            // key = 3 (all nulls)
      static_cast<T>(vals_data[6])                  // key = 4
    };
    auto const expected_vals_valid = std::vector<bool>{true, true, false, true};
    auto const expected_vals =
      column<T>(expected_vals_data.begin(), expected_vals_data.end(), expected_vals_valid.begin());

    auto constexpr bit_op           = cudf::bitwise_op::OR;
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
  }

  {
    auto const expected_vals_data = std::vector<T>{
      static_cast<T>(vals_data[0] ^ vals_data[1]),  // key = 1
      static_cast<T>(vals_data[2] ^ vals_data[3]),  // key = 2
      static_cast<T>(0),                            // key = 3 (all nulls)
      static_cast<T>(vals_data[6])                  // key = 4
    };
    auto const expected_vals_valid = std::vector<bool>{true, true, false, true};
    auto const expected_vals =
      column<T>(expected_vals_data.begin(), expected_vals_data.end(), expected_vals_valid.begin());

    auto constexpr bit_op           = cudf::bitwise_op::XOR;
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_vals, out_vals->view());
  }
}

// Test with empty input
TYPED_TEST(GroupByBitwiseTypedTest, EmptyInput)
{
  using T = TypeParam;

  // Empty keys and values
  auto const keys = keys_column{};
  auto const vals = column<T>{};

  for (auto const bit_op : {cudf::bitwise_op::AND, cudf::bitwise_op::OR, cudf::bitwise_op::XOR}) {
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals, out_vals->view());
  };
}

// Test with multiple aggs
TYPED_TEST(GroupByBitwiseTypedTest, MultipleAggs)
{
  using T = TypeParam;

  // Keys and values
  auto const keys      = keys_column{1, 1, 2, 2, 2, 3, 3, 4};
  auto const vals_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true, false, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, 0x55, 0x3F, 0x2F, 0x0F, 0x42};
    }
  }();
  auto const vals = column<T>(vals_data.begin(), vals_data.end());

  // Expected output
  auto const expected_keys = keys_column{1, 2, 3, 4};

  auto const expected_and_vals = column<T>{
    static_cast<T>(vals_data[0] & vals_data[1]),                 // key = 1
    static_cast<T>(vals_data[2] & vals_data[3] & vals_data[4]),  // key = 2
    static_cast<T>(vals_data[5] & vals_data[6]),                 // key = 3
    static_cast<T>(vals_data[7])                                 // key = 4
  };

  auto const expected_or_vals = column<T>{
    static_cast<T>(vals_data[0] | vals_data[1]),                 // key = 1
    static_cast<T>(vals_data[2] | vals_data[3] | vals_data[4]),  // key = 2
    static_cast<T>(vals_data[5] | vals_data[6]),                 // key = 3
    static_cast<T>(vals_data[7])                                 // key = 4
  };

  auto const expected_xor_vals = column<T>{
    static_cast<T>(vals_data[0] ^ vals_data[1]),                 // key = 1
    static_cast<T>(vals_data[2] ^ vals_data[3] ^ vals_data[4]),  // key = 2
    static_cast<T>(vals_data[5] ^ vals_data[6]),                 // key = 3
    static_cast<T>(vals_data[7])                                 // key = 4
  };

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests.back().values = vals;
  requests.back().aggregations.emplace_back(
    cudf::make_bitwise_aggregation<cudf::groupby_aggregation>(cudf::bitwise_op::AND));
  requests.back().aggregations.emplace_back(
    cudf::make_bitwise_aggregation<cudf::groupby_aggregation>(cudf::bitwise_op::OR));
  requests.back().aggregations.emplace_back(
    cudf::make_bitwise_aggregation<cudf::groupby_aggregation>(cudf::bitwise_op::XOR));
  auto gb_obj       = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const result = gb_obj.aggregate(requests);
  EXPECT_EQ(result.second.front().results.size(), 3);

  auto const sort_order = cudf::sorted_order(result.first->view(), {}, {});
  auto const sorted_keys =
    std::move(cudf::gather(result.first->view(), *sort_order)->release().front());
  auto const sorted_and_vals = std::move(
    cudf::gather(cudf::table_view({result.second.front().results[0]->view()}), *sort_order)
      ->release()
      .front());
  auto const sorted_or_vals = std::move(
    cudf::gather(cudf::table_view({result.second.front().results[1]->view()}), *sort_order)
      ->release()
      .front());
  auto const sorted_xor_vals = std::move(
    cudf::gather(cudf::table_view({result.second.front().results[2]->view()}), *sort_order)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_and_vals, sorted_and_vals->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_or_vals, sorted_or_vals->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_xor_vals, sorted_xor_vals->view());
}

// Test with invalid data type
TEST_F(GroupByBitwiseTest, InvalidType)
{
  // Keys (integers)
  auto const keys = keys_column{1, 1, 2, 2, 3};

  // Values (float - invalid for bitwise operations)
  auto const float_vals = column<float>{1.1f, 2.2f, 3.3f, 4.4f, 5.5f};

  // String values (also invalid)
  auto const string_vals = cudf::test::strings_column_wrapper{"a", "b", "c", "d", "e"};

  EXPECT_THROW(bitwise_aggregate(keys, float_vals, cudf::bitwise_op::AND), cudf::logic_error);
  EXPECT_THROW(bitwise_aggregate(keys, string_vals, cudf::bitwise_op::AND), cudf::logic_error);
}

// Test with single entry per group
TYPED_TEST(GroupByBitwiseTypedTest, SingleEntryGroups)
{
  using T = TypeParam;

  // Keys and values - single value per key
  auto const keys      = keys_column{1, 2, 3, 4, 5};
  auto const vals_data = [] {
    if constexpr (std::is_same_v<T, bool>) {
      return std::vector<T>{true, false, true, true, true};
    } else {
      return std::vector<T>{0x1F, 0x0F, 0x33, 0x55, 0x3F};
    }
  }();
  auto const vals = column<T>(vals_data.begin(), vals_data.end());

  // Expected output - values should be unchanged
  // Test all three bitwise ops
  for (auto const bit_op : {cudf::bitwise_op::AND, cudf::bitwise_op::OR, cudf::bitwise_op::XOR}) {
    auto const [out_keys, out_vals] = bitwise_aggregate(keys, vals, bit_op);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys, out_keys->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals, out_vals->view());
  }
}
