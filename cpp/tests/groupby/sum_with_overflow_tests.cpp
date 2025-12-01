/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

using namespace cudf::test::iterators;
using namespace numeric;

// SUM_WITH_OVERFLOW tests - supports signed integer and decimal types
template <typename V>
struct groupby_sum_with_overflow_test : public cudf::test::BaseFixture {};

using sum_with_overflow_supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                     cudf::test::FixedPointTypes>;

TYPED_TEST_SUITE(groupby_sum_with_overflow_test, sum_with_overflow_supported_types);

TYPED_TEST(groupby_sum_with_overflow_test, basic)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  if constexpr (cudf::is_fixed_point<V>()) {
    // For decimal types
    using RepType    = cudf::device_storage_type_t<V>;
    auto const scale = scale_type{0};
    auto vals =
      cudf::test::fixed_point_column_wrapper<RepType>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    // Create expected struct column with sum and overflow children
    auto sum_col      = cudf::test::fixed_point_column_wrapper<RepType>{{9, 19, 17}, scale};
    auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});

    auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

    // SUM_WITH_OVERFLOW should throw with sort-based groupby
    auto agg_sort = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
    EXPECT_THROW(
      test_single_agg(
        keys, vals, expect_keys, *expect_vals, std::move(agg_sort), force_use_sort_impl::YES),
      cudf::logic_error);
  } else {
    // For integer types
    cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Create expected struct column with sum and overflow children
    // Sum column type matches input type V
    auto sum_col      = cudf::test::fixed_width_column_wrapper<V>{9, 19, 17};
    auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});

    auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

    // SUM_WITH_OVERFLOW should throw with sort-based groupby
    auto agg_sort = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
    EXPECT_THROW(
      test_single_agg(
        keys, vals, expect_keys, *expect_vals, std::move(agg_sort), force_use_sort_impl::YES),
      cudf::logic_error);
  }

  // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
}

TYPED_TEST(groupby_sum_with_overflow_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{};

  // For integer types
  cudf::test::fixed_width_column_wrapper<V> vals{};

  // Create expected empty struct column with sum and overflow children
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>{};
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  auto expect_vals = cudf::create_structs_hierarchy(0, std::move(children), 0, {});

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
}

TYPED_TEST(groupby_sum_with_overflow_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<K> expect_keys{};

  // For integer types
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  // Create expected empty struct column with sum and overflow children
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>{};
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  auto expect_vals = cudf::create_structs_hierarchy(0, std::move(children), 0, {});

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
}

TYPED_TEST(groupby_sum_with_overflow_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};

  // Create expected struct column with sum and overflow children (null result)
  // Child columns have no null masks, only struct-level null mask matters
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>({0});
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>({false});
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  std::vector<int> validity{0};  // null struct
  auto [validity_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());
  auto expect_vals =
    cudf::create_structs_hierarchy(1, std::move(children), null_count, std::move(validity_mask));

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
}

TYPED_TEST(groupby_sum_with_overflow_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());

  // Create expected struct column with sum and overflow children
  //  { 3, 6,     1, 4, 9,   2, 8,    -}
  // Child columns have no null masks, only struct-level null mask matters
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>({9, 14, 10, 0});
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>({false, false, false, false});
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  std::vector<int> validity{1, 1, 1, 0};
  auto [validity_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());
  auto expect_vals =
    cudf::create_structs_hierarchy(4, std::move(children), null_count, std::move(validity_mask));

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
}

TYPED_TEST(groupby_sum_with_overflow_test, overflow_detection)
{
  using K = int32_t;
  using V = TypeParam;

  if constexpr (cudf::is_fixed_point<V>()) {
    using namespace numeric;
    using RepType = cudf::device_storage_type_t<V>;

    // Test decimal overflow detection for all decimal types
    auto constexpr scale = scale_type{0};  // Use scale 0 for simplicity

    // Use type-specific values that will cause overflow for decimal types
    auto constexpr type_max = cuda::std::numeric_limits<RepType>::max();
    auto constexpr type_min = cuda::std::numeric_limits<RepType>::min();

    cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 4, 1, 2, 2, 1, 3, 3, 2, 4, 4};

    // Create values that will cause overflow for this specific decimal type
    RepType large_positive  = type_max - 5;  // Close to max
    RepType small_increment = 10;            // Will cause overflow when added to large_positive
    RepType large_negative  = type_min + 5;  // Close to min (decimal types are always signed)
    RepType small_decrement = -10;           // Will cause underflow when added to large_negative

    // Use values that fit within the type range for non-overflowing groups
    RepType small_val1 = 10;
    RepType small_val2 = 20;
    RepType small_val3 = 30;
    RepType small_val4 = 40;

    cudf::test::fixed_point_column_wrapper<RepType> vals{
      {large_positive,   // Group 1: Close to max
       small_val1,       // Group 2: Small value
       small_val2,       // Group 3: Small value
       large_negative,   // Group 4: Close to min
       small_increment,  // Group 1: Will cause positive overflow
       small_val2,       // Group 2: Small value
       small_val3,       // Group 2: Small value
       large_positive,   // Group 1: Close to max (second occurrence)
       large_positive,   // Group 3: Close to max
       1,                // Group 3: Small value
       small_val4,       // Group 2: Small value
       small_decrement,  // Group 4: Will cause negative overflow
       large_negative},  // Group 4: Close to min (second occurrence)
      scale};

    cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3, 4};

    // Expected sums (with overflow handled by wrapping)
    auto overflow_sum_1 = static_cast<RepType>(static_cast<RepType>(large_positive) +
                                               static_cast<RepType>(small_increment) +
                                               static_cast<RepType>(large_positive));
    auto normal_sum_2   = static_cast<RepType>(small_val1 + small_val2 + small_val3 + small_val4);
    auto overflow_sum_3 =
      static_cast<RepType>(static_cast<RepType>(small_val2) + static_cast<RepType>(large_positive) +
                           static_cast<RepType>(1));
    auto overflow_sum_4 = static_cast<RepType>(static_cast<RepType>(large_negative) +
                                               static_cast<RepType>(small_decrement) +
                                               static_cast<RepType>(large_negative));

    cudf::test::fixed_point_column_wrapper<RepType> expect_sum_vals{
      {overflow_sum_1, normal_sum_2, overflow_sum_3, overflow_sum_4}, scale};
    cudf::test::fixed_width_column_wrapper<bool> expect_overflow_vals{true, false, true, true};

    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(expect_sum_vals.release());
    children.push_back(expect_overflow_vals.release());
    auto expect_vals = cudf::create_structs_hierarchy(4, std::move(children), 0, {});

    auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

    // Verify that sort-based groupby throws for decimals
    auto agg2 = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
    EXPECT_THROW(
      test_single_agg(
        keys, vals, expect_keys, *expect_vals, std::move(agg2), force_use_sort_impl::YES),
      cudf::logic_error);
  } else {
    using DeviceType = cudf::device_storage_type_t<V>;

    // Use type-specific values that will cause overflow for each integer type
    auto constexpr type_max = cuda::std::numeric_limits<DeviceType>::max();
    auto constexpr type_min = cuda::std::numeric_limits<DeviceType>::min();

    cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 4, 1, 2, 2, 1, 3, 3, 2, 4, 4};

    // Create values that will cause overflow for this specific type
    // Use smaller increments for smaller types to avoid immediate wrapping
    DeviceType large_positive  = type_max - 5;  // Close to max
    DeviceType small_increment = 10;            // Will cause overflow when added to large_positive
    DeviceType large_negative, small_decrement;
    if constexpr (cuda::std::is_signed_v<DeviceType>) {
      large_negative  = type_min + 5;  // Close to min (only for signed types)
      small_decrement = -10;  // Will cause underflow when added to large_negative (signed only)
    }

    // Use values that fit within the type range for non-overflowing groups
    DeviceType small_val1 = 10;
    DeviceType small_val2 = 20;
    DeviceType small_val3 = 30;
    DeviceType small_val4 = 40;

    if constexpr (cuda::std::is_signed_v<DeviceType>) {
      // For signed types: test both positive and negative overflow
      cudf::test::fixed_width_column_wrapper<V> vals{
        static_cast<V>(large_positive),   // Group 1: Close to max
        static_cast<V>(small_val1),       // Group 2: Small value
        static_cast<V>(small_val2),       // Group 3: Small value
        static_cast<V>(large_negative),   // Group 4: Close to min
        static_cast<V>(small_increment),  // Group 1: Will cause positive overflow
        static_cast<V>(small_val2),       // Group 2: Small value
        static_cast<V>(small_val3),       // Group 2: Small value
        static_cast<V>(large_positive),   // Group 1: Close to max (second occurrence)
        static_cast<V>(large_positive),   // Group 3: Close to max
        static_cast<V>(1),                // Group 3: Small value
        static_cast<V>(small_val4),       // Group 2: Small value
        static_cast<V>(small_decrement),  // Group 4: Will cause negative overflow
        static_cast<V>(large_negative)};  // Group 4: Close to min (second occurrence)

      cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3, 4};

      // Expected results: Groups 1, 3, and 4 overflow; Group 2 does not
      auto sum_col = cudf::test::fixed_width_column_wrapper<V>{
        static_cast<V>(static_cast<DeviceType>(large_positive) + small_increment +
                       static_cast<DeviceType>(large_positive)),  // Group 1: overflowed result
        static_cast<V>(small_val1 + small_val2 + small_val3 + small_val4),  // Group 2: no overflow
        static_cast<V>(static_cast<DeviceType>(small_val2) +
                       static_cast<DeviceType>(large_positive) +
                       static_cast<DeviceType>(1)),  // Group 3: overflowed result
        static_cast<V>(static_cast<DeviceType>(large_negative) + small_decrement +
                       static_cast<DeviceType>(large_negative))  // Group 4: overflowed result
      };
      auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{true, false, true, true};
      std::vector<std::unique_ptr<cudf::column>> children;
      children.push_back(sum_col.release());
      children.push_back(overflow_col.release());
      auto expect_vals = cudf::create_structs_hierarchy(4, std::move(children), 0, {});

      auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
      test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));
    } else {
      // For unsigned types: only test positive overflow
      cudf::test::fixed_width_column_wrapper<V> vals{
        static_cast<V>(large_positive),   // Group 1: Close to max
        static_cast<V>(small_val1),       // Group 2: Small value
        static_cast<V>(small_val2),       // Group 3: Small value
        static_cast<V>(small_val1),       // Group 4: Small value
        static_cast<V>(small_increment),  // Group 1: Will cause positive overflow
        static_cast<V>(small_val2),       // Group 2: Small value
        static_cast<V>(small_val3),       // Group 2: Small value
        static_cast<V>(large_positive),   // Group 1: Close to max (second occurrence)
        static_cast<V>(large_positive),   // Group 3: Close to max
        static_cast<V>(1),                // Group 3: Small value
        static_cast<V>(small_val4),       // Group 2: Small value
        static_cast<V>(small_val2),       // Group 4: Small value
        static_cast<V>(small_val3)};      // Group 4: Small value

      cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3, 4};

      // Expected results: Groups 1 and 3 overflow; Groups 2 and 4 do not
      auto sum_col = cudf::test::fixed_width_column_wrapper<V>{
        static_cast<V>(static_cast<DeviceType>(large_positive) + small_increment +
                       static_cast<DeviceType>(large_positive)),  // Group 1: overflowed result
        static_cast<V>(small_val1 + small_val2 + small_val3 + small_val4),  // Group 2: no overflow
        static_cast<V>(static_cast<DeviceType>(small_val2) +
                       static_cast<DeviceType>(large_positive) +
                       static_cast<DeviceType>(1)),           // Group 3: overflowed result
        static_cast<V>(small_val1 + small_val2 + small_val3)  // Group 4: no overflow
      };
      auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{true, false, true, false};
      std::vector<std::unique_ptr<cudf::column>> children;
      children.push_back(sum_col.release());
      children.push_back(overflow_col.release());
      auto expect_vals = cudf::create_structs_hierarchy(4, std::move(children), 0, {});

      auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
      test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));
    }

    // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
  }  // end else block for non-decimal types
}

// Test that SUM_WITH_OVERFLOW throws an error for bool type (which is not supported)
TEST(groupby_sum_with_overflow_error_test, bool_type_not_supported)
{
  using K = int32_t;
  using V = bool;  // bool type should not be supported by SUM_WITH_OVERFLOW

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals{true, false, true};

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));

  auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));

  // This should throw an exception since bool is not supported
  EXPECT_THROW(gb_obj.aggregate(requests), cudf::logic_error);
}

// Test that SUM_WITH_OVERFLOW throws an error for unsigned integer types (which are not supported)
TEST(groupby_sum_with_overflow_error_test, unsigned_types_not_supported)
{
  using K = int32_t;

  // Test uint32_t
  {
    using V = uint32_t;  // unsigned types should not be supported by SUM_WITH_OVERFLOW

    cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
    cudf::test::fixed_width_column_wrapper<V> vals{1, 2, 3};

    auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();

    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(std::move(agg));

    auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));

    // This should throw an exception since unsigned types are not supported
    EXPECT_THROW(gb_obj.aggregate(requests), cudf::logic_error);
  }

  // Test uint64_t
  {
    using V = uint64_t;  // unsigned types should not be supported by SUM_WITH_OVERFLOW

    cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
    cudf::test::fixed_width_column_wrapper<V> vals{1, 2, 3};

    auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();

    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(std::move(agg));

    auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));

    // This should throw an exception since unsigned types are not supported
    EXPECT_THROW(gb_obj.aggregate(requests), cudf::logic_error);
  }
}
