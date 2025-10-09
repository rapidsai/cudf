/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_sum_test : public cudf::test::BaseFixture {};

using K = int32_t;
using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_SUITE(groupby_sum_test, supported_types);

TYPED_TEST(groupby_sum_test, basic)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{9, 19, 17};

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_sum_test, empty_cols)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_sum_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_sum_test, zero_valid_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, cudf::test::iterators::all_nulls());

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_sum_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());
  //  { 3, 6,     1, 4, 9,   2, 8,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({9, 14, 10, 0}, {1, 1, 1, 0});

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_sum_test, dictionary)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{9, 19, 17};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

struct overflow_test : public cudf::test::BaseFixture {};
TEST_F(overflow_test, overflow_integer)
{
  using int32_col = cudf::test::fixed_width_column_wrapper<int32_t>;
  using int64_col = cudf::test::fixed_width_column_wrapper<int64_t>;

  auto const keys        = int32_col{0, 0};
  auto const vals        = int32_col{-2147483648, -2147483648};
  auto const expect_keys = int32_col{0};
  auto const expect_vals = int64_col{-4294967296L};

  auto test_sum = [&](auto const use_sort) {
    auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), use_sort);
  };

  test_sum(force_use_sort_impl::NO);
  test_sum(force_use_sort_impl::YES);
}

template <typename T>
struct GroupBySumFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupBySumFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupBySumFixedPointTest, GroupBySortSumDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_sum = fp_wrapper{{9, 19, 17}, scale};

    auto agg1 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_sum, std::move(agg1), force_use_sort_impl::YES);

    auto agg4 = cudf::make_product_aggregation<cudf::groupby_aggregation>();
    EXPECT_THROW(
      test_single_agg(keys, vals, expect_keys, {}, std::move(agg4), force_use_sort_impl::YES),
      cudf::logic_error);
  }
}

TYPED_TEST(GroupBySumFixedPointTest, GroupByHashSumDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_sum = fp_wrapper{{9, 19, 17}, scale};

    auto agg5 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals_sum, std::move(agg5));

    auto agg8 = cudf::make_product_aggregation<cudf::groupby_aggregation>();
    EXPECT_THROW(test_single_agg(keys, vals, expect_keys, {}, std::move(agg8)), cudf::logic_error);
  }
}

// SUM_WITH_OVERFLOW tests - only supports int64_t input values and outputs int64_t
template <typename V>
struct groupby_sum_with_overflow_test : public cudf::test::BaseFixture {};

using sum_with_overflow_supported_types = cudf::test::Types<int64_t>;

TYPED_TEST_SUITE(groupby_sum_with_overflow_test, sum_with_overflow_supported_types);

TYPED_TEST(groupby_sum_with_overflow_test, basic)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  // Create expected struct column with sum and overflow children
  auto sum_col      = cudf::test::fixed_width_column_wrapper<int64_t>{9, 19, 17};
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
}

TYPED_TEST(groupby_sum_with_overflow_test, empty_cols)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};

  // Create expected empty struct column with sum and overflow children
  auto sum_col      = cudf::test::fixed_width_column_wrapper<int64_t>{};
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
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};

  // Create expected empty struct column with sum and overflow children
  auto sum_col      = cudf::test::fixed_width_column_wrapper<int64_t>{};
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
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};

  // Create expected struct column with sum and overflow children (null result)
  // Child columns have no null masks, only struct-level null mask matters
  auto sum_col      = cudf::test::fixed_width_column_wrapper<int64_t>({0});
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
  auto sum_col      = cudf::test::fixed_width_column_wrapper<int64_t>({9, 14, 10, 0});
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
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 4, 1, 2, 2, 1, 3, 3, 2, 4, 4};
  // Mix of values that will cause positive and negative overflow for some groups but not others
  cudf::test::fixed_width_column_wrapper<V> vals{
    9223372036854775800L,    // Close to INT64_MAX
    100L,                    // Small value
    200L,                    // Small value
    -9223372036854775800L,   // Close to INT64_MIN
    20L,                     // Small value that will cause positive overflow when added to first
    200L,                    // Small value
    300L,                    // Small value
    9223372036854775800L,    // Close to INT64_MAX
    9223372036854775800L,    // Close to INT64_MAX
    1L,                      // Small value
    400L,                    // Small value
    -20L,                    // Small value that will cause negative overflow when added to fourth
    -9223372036854775800L};  // Close to INT64_MIN

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3, 4};

  // Create expected struct column with sum and overflow children
  // Group 1: 9223372036854775800 + 20 + 9223372036854775800 = positive overflow
  // Group 2: 100 + 200 + 300 + 400 = 1000 (no overflow)
  // Group 3: 200 + 9223372036854775800 + 1 = positive overflow
  // Group 4: -9223372036854775800 + (-20) + (-9223372036854775800) = negative overflow
  auto sum_col = cudf::test::fixed_width_column_wrapper<int64_t>{
    4L,                     // Positive overflow result for group 1
    1000L,                  // Normal sum for group 2 (no overflow)
    -9223372036854775615L,  // Positive overflow result for group 3
    -4L                     // Negative overflow result for group 4
  };
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{true, false, true, true};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  auto expect_vals = cudf::create_structs_hierarchy(4, std::move(children), 0, {});

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Note: SUM_WITH_OVERFLOW only works with hash groupby, not sort groupby
}

// Test that SUM_WITH_OVERFLOW throws an error for invalid value types
TEST(groupby_sum_with_overflow_error_test, invalid_value_type)
{
  using K = int32_t;
  using V = int32_t;  // Invalid type for SUM_WITH_OVERFLOW, should only support int64_t

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1, 2, 2, 2, 3, 3, 3};
  cudf::test::fixed_width_column_wrapper<V> vals{1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  auto agg = cudf::make_sum_with_overflow_aggregation<cudf::groupby_aggregation>();

  // SUM_WITH_OVERFLOW should throw a logic_error when used with non-int64_t value types
  EXPECT_THROW(
    test_single_agg(keys, vals, expect_keys, {}, std::move(agg), force_use_sort_impl::NO),
    cudf::logic_error);
}
