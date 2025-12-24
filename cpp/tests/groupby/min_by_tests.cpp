/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_min_by_test : public cudf::test::BaseFixture {};
using K = int32_t;

TYPED_TEST_SUITE(groupby_min_by_test, cudf::test::NumericTypes);

TYPED_TEST(groupby_min_by_test, basic)
{
  using V = TypeParam;
  using R = TypeParam;

  // Keys: groupby keys
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  
  // Ordering column (first child): values to find min
  cudf::test::fixed_width_column_wrapper<V> ordering{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  
  // Value column (second child): values to return
  cudf::test::fixed_width_column_wrapper<R> vals{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  // Create struct column with ordering and value columns
  cudf::test::structs_column_wrapper struct_col{{ordering, vals}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  // For group 1: min ordering is 3 at index 6, value is 70
  // For group 2: min ordering is 0 at index 9, value is 100
  // For group 3: min ordering is 1 at index 8, value is 90
  cudf::test::fixed_width_column_wrapper<R> expect_vals{70, 100, 90};

  auto agg = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, struct_col, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, struct_col, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_min_by_test, with_nulls)
{
  using V = TypeParam;
  using R = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2},
                                                 {1, 1, 1, 1, 1, 1, 1, 0, 1, 1});
  
  cudf::test::fixed_width_column_wrapper<V> ordering({9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
                                                     {1, 1, 1, 1, 1, 0, 0, 1, 1, 1});
  
  cudf::test::fixed_width_column_wrapper<R> vals{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  cudf::test::structs_column_wrapper struct_col{{ordering, vals}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3}, {1, 1, 1});
  // Group 1: min non-null ordering is 6, value is 40
  // Group 2: min non-null ordering is 0, value is 100
  // Group 3: min non-null ordering is 1, value is 90
  cudf::test::fixed_width_column_wrapper<R> expect_vals({40, 100, 90}, {1, 1, 1});

  auto agg = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, struct_col, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, struct_col, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}
