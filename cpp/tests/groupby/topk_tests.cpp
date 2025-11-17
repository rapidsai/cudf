/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

template <typename V>
struct groupby_topk_test_types : public cudf::test::BaseFixture {};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes,
                                                  cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(groupby_topk_test_types, FixedWidthTypesNotBool);

TYPED_TEST(groupby_topk_test_types, FixedWidth)
{
  using K = int32_t;
  using V = TypeParam;

  auto keys   = cudf::test::fixed_width_column_wrapper<K, int32_t>{1, 1, 1, 2, 2, 2};
  auto values = cudf::test::fixed_width_column_wrapper<V, int32_t>{1, 2, 3, 4, 5, 6};

  auto expect_keys = cudf::test::fixed_width_column_wrapper<K, int32_t>{1, 2};
  auto expect_vals = cudf::test::lists_column_wrapper<V, int32_t>{{3, 2, 1}, {6, 5, 4}};

  auto agg = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(3);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));

  expect_vals = cudf::test::lists_column_wrapper<V, int32_t>{{1, 2, 3}, {4, 5, 6}};
  agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(4, cudf::order::ASCENDING);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

struct groupby_topk_test : public cudf::test::BaseFixture {};

TEST_F(groupby_topk_test, Strings)
{
  auto keys   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 1, 1, 2, 2, 2});
  auto values = cudf::test::strings_column_wrapper({"a", "b", "c", "d", "e", "f"});

  auto expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  auto expect_vals = cudf::test::lists_column_wrapper<cudf::string_view>({{"c", "b"}, {"f", "e"}});

  auto agg = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(2);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));

  agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(2, cudf::order::ASCENDING);
  expect_vals = cudf::test::lists_column_wrapper<cudf::string_view>({{"a", "b"}, {"d", "e"}});
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_topk_test, EmptyInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{};
  cudf::test::fixed_width_column_wrapper<int32_t> values{};

  cudf::test::fixed_width_column_wrapper<int32_t> expect_keys{};
  cudf::test::lists_column_wrapper<int32_t> expect_vals{};

  auto agg = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(3);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}
