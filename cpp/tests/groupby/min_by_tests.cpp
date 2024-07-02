/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
struct groupby_min_by_test : public cudf::test::BaseFixture {};
using K = int32_t;

TYPED_TEST_SUITE(groupby_min_by_test, cudf::test::FixedWidthTypes);

TYPED_TEST(groupby_min_by_test, basic)
{
  using V = TypeParam;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<K> values{4, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> orders{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::test::structs_column_wrapper vals{values, orders};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<K> expect_values{4, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> expect_orders{1, 2, 3};
  cudf::test::structs_column_wrapper expect_vals{expect_values, expect_orders};

  auto agg = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_min_by_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_min_by_string_test, basic)
{
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<K> values{4, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::strings_column_wrapper orders{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
  cudf::test::structs_column_wrapper vals{values, orders};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<K> expect_values{3, 5, 7};
  cudf::test::strings_column_wrapper expect_orders{"aaa", "bat", "$1"};
  cudf::test::structs_column_wrapper expect_vals{expect_values, expect_orders};

  auto agg = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_by_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}
