/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {
template <typename V>
struct groupby_sum_scan_test : public cudf::test::BaseFixture {
};

inline void test_single_scan(column_view const& keys,
                             column_view const& values,
                             column_view const& expect_keys,
                             column_view const& expect_vals,
                             std::unique_ptr<aggregation>&& agg,
                             null_policy include_null_keys                  = null_policy::EXCLUDE,
                             sorted keys_are_sorted                         = sorted::NO,
                             std::vector<order> const& column_order         = {},
                             std::vector<null_order> const& null_precedence = {})
{
  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  groupby::groupby gb_obj(
    table_view({keys}), include_null_keys, keys_are_sorted, column_order, null_precedence);

  auto result = gb_obj.scan(requests);

  auto const sort_order  = sorted_order(result.first->view(), {}, {null_order::AFTER});
  auto const sorted_keys = gather(result.first->view(), *sort_order);
  auto const sorted_vals = gather(table_view({result.second[0].results[0]->view()}), *sort_order);

  // cudf::test::print(sorted_vals->get_column(0));
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_keys}), *sorted_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals, sorted_vals->get_column(0), true);
}

using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_CASE(groupby_sum_scan_test, supported_types);

// clang-format off
TYPED_TEST(groupby_sum_scan_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  fixed_width_column_wrapper<K> keys            {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V, int> vals       {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  fixed_width_column_wrapper<K> expect_keys     {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
                                             // { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  fixed_width_column_wrapper<R, int> expect_vals{0, 3, 9, 1, 5, 10, 19, 2, 9, 17};

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  fixed_width_column_wrapper<K> keys{};
  fixed_width_column_wrapper<V, int> vals{};

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R, int> expect_vals{};

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_sum_scan_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  fixed_width_column_wrapper<K> keys    ({1, 2, 3}, all_null());
  fixed_width_column_wrapper<V, int> vals{3, 4, 5};

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R, int> expect_vals{};

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_sum_scan_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  fixed_width_column_wrapper<K> keys            {1, 1, 1};
  fixed_width_column_wrapper<V, int> vals      ({3, 4, 5}, all_null());

  fixed_width_column_wrapper<K> expect_keys      {1, 1, 1};
  fixed_width_column_wrapper<R, int> expect_vals({3, 4, 5}, all_null());

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_sum_scan_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  fixed_width_column_wrapper<K> keys     ({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                          {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V, int> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                          {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                              // {1, 1, 1, 2, 2, 2, 2, 3, *, 3, 4};
  fixed_width_column_wrapper<K> expect_keys     ({1, 1, 1, 2, 2, 2, 2, 3,     3, 4}, all_valid());
                                              // {-, 3, 6, 1, 4, -, 9, 2, _, 8, -}
  fixed_width_column_wrapper<R, int> expect_vals({-1, 3, 9, 1, 5, -1, 14, 2, /**/ 10, -1},
                                                 { 0, 1, 1, 1, 1,  0,  1, 1, /**/ 1, 0});

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}
// clang-format on

}  // namespace test
}  // namespace cudf
