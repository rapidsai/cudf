/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

namespace cudf {
namespace test {
template <typename V>
struct groupby_keys_test : public cudf::test::BaseFixture {
};

using supported_types = cudf::test::
  Types<int8_t, int16_t, int32_t, int64_t, float, double, numeric::decimal32, numeric::decimal64>;

TYPED_TEST_SUITE(groupby_keys_test, supported_types);

TYPED_TEST(groupby_keys_test, basic)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

  // clang-format off
  fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
  fixed_width_column_wrapper<R> expect_vals { 3, 4, 3 };
  // clang-format on

  auto agg = cudf::make_count_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, zero_valid_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

  // clang-format off
  fixed_width_column_wrapper<K> keys      ( { 1, 2, 3}, all_nulls() );
  fixed_width_column_wrapper<V> vals        { 3, 4, 5};

  fixed_width_column_wrapper<K> expect_keys { };
  fixed_width_column_wrapper<R> expect_vals { };
  // clang-format on

  auto agg = cudf::make_count_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, some_null_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

  // clang-format off
  fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                            { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                        //  { 1, 1, 1,  2, 2, 2, 2,  3, 3,  4}
  fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3,     4}, no_nulls() );
                                        //  { 0, 3, 6,  1, 4, 5, 9,  2, 8,  -}
  fixed_width_column_wrapper<R> expect_vals { 3,        4,           2,     1};
  // clang-format on

  auto agg = cudf::make_count_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, include_null_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  // clang-format off
  fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                            { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                        //  { 1, 1, 1,  2, 2, 2, 2,  3, 3,  4,  -}
  fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3,     4,  3},
                                            { 1,        1,           1,     1,  0});
                                        //  { 0, 3, 6,  1, 4, 5, 9,  2, 8,  -,  -}
  fixed_width_column_wrapper<R> expect_vals { 9,        19,          10,    4,  7};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::NO,
                  null_policy::INCLUDE);
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  // clang-format off
  fixed_width_column_wrapper<K> keys        { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4};
  fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

  fixed_width_column_wrapper<K> expect_keys { 1,       2,          3,       4};
  fixed_width_column_wrapper<R> expect_vals { 3,       18,         24,      4};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  null_policy::EXCLUDE,
                  sorted::YES);
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_descending)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  // clang-format off
  fixed_width_column_wrapper<K> keys        { 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1};
  fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

  fixed_width_column_wrapper<K> expect_keys { 4, 3,       2,          1      };
  fixed_width_column_wrapper<R> expect_vals { 0, 6,       22,        21      };
  // clang-format on

  auto agg = cudf::make_sum_aggregation<groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  null_policy::EXCLUDE,
                  sorted::YES,
                  {order::DESCENDING});
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_nullable)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  // clang-format off
  fixed_width_column_wrapper<K> keys(       { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4},
                                            { 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

  fixed_width_column_wrapper<K> expect_keys({ 1,       2,          3,       4}, no_nulls() );
  fixed_width_column_wrapper<R> expect_vals { 3,       15,         17,      4};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  null_policy::EXCLUDE,
                  sorted::YES);
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_nulls_before_include_nulls)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  // clang-format off
  fixed_width_column_wrapper<K> keys(       { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4},
                                            { 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                        //  { 1, 1, 1,  -, -,  2, 2,  -,  3, 3,  4}
  fixed_width_column_wrapper<K> expect_keys({ 1,        2,     2,     3,  3,     4},
                                            { 1,        0,     1,     0,  1,     1});
  fixed_width_column_wrapper<R> expect_vals { 3,        7,     11,    7,  17,    4};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  null_policy::INCLUDE,
                  sorted::YES);
}

TYPED_TEST(groupby_keys_test, mismatch_num_rows)
{
  using K = TypeParam;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys{1, 2, 3};
  fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4};

  auto agg = cudf::make_count_aggregation<groupby_aggregation>();
  CUDF_EXPECT_THROW_MESSAGE(test_single_agg(keys, vals, keys, vals, std::move(agg)),
                            "Size mismatch between request values and groupby keys.");
  auto agg2 = cudf::make_count_aggregation<groupby_scan_aggregation>();
  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys, vals, keys, vals, std::move(agg2)),
                            "Size mismatch between request values and groupby keys.");
}

struct groupby_string_keys_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_string_keys_test, basic)
{
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  // clang-format off
  strings_column_wrapper        keys        { "aaa", "año", "₹1", "aaa", "año", "año", "aaa", "₹1", "₹1", "año"};
  fixed_width_column_wrapper<V> vals        {     0,     1,    2,     3,     4,     5,     6,    7,    8,     9};

  strings_column_wrapper        expect_keys({ "aaa", "año", "₹1" });
  fixed_width_column_wrapper<R> expect_vals {     9,    19,   17 };
  // clang-format on

  auto agg = cudf::make_sum_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
// clang-format on

struct groupby_dictionary_keys_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_dictionary_keys_test, basic)
{
  using K = std::string;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, aggregation::SUM>;

  // clang-format off
  dictionary_column_wrapper<K> keys { "aaa", "año", "₹1", "aaa", "año", "año", "aaa", "₹1", "₹1", "año"};
  fixed_width_column_wrapper<V> vals{     0,     1,    2,     3,     4,     5,     6,    7,    8,     9};
  dictionary_column_wrapper<K>expect_keys  ({ "aaa", "año", "₹1" });
  fixed_width_column_wrapper<R> expect_vals({     9,    19,   17 });
  // clang-format on

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_sum_aggregation<groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_sum_aggregation<groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

struct groupby_cache_test : public cudf::test::BaseFixture {
};

// To check if the cache doesn't insert multiple times to cache for same aggregation on a column in
// same request.
// If this test fails, then insert happened and key stored in cache map becomes dangling reference.
// Any comparison with same aggregation as key will fail.
TEST_F(groupby_cache_test, duplicate_agggregations)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  groupby::groupby gb_obj(table_view({keys}));

  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<groupby_aggregation>());

  // hash groupby
  EXPECT_NO_THROW(gb_obj.aggregate(requests));

  // sort groupby
  // WAR to force groupby to use sort implementation
  requests[0].aggregations.push_back(make_nth_element_aggregation<groupby_aggregation>(0));
  EXPECT_NO_THROW(gb_obj.aggregate(requests));
}

// To check if the cache doesn't insert multiple times to cache for same aggregation on same column
// but in different requests.
// If this test fails, then insert happened and key stored in cache map becomes dangling reference.
// Any comparison with same aggregation as key will fail.
TEST_F(groupby_cache_test, duplicate_columns)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  groupby::groupby gb_obj(table_view({keys}));

  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<groupby_aggregation>());
  requests.emplace_back(groupby::aggregation_request());
  requests[1].values = vals;
  requests[1].aggregations.push_back(cudf::make_sum_aggregation<groupby_aggregation>());

  // hash groupby
  EXPECT_NO_THROW(gb_obj.aggregate(requests));

  // sort groupby
  // WAR to force groupby to use sort implementation
  requests[0].aggregations.push_back(make_nth_element_aggregation<groupby_aggregation>(0));
  EXPECT_NO_THROW(gb_obj.aggregate(requests));
}

}  // namespace test
}  // namespace cudf
