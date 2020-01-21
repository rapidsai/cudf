/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/groupby/common/groupby_test_util.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {


template <typename V>
struct groupby_keys_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(groupby_keys_test, supported_types);

TYPED_TEST(groupby_keys_test, basic)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::COUNT>;

    fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
    fixed_width_column_wrapper<R> expect_vals { 3, 4, 3 };

    auto agg = cudf::experimental::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, zero_valid_keys)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::COUNT>;

    fixed_width_column_wrapper<K> keys      ( { 1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V> vals        { 3, 4, 5};

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::experimental::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, some_null_keys)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::COUNT>;

    fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                              { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                          //  { 1, 1, 1,  2, 2, 2, 2,  3, 3,  4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3,     4}, all_valid());
                                          //  { 0, 3, 6,  1, 4, 5, 9,  2, 8,  -}
    fixed_width_column_wrapper<R> expect_vals { 3,        4,           2,     1};

    auto agg = cudf::experimental::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, include_null_keys)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::SUM>;

    fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                              { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                          //  { 1, 1, 1,  2, 2, 2, 2,  3, 3,  4,  -}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3,     4,  3},
                                              { 1,        1,           1,     1,  0});
                                          //  { 0, 3, 6,  1, 4, 5, 9,  2, 8,  -,  -}
    fixed_width_column_wrapper<R> expect_vals { 9,        19,          10,    4,  7};

    auto agg = cudf::experimental::make_sum_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg),
        false); //< Don't ignore nulls in keys
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::SUM>;

    fixed_width_column_wrapper<K> keys        { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4};
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

    fixed_width_column_wrapper<K> expect_keys { 1,       2,          3,       4};
    fixed_width_column_wrapper<R> expect_vals { 3,       18,         24,      4};

    auto agg = cudf::experimental::make_sum_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), true,
        true); //< keys_are_sorted
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_descending)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::SUM>;

    fixed_width_column_wrapper<K> keys        { 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1};
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

    fixed_width_column_wrapper<K> expect_keys { 4, 3,       2,          1      };
    fixed_width_column_wrapper<R> expect_vals { 0, 6,       22,        21      };

    auto agg = cudf::experimental::make_sum_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), true,
        true,
        {order::DESCENDING}); //< keys_are_sorted
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_nullable)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::SUM>;

    fixed_width_column_wrapper<K> keys(       { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4},
                                              { 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

    fixed_width_column_wrapper<K> expect_keys({ 1,       2,          3,       4}, all_valid());
    fixed_width_column_wrapper<R> expect_vals { 3,       15,         17,      4};

    auto agg = cudf::experimental::make_sum_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), true,
        true /* keys_are_sorted */); 
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_nulls_before_include_nulls)
{
    using K = TypeParam;
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::SUM>;

    fixed_width_column_wrapper<K> keys(       { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4},
                                              { 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                          //  { 1, 1, 1,  -, -,  2, 2,  -,  3, 3,  4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,     2,     3,  3,     4},
                                              { 1,        0,     1,     0,  1,     1});
    fixed_width_column_wrapper<R> expect_vals { 3,        7,     11,    7,  17,    4};

    auto agg = cudf::experimental::make_sum_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg),
        false /* don't ignore null keys */, 
        true /* keys_are_sorted */); 
}

struct groupby_string_keys_test : public cudf::test::BaseFixture {};

TEST_F(groupby_string_keys_test, basic)
{
    using V = int32_t;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::SUM>;

    strings_column_wrapper        keys        { "aaa", "año", "₹1", "aaa", "año", "año", "aaa", "₹1", "₹1", "año"};
    fixed_width_column_wrapper<V> vals        {     0,     1,    2,     3,     4,     5,     6,    7,    8,     9};

    strings_column_wrapper        expect_keys({ "aaa", "año", "₹1" });
    fixed_width_column_wrapper<R> expect_vals {     9,    19,   17 };

    auto agg = cudf::experimental::make_sum_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

} // namespace test
} // namespace cudf
