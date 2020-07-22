/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {
template <typename V>
struct groupby_nunique_test : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(groupby_nunique_test, cudf::test::AllTypes);

// clang-format off
TYPED_TEST(groupby_nunique_test, basic)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V, int32_t> vals(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
    fixed_width_column_wrapper<R> expect_vals { 3, 4, 3 };
    fixed_width_column_wrapper<R> expect_bool_vals { 2, 1, 1 };

    auto agg = cudf::make_nunique_aggregation();
    if(std::is_same<V, bool>())
        test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
    else
        test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, empty_cols)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys        { };
    fixed_width_column_wrapper<V> vals        { };

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::make_nunique_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, basic_duplicates)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 3, 2, 2, 9});

    fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
    fixed_width_column_wrapper<R> expect_vals { 2, 4, 1 };
    fixed_width_column_wrapper<R> expect_bool_vals { 2, 1, 1 };

    auto agg = cudf::make_nunique_aggregation();
    if(std::is_same<V, bool>())
        test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
    else
        test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, zero_valid_keys)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys( { 1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5});

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::make_nunique_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, zero_valid_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys  { 1, 1, 1};
    fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5}, all_null());

    fixed_width_column_wrapper<K> expect_keys { 1 };
    fixed_width_column_wrapper<R> expect_vals { 0 };

    auto agg = cudf::make_nunique_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, null_keys_and_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys({ 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                       { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                          //  { 1, 1,     2, 2, 2,   3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,         3,       4}, all_valid());
                // all unique values only //  { 3, 6,     1, 4, 9,   2, 8,    -}                                   
    fixed_width_column_wrapper<R> expect_vals { 2,        3,         2,       0};
    fixed_width_column_wrapper<R> expect_bool_vals { 1, 1, 1, 0};


    auto agg = cudf::make_nunique_aggregation();
    if(std::is_same<V, bool>())
        test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
    else 
        test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, null_keys_and_values_with_duplicates)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys({ 1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
                                       { 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});
    fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2},
                                                {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0});

                                          //  { 1, 1,     2, 2, 2,    3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,          3,       4}, all_valid());
                                          //  { 3, 6,-    1, 4, 9,-   2*, 8,   -*}
                                          //  unique,     with null,  dup,     dup null
    fixed_width_column_wrapper<R> expect_vals { 2,        3,          2,       0};
    fixed_width_column_wrapper<R> expect_bool_vals { 1, 1, 1, 0};


    auto agg = cudf::make_nunique_aggregation();
    if(std::is_same<V, bool>())
        test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
    else 
        test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}


TYPED_TEST(groupby_nunique_test, include_nulls)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::NUNIQUE>;

    fixed_width_column_wrapper<K> keys({ 1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
                                       { 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});
    fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2},
                                                {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0});

                                          //  { 1, 1,     2, 2, 2,    3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,          3,       4}, all_valid());
                                          //  { 3, 6,-    1, 4, 9,-   2*, 8,   -*}
                                          //  unique,     with null,  dup,     dup null
    fixed_width_column_wrapper<R> expect_vals { 3,        4,          2,       1};
    fixed_width_column_wrapper<R> expect_bool_vals { 2, 2, 1, 1};


    auto agg = cudf::make_nunique_aggregation(null_policy::INCLUDE);
    if(std::is_same<V, bool>())
        test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
    else 
        test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
// clang-format on

}  // namespace test
}  // namespace cudf
