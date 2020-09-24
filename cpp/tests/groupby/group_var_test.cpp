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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {
template <typename V>
struct groupby_var_test : public cudf::test::BaseFixture {
};

using supported_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(groupby_var_test, supported_types);

// clang-format off
TYPED_TEST(groupby_var_test, basic)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::VARIANCE>;

    fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                          //  { 1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
    fixed_width_column_wrapper<K> expect_keys { 1,        2,           3      };
                                          //  { 0, 3, 6,  1, 4, 5, 9,  2, 7, 8}
    fixed_width_column_wrapper<R> expect_vals({   9.,      131./12,     31./3 }, all_valid());

    auto agg = cudf::make_variance_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, empty_cols)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::VARIANCE>;

    fixed_width_column_wrapper<K> keys        { };
    fixed_width_column_wrapper<V> vals        { };

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::make_variance_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, zero_valid_keys)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::VARIANCE>;

    fixed_width_column_wrapper<K> keys      ( { 1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V> vals        { 3, 4, 5};

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::make_variance_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, zero_valid_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::VARIANCE>;

    fixed_width_column_wrapper<K> keys        { 1, 1, 1};
    fixed_width_column_wrapper<V> vals      ( { 3, 4, 5}, all_null() );

    fixed_width_column_wrapper<K> expect_keys { 1 };
    fixed_width_column_wrapper<R> expect_vals({ 0 }, all_null());

    auto agg = cudf::make_variance_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, null_keys_and_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::VARIANCE>;

    fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                              { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V> vals(       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3},
                                              { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

                                          //  { 1, 1,     2, 2, 2,   3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,         3,       4}, all_valid());
                                          //  { 3, 6,     1, 4, 9,   2, 8,    3}
    fixed_width_column_wrapper<R> expect_vals({ 4.5,      49./3,    18.,     0.},
                                              { 1,        1,         1,       0});

    auto agg = cudf::make_variance_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, ddof_non_default)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::VARIANCE>;

    fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                              { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V> vals(       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3},
                                              { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

                                          //  { 1, 1,     2, 2, 2,   3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,         3,       4}, all_valid());
                                          //  { 3, 6,     1, 4, 9,   2, 8,    3}
    fixed_width_column_wrapper<R> expect_vals({ 0.,       98./3,    0.,      0.},
                                              { 0,        1,         0,       0});

    auto agg = cudf::make_variance_aggregation(2);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
// clang-format on

}  // namespace test
}  // namespace cudf
