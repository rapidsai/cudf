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
struct groupby_quantile_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(groupby_quantile_test, supported_types);

TYPED_TEST(groupby_quantile_test, basic)
{
    using K = int32_t;
    using V = TypeParam;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::QUANTILE>;

    fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                          //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    fixed_width_column_wrapper<K> expect_keys { 1,       2,          3      };
                                          //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    fixed_width_column_wrapper<R> expect_vals({    3,        4.5,       7   }, all_valid());

    auto agg = cudf::experimental::make_quantile_aggregation({0.5},
                                        experimental::interpolation::LINEAR);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_quantile_test, zero_valid_keys)
{
    using K = int32_t;
    using V = TypeParam;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::QUANTILE>;

    fixed_width_column_wrapper<K> keys      ( { 1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V> vals        { 3, 4, 5};

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::experimental::make_quantile_aggregation({0.5},
                                        experimental::interpolation::LINEAR);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_quantile_test, zero_valid_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::QUANTILE>;

    fixed_width_column_wrapper<K> keys        { 1, 1, 1};
    fixed_width_column_wrapper<V> vals      ( { 3, 4, 5}, all_null() );

    fixed_width_column_wrapper<K> expect_keys { 1 };
    fixed_width_column_wrapper<R> expect_vals({ 0 }, all_null());

    auto agg = cudf::experimental::make_quantile_aggregation({0.5},
                                        experimental::interpolation::LINEAR);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_quantile_test, null_keys_and_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::QUANTILE>;

    fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                              { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V> vals(       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                              { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                          //  { 1, 1,     2, 2, 2,   3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,         3,       4}, all_valid());
                                          //  { 3, 6,     1, 4, 9,   2, 8,    -}
    fixed_width_column_wrapper<R> expect_vals({  4.5,        4,        5,     0},
                                              {   1,         1,        1,     0});

    auto agg = cudf::experimental::make_quantile_aggregation({0.5},
                                        experimental::interpolation::LINEAR);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_quantile_test, multiple_quantile)
{
    using K = int32_t;
    using V = TypeParam;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::QUANTILE>;

    fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                          //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    fixed_width_column_wrapper<K> expect_keys { 1,       2,          3      };
                                          //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    fixed_width_column_wrapper<R> expect_vals({  1.5,4.5, 3.25, 6,    4.5,7.5}, all_valid());

    auto agg = cudf::experimental::make_quantile_aggregation({0.25, 0.75},
                                        experimental::interpolation::LINEAR);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_quantile_test, interpolation_types)
{
    using K = int32_t;
    using V = TypeParam;
    using R = experimental::detail::target_type_t<V, experimental::aggregation::QUANTILE>;

    fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 2};
    fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 9};

                                          //  { 1, 1, 1,  2, 2, 2, 2,  3, 3}
    fixed_width_column_wrapper<K> expect_keys { 1,        2,           3   };


                                           //  { 0, 3, 6,  1, 4, 5, 9,  2, 7}
    fixed_width_column_wrapper<R> expect_vals1({  2.4,         4.2,       4 }, all_valid());
    auto agg1 = cudf::experimental::make_quantile_aggregation({0.4},
                                        experimental::interpolation::LINEAR);
    test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg1));

                                           //  { 0, 3, 6,  1, 4, 5, 9,  2, 7}
    fixed_width_column_wrapper<R> expect_vals2({    3,          4,        2 }, all_valid());
    auto agg2 = cudf::experimental::make_quantile_aggregation({0.4},
                                        experimental::interpolation::NEAREST);
    test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2));

                                           //  { 0, 3, 6,  1, 4, 5, 9,  2, 7}
    fixed_width_column_wrapper<R> expect_vals3({    0,          4,        2 }, all_valid());
    auto agg3 = cudf::experimental::make_quantile_aggregation({0.4},
                                        experimental::interpolation::LOWER);
    test_single_agg(keys, vals, expect_keys, expect_vals3, std::move(agg3));

                                           //  { 0, 3, 6,  1, 4, 5, 9,  2, 7}
    fixed_width_column_wrapper<R> expect_vals4({    3,          5,        7 }, all_valid());
    auto agg4 = cudf::experimental::make_quantile_aggregation({0.4},
                                        experimental::interpolation::HIGHER);
    test_single_agg(keys, vals, expect_keys, expect_vals4, std::move(agg4));

                                           //  { 0, 3, 6,  1, 4, 5, 9,  2, 7}
    fixed_width_column_wrapper<R> expect_vals5({  1.5,         4.5,      4.5}, all_valid());
    auto agg5 = cudf::experimental::make_quantile_aggregation({0.4},
                                        experimental::interpolation::MIDPOINT);
    test_single_agg(keys, vals, expect_keys, expect_vals5, std::move(agg5));

}


} // namespace test
} // namespace cudf
