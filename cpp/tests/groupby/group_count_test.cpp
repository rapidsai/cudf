/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
struct groupby_count_test : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(groupby_count_test, cudf::test::AllTypes);

// clang-format off
TYPED_TEST(groupby_count_test, basic)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

    fixed_width_column_wrapper<K> keys { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V, int> vals { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
    fixed_width_column_wrapper<R, int> expect_vals { 3, 4, 3 };

    auto agg = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg1 = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_test, empty_cols)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

    fixed_width_column_wrapper<K> keys        { };
    fixed_width_column_wrapper<V> vals;

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals;

    auto agg = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg1 = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_count_test, zero_valid_keys)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

    fixed_width_column_wrapper<K> keys( { 1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V, int> vals  { 3, 4, 5};

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R, int> expect_vals { };

    auto agg = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg1 = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_test, zero_valid_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

    fixed_width_column_wrapper<K> keys   { 1, 1, 1};
    fixed_width_column_wrapper<V, int> vals ( { 3, 4, 5}, all_null() );

    fixed_width_column_wrapper<K> expect_keys { 1 };
    fixed_width_column_wrapper<R, int> expect_vals { 0 };

    auto agg = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg1 = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

    fixed_width_column_wrapper<R, int> expect_vals2 { 3 };
    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2));
}

TYPED_TEST(groupby_count_test, null_keys_and_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

    fixed_width_column_wrapper<K> keys({ 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                       { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V, int> vals({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                       { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                          //  { 1, 1,     2, 2, 2,   3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,         3,       4}, all_valid());
                                          //  { 3, 6,     1, 4, 9,   2, 8,    -}
    fixed_width_column_wrapper<R, int> expect_vals { 2,        3,         2,       0};

    auto agg = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg1 = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

    fixed_width_column_wrapper<R, int> expect_vals2{ 3,        4,         2,       1};
    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2));

}

struct groupby_count_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_count_string_test, basic)
{
    using K = int32_t;
    using V = cudf::string_view;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

    fixed_width_column_wrapper<K> keys        {   1,   3,   3,   5,   5,   0};
    strings_column_wrapper        vals        { "1", "1", "1", "1", "1", "1"};

    fixed_width_column_wrapper<K> expect_keys   {   0,   1,   3,   5};
    fixed_width_column_wrapper<R, int> expect_vals   {   1,   1,   2,   2};

    auto agg = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg1 = cudf::make_count_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);
}
// clang-format on

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, GroupByCount)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  using K = int32_t;
  using V = decimalXX;
  using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

  auto const scale = scale_type{-1};
  auto const keys  = fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals  = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

  auto const expect_keys = fixed_width_column_wrapper<K>{1, 2, 3};
  auto const expect_vals = fixed_width_column_wrapper<R, int>{3, 4, 3};

  auto agg = cudf::make_count_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

  auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(FixedPointTestBothReps, GroupBySumProductMinMaxDecimalAsValue)
{
  using namespace numeric;
  using decimalXX    = TypeParam;
  using RepType      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper   = cudf::test::fixed_point_column_wrapper<RepType>;
  using fp64_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;
  using K            = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    auto const keys  = fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    auto const expect_keys = fixed_width_column_wrapper<K>{1, 2, 3};

    auto const expect_vals_sum = fp64_wrapper{{9, 19, 17}, scale};
    auto const expect_vals_min = fp_wrapper{{0, 1, 2}, scale};
    auto const expect_vals_max = fp_wrapper{{6, 9, 8}, scale};

    auto agg1 = cudf::make_sum_aggregation();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_sum, std::move(agg1), force_use_sort_impl::YES);

    auto agg2 = cudf::make_min_aggregation();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_min, std::move(agg2), force_use_sort_impl::YES);

    auto agg3 = cudf::make_max_aggregation();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_max, std::move(agg3), force_use_sort_impl::YES);

    auto agg4 = cudf::make_product_aggregation();
    EXPECT_THROW(
      test_single_agg(keys, vals, expect_keys, {}, std::move(agg4), force_use_sort_impl::YES),
      cudf::logic_error);
  }
}

TYPED_TEST(FixedPointTestBothReps, GroupBySumProductMinMaxDecimalAsValueAndKey)
{
  using namespace numeric;
  using decimalXX    = TypeParam;
  using RepType      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper   = cudf::test::fixed_point_column_wrapper<RepType>;
  using fp64_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    auto const keys  = fp_wrapper{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}, scale};
    auto const vals  = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    auto const expect_keys = fp_wrapper{{1, 2, 3}, scale};

    auto const expect_vals_sum = fp64_wrapper{{9, 19, 17}, scale};
    auto const expect_vals_min = fp_wrapper{{0, 1, 2}, scale};
    auto const expect_vals_max = fp_wrapper{{6, 9, 8}, scale};

    auto agg1 = cudf::make_sum_aggregation();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_sum, std::move(agg1), force_use_sort_impl::YES);

    auto agg2 = cudf::make_min_aggregation();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_min, std::move(agg2), force_use_sort_impl::YES);

    auto agg3 = cudf::make_max_aggregation();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_max, std::move(agg3), force_use_sort_impl::YES);

    auto agg4 = cudf::make_product_aggregation();
    EXPECT_THROW(
      test_single_agg(keys, vals, expect_keys, {}, std::move(agg4), force_use_sort_impl::YES),
      cudf::logic_error);
  }
}

struct groupby_dictionary_count_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_dictionary_count_test, basic)
{
  using K = int32_t;
  using V = std::string;
  using R = cudf::detail::target_type_t<V, aggregation::COUNT_VALID>;

  // clang-format off
  strings_column_wrapper       keys{"1", "3", "3", "5", "5", "0"};
  dictionary_column_wrapper<K> vals{ 1,   1,   1,   1,   1,   1};
  strings_column_wrapper             expect_keys{"0", "1", "3", "5"};
  fixed_width_column_wrapper<R, int> expect_vals{ 1,   1,   2,   2};
  // clang-format on

  test_single_agg(keys, vals, expect_keys, expect_vals, cudf::make_count_aggregation());
  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_count_aggregation(), force_use_sort_impl::YES);
}

}  // namespace test
}  // namespace cudf
