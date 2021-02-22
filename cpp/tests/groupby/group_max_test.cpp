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
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/dictionary/update_keys.hpp>

namespace cudf {
namespace test {
template <typename V>
struct groupby_max_test : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(groupby_max_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

// clang-format off
TYPED_TEST(groupby_max_test, basic)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::MAX>;

    fixed_width_column_wrapper<K> keys { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V, int32_t> vals({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
    fixed_width_column_wrapper<R, int32_t> expect_vals({ 6, 9, 8 });

    auto agg = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg2 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, empty_cols)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::MAX>;

    fixed_width_column_wrapper<K> keys        { };
    fixed_width_column_wrapper<V> vals        { };

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg2 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, zero_valid_keys)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::MAX>;

    fixed_width_column_wrapper<K> keys( { 1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5});

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };

    auto agg = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg2 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, zero_valid_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::MAX>;

    fixed_width_column_wrapper<K> keys   { 1, 1, 1};
    fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5}, all_null());

    fixed_width_column_wrapper<K> expect_keys { 1 };
    fixed_width_column_wrapper<R, int32_t> expect_vals({ 0 }, all_null());

    auto agg = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg2 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, null_keys_and_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::MAX>;

    fixed_width_column_wrapper<K> keys({ 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                       { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                {1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0});

                                          //  { 1, 1,     2, 2, 2,   3, 3,    4}
    fixed_width_column_wrapper<K> expect_keys({ 1,        2,         3,       4}, all_valid());
                                          //  { 0, 3,     1, 4, 5,   2, 8,    -}
    fixed_width_column_wrapper<R, int32_t> expect_vals({ 3,        5,         8,       0},
                                                       { 1,        1,         1,       0});

    auto agg = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg2 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}


struct groupby_max_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_max_string_test, basic)
{
    using K = int32_t;

    fixed_width_column_wrapper<K> keys        {     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
    strings_column_wrapper        vals        { "año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};

    fixed_width_column_wrapper<K> expect_keys {     1,     2,    3 };
    strings_column_wrapper        expect_vals({ "año", "zit", "₹1" });

    auto agg = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg2 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_max_string_test, zero_valid_values)
{
    using K = int32_t;

    fixed_width_column_wrapper<K> keys        { 1, 1, 1};
    strings_column_wrapper        vals      ( { "año", "bit", "₹1"}, all_null() );

    fixed_width_column_wrapper<K> expect_keys { 1 };
    strings_column_wrapper        expect_vals({ "" }, all_null());

    auto agg = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

    auto agg2 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}
// clang-format on

struct groupby_dictionary_max_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_dictionary_max_test, basic)
{
  using K = int32_t;
  using V = std::string;

  // clang-format off
  fixed_width_column_wrapper<K> keys{     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  dictionary_column_wrapper<V>  vals{ "año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};
  fixed_width_column_wrapper<K> expect_keys   {     1,     2,    3 };
  dictionary_column_wrapper<V>  expect_vals_w({ "año", "zit", "₹1" });
  // clang-format on

  auto expect_vals = cudf::dictionary::set_keys(expect_vals_w, vals.keys());

  test_single_agg(keys, vals, expect_keys, expect_vals->view(), cudf::make_max_aggregation());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_max_aggregation(),
                  force_use_sort_impl::YES);
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, GroupBySortMaxDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    auto const keys  = fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    auto const expect_keys     = fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_max = fp_wrapper{{6, 9, 8}, scale};

    auto agg3 = cudf::make_max_aggregation();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_max, std::move(agg3), force_use_sort_impl::YES);
  }
}

// This test will not work until the following ptxas bug is fixed in 10.2
// https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3186317&cp=
TYPED_TEST(FixedPointTestBothReps, DISABLED_GroupByHashMaxDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    auto const keys  = fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    auto const expect_keys     = fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_max = fp_wrapper{{6, 9, 8}, scale};

    auto agg7 = cudf::make_max_aggregation();
    test_single_agg(keys, vals, expect_keys, expect_vals_max, std::move(agg7));
  }
}

}  // namespace test
}  // namespace cudf
