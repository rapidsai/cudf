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
struct groupby_count_scan_test : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(groupby_count_scan_test, cudf::test::AllTypes);

// clang-format off
TYPED_TEST(groupby_count_scan_test, basic)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

    fixed_width_column_wrapper<K> keys      { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V, int> vals { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    fixed_width_column_wrapper<K> expect_keys      {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    fixed_width_column_wrapper<R, int> expect_vals {0, 1, 2, 0, 1, 2, 3, 0, 1, 2};

    auto agg1 = cudf::make_count_aggregation();
    CUDF_EXPECT_THROW_MESSAGE(
      test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1)),
      "Unsupported groupby scan aggregation");

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, empty_cols)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

    fixed_width_column_wrapper<K> keys        { };
    fixed_width_column_wrapper<V> vals;

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals;

    auto agg1 = cudf::make_count_aggregation();
    EXPECT_NO_THROW(test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1)));

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, zero_valid_keys)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

    fixed_width_column_wrapper<K> keys(      { 1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V, int> vals  { 3, 4, 5};

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R, int> expect_vals { };

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, zero_valid_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

    fixed_width_column_wrapper<K> keys        { 1, 1, 1};
    fixed_width_column_wrapper<V, int> vals ( { 3, 4, 5}, all_null() );

    fixed_width_column_wrapper<K> expect_keys { 1, 1, 1};
    fixed_width_column_wrapper<R, int> expect_vals { 0, 1, 2};

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, null_keys_and_values)
{
    using K = int32_t;
    using V = TypeParam;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

    fixed_width_column_wrapper<K> keys(     { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                            { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V, int> vals({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                            { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                          //  {1, 1, 1, 2, 2, 2, 2, 3, _, 3, 4}
    fixed_width_column_wrapper<K> expect_keys({1, 1, 1, 2, 2, 2, 2, 3,    3, 4}, all_valid());
                                          //  {0, 3, 6, 1, 4, _, 9, 2, 7, 8, -}
    fixed_width_column_wrapper<R, int> expect_vals {0, 1, 2, 0, 1, 2, 3, 0, 1, 0};

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

struct groupby_count_scan_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_count_scan_string_test, basic)
{
    using K = int32_t;
    using V = cudf::string_view;
    using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

    fixed_width_column_wrapper<K> keys        {   1,   3,   3,   5,   5,   0};
    strings_column_wrapper        vals        { "1", "1", "1", "1", "1", "1"};

    fixed_width_column_wrapper<K> expect_keys     {0, 1, 3, 3, 5, 5};
    fixed_width_column_wrapper<R, int> expect_vals{0, 0, 0, 1, 0, 1};

    auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
    test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}
// clang-format on

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, GroupByCountScan)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  using K = int32_t;
  using V = decimalXX;
  using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

  auto const scale = scale_type{-1};
  auto const keys  = fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals  = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

  auto const expect_keys = fixed_width_column_wrapper<K>{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  auto const expect_vals = fixed_width_column_wrapper<R, int>{0, 1, 2, 0, 1, 2, 3, 0, 1, 2};

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(keys, vals, expect_keys, expect_vals, cudf::make_count_aggregation()),
    "Unsupported groupby scan aggregation");

  auto agg2 = cudf::make_count_aggregation(null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

struct groupby_dictionary_count_scan_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_dictionary_count_scan_test, basic)
{
  using K = int32_t;
  using V = std::string;
  using R = cudf::detail::target_type_t<V, aggregation::COUNT_ALL>;

  // clang-format off
  strings_column_wrapper       keys{"1", "3", "3", "5", "5", "0"};
  dictionary_column_wrapper<K> vals{ 1,   1,   1,   1,   1,   1};
  strings_column_wrapper             expect_keys{"0", "1", "3", "3", "5", "5"};
  fixed_width_column_wrapper<R, int> expect_vals{ 0,   0,   0,   1,   0,   1};
  // clang-format on

  auto agg1 = cudf::make_count_aggregation();
  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1)),
                            "Unsupported groupby scan aggregation");
  test_single_scan(
    keys, vals, expect_keys, expect_vals, cudf::make_count_aggregation(null_policy::INCLUDE));
}

}  // namespace test
}  // namespace cudf
