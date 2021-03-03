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
struct groupby_min_scan_test : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(groupby_min_scan_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(groupby_min_scan_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::MIN>;

  // clang-format off
    fixed_width_column_wrapper<K> keys          {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<V, int32_t> vals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4});

    fixed_width_column_wrapper<K> expect_keys          {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    fixed_width_column_wrapper<R, int32_t> expect_vals({5, 5, 1, 6, 6, 0, 0, 7, 2, 2});
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_min_scan_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::MIN>;

  // clang-format off
    fixed_width_column_wrapper<K> keys        { };
    fixed_width_column_wrapper<V> vals        { };

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_min_scan_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::MIN>;

  // clang-format off
    fixed_width_column_wrapper<K> keys(         {1, 2, 3}, all_null() );
    fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5});

    fixed_width_column_wrapper<K> expect_keys { };
    fixed_width_column_wrapper<R> expect_vals { };
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_min_scan_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::MIN>;

  // clang-format off
    fixed_width_column_wrapper<K> keys          {1, 1, 1};
    fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5}, all_null());

    fixed_width_column_wrapper<K> expect_keys          {  1,  1,  1};
    fixed_width_column_wrapper<R, int32_t> expect_vals({ -1, -1, -1}, all_null());
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_min_scan_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::MIN>;

  // clang-format off
    fixed_width_column_wrapper<K> keys(         {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<V, int32_t> vals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 4},
                                                {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                          //  { 1, 1, 1, 2, 2, 2, 2, 3, _, 3, 4}
    fixed_width_column_wrapper<K> expect_keys({ 1, 1, 1, 2, 2, 2, 2, 3,/**/3, 4}, all_valid());
                                          //  { _, 8, 1, 6, 9, _, 4, 7, 2, 3, _}
    fixed_width_column_wrapper<R, int32_t> expect_vals({-1, 8, 1, 6, 6,-1, 4, 7,/*2,*/3,-1},
                                                       { 0, 1, 1, 1, 1, 0, 1, 1,/*1,*/1, 0});
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

struct groupby_min_scan_string_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_min_scan_string_test, basic)
{
  using K = int32_t;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  strings_column_wrapper vals{"año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};

  fixed_width_column_wrapper<K> expect_keys{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  strings_column_wrapper expect_vals;

  auto agg = cudf::make_min_aggregation();
  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg)),
                            "Unsupported groupby scan type-agg combination");
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, GroupBySortMinScanDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  using K = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};

    // clang-format off
    auto const keys  = fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                  {5, 6, 7, 8, 9, 0, 1, 2, 3, 4}, scale};

                                                           // {5, 8, 1, 6, 9, 0, 4, 7, 2, 3}
    auto const expect_keys     = fixed_width_column_wrapper<K>{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    auto const expect_vals_min = fp_wrapper{                  {5, 5, 1, 6, 6, 0, 0, 7, 2, 2}, scale};
    // clang-format on

    auto agg2 = cudf::make_min_aggregation();
    test_single_scan(keys, vals, expect_keys, expect_vals_min, std::move(agg2));
  }
}

}  // namespace test
}  // namespace cudf
