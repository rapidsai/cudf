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
using K           = int32_t;
using key_wrapper = fixed_width_column_wrapper<K>;

template <typename T>
struct groupby_sum_scan_test : public cudf::test::BaseFixture {
  using V              = T;
  using R              = cudf::detail::target_type_t<V, aggregation::SUM>;
  using value_wrapper  = fixed_width_column_wrapper<V, int32_t>;
  using result_wrapper = fixed_width_column_wrapper<R, int32_t>;
};

using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_CASE(groupby_sum_scan_test, supported_types);

TYPED_TEST(groupby_sum_scan_test, basic)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  value_wrapper vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  key_wrapper expect_keys   {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  //                        {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  result_wrapper expect_vals{0, 3, 9, 1, 5, 10, 19, 2, 9, 17};
  // clang-format on
  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, empty_cols)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys{};
  value_wrapper vals{};

  key_wrapper expect_keys{};
  result_wrapper expect_vals{};
  // clang-format on

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, zero_valid_keys)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys({1, 2, 3}, all_null());
  value_wrapper vals{3, 4, 5};

  key_wrapper expect_keys{};
  result_wrapper expect_vals{};
  // clang-format on

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, zero_valid_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys   {1, 1, 1};
  value_wrapper vals({3, 4, 5}, all_null());

  key_wrapper expect_keys    {1, 1, 1};
  result_wrapper expect_vals({3, 4, 5}, all_null());
  // clang-format on

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, null_keys_and_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys(  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4}, {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  value_wrapper vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //                         { 1, 1, 1, 2, 2,  2,  2, 3, *, 3, 4};
  key_wrapper expect_keys(   { 1, 1, 1, 2, 2,  2,  2, 3,    3, 4}, all_valid());
                          // { -, 3, 6, 1, 4,  -,  9, 2, _, 8, -}
  result_wrapper expect_vals({-1, 3, 9, 1, 5, -1, 14, 2,   10, -1},
                             { 0, 1, 1, 1, 1,  0,  1, 1,    1, 0});
  // clang-format on

  auto agg = cudf::make_sum_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, GroupBySortSumScanDecimalAsValue)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepType        = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper     = fixed_point_column_wrapper<RepType>;
  using out_fp_wrapper = fixed_point_column_wrapper<int64_t>;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys = key_wrapper{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    auto const expect_keys     = key_wrapper    {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    auto const expect_vals_sum = out_fp_wrapper{{0, 3, 9, 1, 5, 10, 19, 2, 9, 17}, scale};
    // clang-format on

    auto agg2 = cudf::make_sum_aggregation();
    test_single_scan(keys, vals, expect_keys, expect_vals_sum, std::move(agg2));
  }
}

}  // namespace test
}  // namespace cudf
