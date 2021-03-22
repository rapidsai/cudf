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
struct groupby_min_scan_test : public cudf::test::BaseFixture {
  using V              = T;
  using R              = cudf::detail::target_type_t<V, aggregation::MIN>;
  using value_wrapper  = fixed_width_column_wrapper<V, int32_t>;
  using result_wrapper = fixed_width_column_wrapper<R, int32_t>;
};

TYPED_TEST_CASE(groupby_min_scan_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(groupby_min_scan_test, basic)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys   {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  value_wrapper vals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4});

  key_wrapper expect_keys    {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  result_wrapper expect_vals({5, 5, 1, 6, 6, 0, 0, 7, 2, 2});
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_min_scan_test, empty_cols)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys{};
  value_wrapper vals{};

  key_wrapper expect_keys{};
  result_wrapper expect_vals{};

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_min_scan_test, zero_valid_keys)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys({1, 2, 3}, all_null());
  value_wrapper vals({3, 4, 5});

  key_wrapper expect_keys{};
  result_wrapper expect_vals{};
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_min_scan_test, zero_valid_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys   {1, 1, 1};
  value_wrapper vals({3, 4, 5}, all_null());

  key_wrapper expect_keys    {1, 1, 1};
  result_wrapper expect_vals({-1, -1, -1}, all_null());
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_min_scan_test, null_keys_and_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys(  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4}, {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  value_wrapper vals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 4}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                         //  { 1, 1, 1, 2, 2,  2, 2, 3, _, 3, 4}
  key_wrapper expect_keys(   { 1, 1, 1, 2, 2,  2, 2, 3,    3, 4}, all_valid());
                         //  { _, 8, 1, 6, 9,  _, 4, 7, 2, 3, _}
  result_wrapper expect_vals({-1, 8, 1, 6, 6, -1, 4, 7,    3, -1},
                             { 0, 1, 1, 1, 1,  0, 1, 1,    1, 0});
  // clang-format on

  auto agg = cudf::make_min_aggregation();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

struct groupby_min_scan_string_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_min_scan_string_test, basic)
{
  key_wrapper keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  strings_column_wrapper vals{"año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};

  key_wrapper expect_keys{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
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
  using fp_wrapper = fixed_point_column_wrapper<RepType>;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};

    // clang-format off
    auto const keys = key_wrapper{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals = fp_wrapper{{5, 6, 7, 8, 9, 0, 1, 2, 3, 4}, scale};

    // {5, 8, 1, 6, 9, 0, 4, 7, 2, 3}
    auto const expect_keys     = key_wrapper{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    auto const expect_vals_min = fp_wrapper{{5, 5, 1, 6, 6, 0, 0, 7, 2, 2}, scale};
    // clang-format on

    auto agg = cudf::make_min_aggregation();
    test_single_scan(keys, vals, expect_keys, expect_vals_min, std::move(agg));
  }
}

}  // namespace test
}  // namespace cudf
