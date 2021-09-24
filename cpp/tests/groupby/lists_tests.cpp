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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {

template <typename V>
struct groupby_lists_test : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(groupby_lists_test, cudf::test::FixedWidthTypes);

namespace {
// Checking with a single aggregation, and aggregation column.
// This test is orthogonal to the aggregation type; it focuses on testing the grouping
// with LISTS keys.
auto sum_agg() { return cudf::make_sum_aggregation<groupby_aggregation>(); }

void test_sort_based_sum_agg(column_view const& keys, column_view const& values)
{
  test_single_agg(
    keys, values, keys, values, sum_agg(), force_use_sort_impl::YES, null_policy::INCLUDE);
}

void test_hash_based_sum_agg(column_view const& keys, column_view const& values)
{
  test_single_agg(
    keys, values, keys, values, sum_agg(), force_use_sort_impl::NO, null_policy::INCLUDE);
}

}  // namespace

TYPED_TEST(groupby_lists_test, top_level_lists_are_unsupported)
{
  // Test that grouping on LISTS columns fails visibly.

  // clang-format off
  auto keys   = lists_column_wrapper<TypeParam, int32_t> { {1,1},  {2,2},  {3,3},   {1,1},   {2,2} };
  auto values = fixed_width_column_wrapper<int32_t>      {     0,      1,      2,      3,       4  };
  // clang-format on

  EXPECT_THROW(test_sort_based_sum_agg(keys, values), cudf::logic_error);
  EXPECT_THROW(test_hash_based_sum_agg(keys, values), cudf::logic_error);
}

}  // namespace test
}  // namespace cudf
