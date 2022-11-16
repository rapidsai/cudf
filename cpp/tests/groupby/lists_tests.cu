/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "groupby_test_util.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <vector>

namespace cudf {
namespace test {

template <typename V>
struct groupby_lists_test : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(groupby_lists_test, cudf::test::FixedWidthTypes);

using namespace cudf::test::iterators;

using R = cudf::detail::target_type_t<int32_t, aggregation::SUM>;  // Type of aggregation result.
using strings = strings_column_wrapper;
using structs = structs_column_wrapper;

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using lcw = cudf::test::lists_column_wrapper<T, int32_t>;

namespace {
static constexpr auto null = -1;

// Checking with a single aggregation, and aggregation column.
// This test is orthogonal to the aggregation type; it focuses on testing the grouping
// with LISTS keys.
auto sum_agg() { return cudf::make_sum_aggregation<groupby_aggregation>(); }

// TODO: this is a naive way to compare expected key/value against resulting key/value. To be
// replaced once list lex comparator is supported (https://github.com/rapidsai/cudf/issues/5890)
template <typename Equal>
struct match_expected_fn {
  match_expected_fn(cudf::size_type const num_rows, Equal equal)
    : _num_rows{num_rows}, _equal{equal}
  {
  }

  __device__ bool operator()(cudf::size_type const idx)
  {
    for (auto i = _num_rows; i < 2 * _num_rows; i++) {
      if (_equal(idx, i)) { return true; }
    }
    return false;
  }

  cudf::size_type const _num_rows;
  Equal _equal;
};

inline void test_hash_based_sum_agg(column_view const& keys,
                                    column_view const& values,
                                    column_view const& expect_keys,
                                    column_view const& expect_vals)
{
  auto const include_null_keys = null_policy::INCLUDE;
  auto const keys_are_sorted   = sorted::NO;

  std::vector<groupby::aggregation_request> requests;
  auto& request  = requests.emplace_back(groupby::aggregation_request());
  request.values = values;
  request.aggregations.push_back(std::move(cudf::make_sum_aggregation<groupby_aggregation>()));

  groupby::groupby gb_obj(cudf::table_view({keys}), include_null_keys, keys_are_sorted);

  auto result = gb_obj.aggregate(requests);

  cudf::table_view result_kv{
    {result.first->get_column(0).view(), result.second[0].results[0]->view()}};
  cudf::table_view expected_kv{{expect_keys, expect_vals}};

  auto const num_rows = result_kv.num_rows();
  EXPECT_EQ(num_rows, expected_kv.num_rows());

  // Concatenate expected table and resulting table into one unique table `t`:
  // expected table:  `t [       0,     num_rows - 1]`
  // resulting table: `t [num_rows, 2 * num_rows - 1]`
  auto combined_table = cudf::concatenate(std::vector{expected_kv, result_kv});
  auto preprocessed_t = cudf::experimental::row::hash::preprocessed_table::create(
    combined_table->view(), cudf::get_default_stream());
  cudf::experimental::row::equality::self_comparator comparator(preprocessed_t);

  auto const null_keys_are_equal =
    include_null_keys == null_policy::INCLUDE ? null_equality::EQUAL : null_equality::UNEQUAL;
  auto row_equal = comparator.equal_to(nullate::DYNAMIC{true}, null_keys_are_equal);
  auto func      = match_expected_fn{num_rows, row_equal};

  // For each row in expected table `t[0, num_rows)`, there must be a match
  // in the resulting table `t[num_rows, 2 * num_rows)`
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy(cudf::get_default_stream()),
                             thrust::make_counting_iterator<cudf::size_type>(0),
                             thrust::make_counting_iterator<cudf::size_type>(num_rows),
                             func));
}

void test_sort_based_sum_agg(column_view const& keys,
                             column_view const& values,
                             column_view const& expect_keys,
                             column_view const& expect_vals)
{
  test_single_agg(keys,
                  values,
                  expect_keys,
                  expect_vals,
                  sum_agg(),
                  force_use_sort_impl::YES,
                  null_policy::INCLUDE);
}

void test_sum_agg(column_view const& keys,
                  column_view const& values,
                  column_view const& expected_keys,
                  column_view const& expected_values)
{
  test_sort_based_sum_agg(keys, values, expected_keys, expected_values);
  test_hash_based_sum_agg(keys, values, expected_keys, expected_values);
}
}  // namespace

TYPED_TEST(groupby_lists_test, basic)
{
  if (std::is_same_v<TypeParam, bool>) { return; }

  // clang-format off
  auto keys   = lcw<TypeParam> { {1,1}, {2,2}, {3,3}, {1,1}, {2,2} };
  auto values = fwcw<int32_t>  {    0,     1,     2,     3,     4  };

  auto expected_keys   = lcw<TypeParam> { {1,1}, {2,2}, {3,3} };
  auto expected_values = fwcw<R>        {    3,     5,     2  };
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_lists_test, all_null_input)
{
  // clang-format off
  auto keys   = lcw<TypeParam> { {{1,1}, {2,2}, {3,3}, {1,1}, {2,2}}, all_nulls()};
  auto values = fwcw<int32_t>  {     0,     1,     2,     3,     4 };

  auto expected_keys   = lcw<TypeParam> { {{null,null}}, all_nulls()};
  auto expected_values = fwcw<R>        {          10 };
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_lists_test, lists_with_nulls)
{
  // clang-format off
  auto keys   = lcw<TypeParam> { {{1,1}, {2,2}, {3,3}, {1,1}, {2,2}}, nulls_at({1,2,4})};
  auto values = fwcw<int32_t>  {     0,     1,     2,     3,     4 };

  auto expected_keys   = lcw<TypeParam> { {{null,null}, {1,1}}, null_at(0)};
  auto expected_values = fwcw<R>        {           7,     3 };
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_lists_test, lists_with_null_elements)
{
  auto keys =
    lcw<TypeParam>{{lcw<TypeParam>{{{1, 2, 3}, {}, {4, 5}, {}, {6, 0}}, nulls_at({1, 3})},
                    lcw<TypeParam>{{{1, 2, 3}, {}, {4, 5}, {}, {6, 0}}, nulls_at({1, 3})},
                    lcw<TypeParam>{{{1, 2, 3}, {}, {4, 5}, {}, {6, 0}}, nulls_at({1, 3})},
                    lcw<TypeParam>{{{1, 2, 3}, {}, {4, 5}, {}, {6, 0}}, nulls_at({1, 3})}},
                   nulls_at({2, 3})};
  auto values = fwcw<int32_t>{1, 2, 4, 5};

  auto expected_keys = lcw<TypeParam>{
    {{}, lcw<TypeParam>{{{1, 2, 3}, {}, {4, 5}, {}, {6, 0}}, nulls_at({1, 3})}}, null_at(0)};
  auto expected_values = fwcw<R>{9, 3};

  test_sum_agg(keys, values, expected_keys, expected_values);
}
}  // namespace test
}  // namespace cudf
