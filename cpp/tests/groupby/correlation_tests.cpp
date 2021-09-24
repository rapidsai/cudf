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

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <tests/groupby/groupby_test_util.hpp>

#include <limits>
#include <vector>

using namespace cudf::test::iterators;
namespace cudf {
namespace test {

using structs = structs_column_wrapper;

template <typename V>
struct groupby_correlation_test : public cudf::test::BaseFixture {
};

using supported_types = RemoveIf<ContainedIn<Types<bool>>, cudf::test::NumericTypes>;

TYPED_TEST_CASE(groupby_correlation_test, supported_types);
using K = int32_t;

TYPED_TEST(groupby_correlation_test, basic)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::CORRELATION>;

  auto keys     = fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = fixed_width_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = fixed_width_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = structs{{member_0, member_1}};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  fixed_width_column_wrapper<R, double> expect_vals{
    {1.0, 0.6, std::numeric_limits<double>::quiet_NaN()}};

  auto agg =
    cudf::make_correlation_aggregation<groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_correlation_test, empty_cols)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::CORRELATION>;

  fixed_width_column_wrapper<K> keys{};
  fixed_width_column_wrapper<V> member_0{}, member_1{};
  auto vals = structs{{member_0, member_1}};

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R> expect_vals{};

  auto agg =
    cudf::make_correlation_aggregation<groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_correlation_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::CORRELATION>;

  fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  fixed_width_column_wrapper<V> member_0{3, 4, 5}, member_1{6, 7, 8};
  auto vals = structs{{member_0, member_1}};

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R> expect_vals{};

  auto agg =
    cudf::make_correlation_aggregation<groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_correlation_test, zero_valid_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::CORRELATION>;

  fixed_width_column_wrapper<K> keys{1, 1, 1};
  fixed_width_column_wrapper<V> member_0({3, 4, 5}, all_nulls());
  fixed_width_column_wrapper<V> member_1({3, 4, 5}, all_nulls());
  auto vals = structs{{member_0, member_1}};

  fixed_width_column_wrapper<K> expect_keys{1};
  fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg =
    cudf::make_correlation_aggregation<groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_correlation_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::CORRELATION>;

  fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V> val0({9, 1, 1, 2, 2, 3, 3, -1, 1, 4, 4},
                                     {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<V> val1({1, 1, 1, 2, 0, 3, 3, -1, 0, 2, 2});
  auto vals = structs{{val0, val1}};

  //                                        { 1, 1,     2, 2, 2,   3, 3,       4}
  fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //                                        { 3, 6,     1, 4, 9,   2, 8,       3}
  fixed_width_column_wrapper<R> expect_vals(
    {1.0, 0.6, std::numeric_limits<double>::quiet_NaN(), 0.}, {1, 1, 1, 0});

  auto agg =
    cudf::make_correlation_aggregation<groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

struct groupby_dictionary_correlation_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_dictionary_correlation_test, basic)
{
  using V = int16_t;
  using R = cudf::detail::target_type_t<V, aggregation::CORRELATION>;

  auto keys     = fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = dictionary_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = dictionary_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = structs{{member_0, member_1}};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  fixed_width_column_wrapper<R, double> expect_vals{
    {1.0, 0.6, std::numeric_limits<double>::quiet_NaN()}};

  auto agg =
    cudf::make_correlation_aggregation<groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

}  // namespace test
}  // namespace cudf
