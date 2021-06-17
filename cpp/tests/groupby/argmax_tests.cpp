/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

using namespace cudf::test::iterators;

namespace cudf {
namespace test {
template <typename V>
struct groupby_argmax_test : public cudf::test::BaseFixture {
};
using K = int32_t;

TYPED_TEST_CASE(groupby_argmax_test, cudf::test::FixedWidthTypes);

TYPED_TEST(groupby_argmax_test, basic)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same<V, bool>::value) return;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V> vals{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  fixed_width_column_wrapper<R> expect_vals{0, 1, 2};

  auto agg = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same<V, bool>::value) return;

  fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  fixed_width_column_wrapper<V> vals({3, 4, 5});

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, zero_valid_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same<V, bool>::value) return;

  fixed_width_column_wrapper<K> keys{1, 1, 1};
  fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  fixed_width_column_wrapper<K> expect_keys{1};
  fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same<V, bool>::value) return;

  fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<V> vals({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 4},
                                     {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  //  {1, 1,     2, 2, 2,   3, 3,    4}
  fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //  {6, 3,     5, 4, 0,   2, 1,    -}
  fixed_width_column_wrapper<R> expect_vals({3, 4, 7, 0}, {1, 1, 1, 0});

  auto agg = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_argmax_string_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_argmax_string_test, basic)
{
  using V = string_view;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  strings_column_wrapper vals{"año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  fixed_width_column_wrapper<R> expect_vals({0, 4, 2});

  auto agg = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_argmax_string_test, zero_valid_values)
{
  using V = string_view;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  fixed_width_column_wrapper<K> keys{1, 1, 1};
  strings_column_wrapper vals({"año", "bit", "₹1"}, all_nulls());

  fixed_width_column_wrapper<K> expect_keys{1};
  fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_dictionary_argmax_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_dictionary_argmax_test, basic)
{
  using V = std::string;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  // clang-format off
  fixed_width_column_wrapper<K> keys{     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  dictionary_column_wrapper<V>  vals{ "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
  fixed_width_column_wrapper<K> expect_keys({ 1, 2, 3 });
  fixed_width_column_wrapper<R> expect_vals({ 0, 4, 2 });
  // clang-format on

  test_single_agg(keys, vals, expect_keys, expect_vals, cudf::make_argmax_aggregation());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_argmax_aggregation(),
                  force_use_sort_impl::YES);
}

}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
