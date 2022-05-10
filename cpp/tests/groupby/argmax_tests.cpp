/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

TYPED_TEST_SUITE(groupby_argmax_test, cudf::test::FixedWidthTypes);

TYPED_TEST(groupby_argmax_test, basic)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V> vals{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  fixed_width_column_wrapper<R> expect_vals{0, 1, 2};

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  fixed_width_column_wrapper<V> vals({3, 4, 5});

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, zero_valid_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  fixed_width_column_wrapper<K> keys{1, 1, 1};
  fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  fixed_width_column_wrapper<K> expect_keys{1};
  fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<V> vals({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 4},
                                     {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  //  {1, 1,     2, 2, 2,   3, 3,    4}
  fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //  {6, 3,     5, 4, 0,   2, 1,    -}
  fixed_width_column_wrapper<R> expect_vals({3, 4, 7, 0}, {1, 1, 1, 0});

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

template <typename T>
using FWCW = cudf::test::fixed_width_column_wrapper<T>;

TYPED_TEST(groupby_argmax_test, structs)
{
  using V = TypeParam;

  using R       = cudf::detail::target_type_t<int, aggregation::ARGMAX>;
  using STRINGS = cudf::test::strings_column_wrapper;
  using STRUCTS = cudf::test::structs_column_wrapper;

  if (std::is_same_v<V, bool>) return;

  /*
    `@` indicates null
       keys:               values:
       /+---------------+
       |s1{s2{a,b},   c}|
       +----------------+
     0 |  { {1, 1}, "a"}|  1
     1 |  { {1, 2}, "b"}|  2
     2 |  {@{2, 1}, "c"}|  3
     3 |  {@{2, 1}, "c"}|  4
     4 | @{ {2, 2}, "d"}|  5
     5 | @{ {2, 2}, "d"}|  6
     6 |  { {1, 1}, "a"}|  7
     7 |  {@{2, 1}, "c"}|  8
     8 |  { {1, 1}, "a"}|  9
       +----------------+
  */

  // clang-format off
  auto col_a = FWCW<V>{ 1,   1,   2,   2,   2,   2,   1,   2,   1};
  auto col_b = FWCW<V>{ 1,   2,   1,   1,   2,   2,   1,   1,   1};
  auto col_c = STRINGS{"a", "b", "c", "c", "d", "d", "a", "c", "a"};
  // clang-format on
  auto s2 = STRUCTS{{col_a, col_b}, nulls_at({2, 3, 7})};

  auto keys = STRUCTS{{s2, col_c}, nulls_at({4, 5})};
  auto vals = FWCW<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};

  // clang-format off
  auto expected_col_a = FWCW<V>{ 1,   1,   2,   2,   2};
  auto expected_col_b = FWCW<V>{ 1,   2,   1,   1,   1};
  auto expected_col_c = STRINGS{"a", "b", "c", "c", "c"};
  // clang-format on
  auto expected_s2 = STRUCTS{{expected_col_a, expected_col_b}, nulls_at({2, 3, 4})};

  auto expect_keys = STRUCTS{{expected_s2, expected_col_c}, no_nulls()};
  auto expect_vals = FWCW<R>{8, 1, 2, 3, 7};

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
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

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<groupby_aggregation>();
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

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<groupby_aggregation>();
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

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_argmax_aggregation<groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_argmax_aggregation<groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

struct groupby_argmax_struct_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_argmax_struct_test, basic)
{
  auto const keys = fixed_width_column_wrapper<int32_t>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = [] {
    auto child1 =
      strings_column_wrapper{"año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return structs_column_wrapper{{child1, child2}};
  }();

  auto const expect_keys    = fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_indices = fixed_width_column_wrapper<int32_t>{0, 4, 2};

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

TEST_F(groupby_argmax_struct_test, slice_input)
{
  constexpr int32_t dont_care{1};
  auto const keys_original = fixed_width_column_wrapper<int32_t>{
    dont_care, dont_care, 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, dont_care};
  auto const vals_original = [] {
    auto child1 = strings_column_wrapper{"dont_care",
                                         "dont_care",
                                         "año",
                                         "bit",
                                         "₹1",
                                         "aaa",
                                         "zit",
                                         "bat",
                                         "aab",
                                         "$1",
                                         "€1",
                                         "wut",
                                         "dont_care"};
    auto child2 = fixed_width_column_wrapper<int32_t>{
      dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return structs_column_wrapper{{child1, child2}};
  }();

  auto const keys           = cudf::slice(keys_original, {2, 12})[0];
  auto const vals           = cudf::slice(vals_original, {2, 12})[0];
  auto const expect_keys    = fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_indices = fixed_width_column_wrapper<int32_t>{0, 4, 2};

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

TEST_F(groupby_argmax_struct_test, null_keys_and_values)
{
  constexpr int32_t null{0};
  auto const keys =
    fixed_width_column_wrapper<int32_t>{{1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4}, null_at(7)};
  auto const vals = [] {
    auto child1 = strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "" /*NULL*/, "" /*NULL*/, "$1", "€1", "wut", "" /*NULL*/};
    auto child2 = fixed_width_column_wrapper<int32_t>{9, 8, 7, 6, 5, null, null, 2, 1, 0, null};
    return structs_column_wrapper{{child1, child2}, nulls_at({5, 6, 10})};
  }();

  auto const expect_keys    = fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4}, no_nulls()};
  auto const expect_indices = fixed_width_column_wrapper<int32_t>{{0, 4, 2, null}, null_at(3)};

  auto agg = cudf::make_argmax_aggregation<groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
