/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

template <typename V>
struct groupby_argmax_test : public cudf::test::BaseFixture {};
using K = int32_t;

TYPED_TEST_SUITE(groupby_argmax_test, cudf::test::FixedWidthTypes);

TYPED_TEST(groupby_argmax_test, basic)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{0, 1, 2};

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, zero_valid_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, cudf::test::iterators::all_nulls());

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmax_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                 {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 4},
                                                 {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  //  {1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());
  //  {6, 3,     5, 4, 0,   2, 1,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 4, 7, 0}, {1, 1, 1, 0});

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_argmax_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_argmax_string_test, basic)
{
  using R = cudf::detail::target_type_t<cudf::string_view, cudf::aggregation::ARGMAX>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0, 4, 2});

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_argmax_string_test, zero_valid_values)
{
  using R = cudf::detail::target_type_t<cudf::string_view, cudf::aggregation::ARGMAX>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::strings_column_wrapper vals({"año", "bit", "₹1"}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, cudf::test::iterators::all_nulls());

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_dictionary_argmax_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_argmax_test, basic)
{
  using V = std::string;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  cudf::test::dictionary_column_wrapper<V>  vals{ "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1, 2, 3 });
  cudf::test::fixed_width_column_wrapper<R> expect_vals({ 0, 4, 2 });
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_argmax_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_argmax_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

struct groupby_argmax_struct_test : public cudf::test::BaseFixture {};

TEST_F(groupby_argmax_struct_test, basic)
{
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const expect_keys    = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_indices = cudf::test::fixed_width_column_wrapper<int32_t>{0, 4, 2};

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

TEST_F(groupby_argmax_struct_test, slice_input)
{
  constexpr int32_t dont_care{1};
  auto const keys_original = cudf::test::fixed_width_column_wrapper<int32_t>{
    dont_care, dont_care, 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, dont_care};
  auto const vals_original = [] {
    auto child1 = cudf::test::strings_column_wrapper{"dont_care",
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
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const keys           = cudf::slice(keys_original, {2, 12})[0];
  auto const vals           = cudf::slice(vals_original, {2, 12})[0];
  auto const expect_keys    = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_indices = cudf::test::fixed_width_column_wrapper<int32_t>{0, 4, 2};

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

TEST_F(groupby_argmax_struct_test, null_keys_and_values)
{
  constexpr int32_t null{0};
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{
    {1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4}, cudf::test::iterators::null_at(7)};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "" /*NULL*/, "" /*NULL*/, "$1", "€1", "wut", "" /*NULL*/};
    auto child2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{9, 8, 7, 6, 5, null, null, 2, 1, 0, null};
    return cudf::test::structs_column_wrapper{{child1, child2},
                                              cudf::test::iterators::nulls_at({5, 6, 10})};
  }();

  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{
    {1, 2, 3, 4}, cudf::test::iterators::no_nulls()};
  auto const expect_indices = cudf::test::fixed_width_column_wrapper<int32_t>{
    {0, 4, 2, null}, cudf::test::iterators::null_at(3)};

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

CUDF_TEST_PROGRAM_MAIN()
