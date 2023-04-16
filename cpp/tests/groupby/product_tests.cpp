/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

template <typename V>
struct groupby_product_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(groupby_product_test, supported_types);

TYPED_TEST(groupby_product_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                                    //  { 1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys { 1,        2,           3      };
                                                    //  { 0, 3, 6,  1, 4, 5, 9,  2, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({   0.,       180.,      112. }, no_nulls());
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

TYPED_TEST(groupby_product_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

TYPED_TEST(groupby_product_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  cudf::test::fixed_width_column_wrapper<K> keys({0, 0, 0}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

TYPED_TEST(groupby_product_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({0, 0, 0}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

TYPED_TEST(groupby_product_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                            { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals(       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3},
                                            { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                        //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,        2,         3,       4}, no_nulls());
                                        //  { _, 3, 6,  1, 4, 9,   2, 8,    _}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({ 18.,      36.,       16.,     3.},
                                            { 1,        1,         1,       0});
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

TYPED_TEST(groupby_product_test, dictionary)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{ 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V>  vals{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                                    //  { 1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3      });
                                                    //  { 0, 3, 6,  1, 4, 5, 9,  2, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({  0.,     180.,        112. }, no_nulls());
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

TYPED_TEST(groupby_product_test, dictionary_with_nulls)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{ 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V>  vals{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                     {1, 0, 0, 1, 1, 1, 1, 1, 1, 1}};

                                                    //  { 1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3      });
                                                    //  { 0, 3, 6,  @, 4, 5, 9,  @, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({  0.,     180.,        56. }, no_nulls());
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_product_aggregation<cudf::groupby_aggregation>());
}
