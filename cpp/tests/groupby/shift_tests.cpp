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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace test {

using K = int32_t;
template <typename T>
struct groupby_shift_fixed_width_test : public BaseFixture {
};

TYPED_TEST_SUITE(groupby_shift_fixed_width_test, FixedWidthTypes);

template <typename V>
void test_groupby_shift_fixed_width_single(fixed_width_column_wrapper<K> const& key,
                                           fixed_width_column_wrapper<V> const& value,
                                           size_type offset,
                                           scalar const& fill_value,
                                           fixed_width_column_wrapper<V> const& expected)
{
  groupby::groupby gb_obj(table_view({key}));
  std::vector<size_type> offsets{offset};
  auto got = gb_obj.shift(table_view{{value}}, offsets, {fill_value});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL((*got.second).view().column(0), expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithoutNull_NullScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<V> expected({-1, -1, 3, 5, -1, -1, 4}, {0, 0, 1, 1, 0, 0, 1});
  size_type offset = 2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width_single<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithNull_NullScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 1, 1, 1, 1});
  fixed_width_column_wrapper<V> expected({-1, -1, -1, -1, -1, -1, -1}, {0, 0, 0, 0, 0, 0, 0});
  size_type offset = 2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width_single<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithoutNull_ValidScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5});
  fixed_width_column_wrapper<V> expected({42, 42, 42, 3, 5, 8, 9, 42, 42, 42, 4, 6, 7});
  size_type offset = 3;
  auto slr         = cudf::scalar_type_t<TypeParam>(make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithNull_ValidScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},
                                    {1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
  fixed_width_column_wrapper<V> expected({42, 42, 42, 3, 5, -1, -1, 42, 42, 42, -1, -1, 7},
                                         {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1});
  size_type offset = 3;
  auto slr         = cudf::scalar_type_t<TypeParam>(make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithoutNull_NullScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<V> expected({5, 8, 9, -1, 6, 7, -1}, {1, 1, 1, 0, 1, 1, 0});
  size_type offset = -1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width_single<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithNull_NullScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 1, 1, 1, 1});
  fixed_width_column_wrapper<V> expected({-1, 8, 9, -1, 6, 7, -1}, {0, 1, 1, 0, 1, 1, 0});
  size_type offset = -1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width_single<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithoutNull_ValidScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  fixed_width_column_wrapper<V> expected({3, 5, 42, 42, 42, 42, 42, 4, 42, 42, 42, 42, 42});
  size_type offset = -5;
  auto slr         = cudf::scalar_type_t<TypeParam>(make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithNull_ValidScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},
                                    {1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
  fixed_width_column_wrapper<V> expected({5, -1, -1, -1, 3, 5, 42, -1, 7, 0, 2, -1, 42},
                                         {1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1});
  size_type offset = -1;
  auto slr         = cudf::scalar_type_t<TypeParam>(make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ZeroShiftNullScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<V> expected({3, 5, 8, 9, 4, 6, 7});
  size_type offset = 0;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width_single<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ZeroShiftValidScalar)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  fixed_width_column_wrapper<V> expected({3, 5, 8, 9, 1, 3, 5, 4, 6, 7, 0, 2, 4});
  size_type offset = 0;
  auto slr         = cudf::scalar_type_t<TypeParam>(make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, VeryLargeForwardOffset)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  fixed_width_column_wrapper<V> expected({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  size_type offset = 1024;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width_single<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, VeryLargeBackwardOffset)
{
  using V = TypeParam;

  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  fixed_width_column_wrapper<V> expected({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  size_type offset = -1024;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width_single<V>(key, val, offset, *slr, expected);
}

struct groupby_shift_string_test : public BaseFixture {
};

void test_groupby_shift_string_single(fixed_width_column_wrapper<K> const& key,
                                      strings_column_wrapper const& value,
                                      size_type offset,
                                      scalar const& fill_value,
                                      strings_column_wrapper const& expected)
{
  groupby::groupby gb_obj(table_view({key}));
  std::vector<size_type> offsets{offset};
  auto got = gb_obj.shift(table_view{{value}}, offsets, {fill_value});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL((*got.second).view().column(0), expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithoutNull_NullScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"", "a", "cc", "f", "", "bb", "d"}, {0, 1, 1, 1, 0, 1, 1});
  size_type offset = 1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithNull_NullScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"}, {1, 0, 1, 1, 0, 0, 0});
  strings_column_wrapper expected({"", "", "a", "cc", "", "", ""}, {0, 0, 1, 1, 0, 0, 0});
  size_type offset = 2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithoutNull_ValidScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"42", "42", "a", "cc", "42", "42", "bb"});

  size_type offset = 2;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithNull_ValidScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"}, {1, 1, 0, 0, 1, 0, 1});
  strings_column_wrapper expected({"42", "a", "", "", "42", "bb", ""}, {1, 1, 0, 0, 1, 1, 0});

  size_type offset = 1;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithoutNull_NullScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"gg", "", "", "", "", "", ""}, {1, 0, 0, 0, 0, 0, 0});

  size_type offset = -3;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithNull_NullScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"}, {1, 0, 1, 1, 0, 0, 0});
  strings_column_wrapper expected({"cc", "", "", "", "d", "", ""}, {1, 0, 0, 0, 1, 0, 0});

  size_type offset = -1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithoutNull_ValidScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  size_type offset = -4;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithNull_ValidScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"}, {1, 1, 0, 0, 1, 0, 1});
  strings_column_wrapper expected({"", "gg", "42", "42", "eee", "42", "42"}, {0, 1, 1, 1, 1, 1, 1});

  size_type offset = -2;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ZeroShiftNullScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"a", "cc", "f", "gg", "bb", "d", "eee"});

  size_type offset = 0;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ZeroShiftValidScalar)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"a", "cc", "f", "gg", "bb", "d", "eee"});

  size_type offset = 0;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, VeryLargeForwardOffset)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  size_type offset = 1024;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, VeryLargeBackwardOffset)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  size_type offset = -1024;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

template <typename T>
struct groupby_shift_mixed_test : public BaseFixture {
};

TYPED_TEST_SUITE(groupby_shift_mixed_test, FixedWidthTypes);

void test_groupby_shift_multi(fixed_width_column_wrapper<K> const& key,
                              table_view const& value,
                              std::vector<size_type> offsets,
                              std::vector<std::reference_wrapper<const scalar>> fill_values,
                              table_view const& expected)
{
  groupby::groupby gb_obj(table_view({key}));
  auto got = gb_obj.shift(value, offsets, fill_values);
  CUDF_TEST_EXPECT_TABLES_EQUAL((*got.second).view(), expected);
}

TYPED_TEST(groupby_shift_mixed_test, NoFill)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper v1{"a", "bb", "cc", "d", "eee", "f", "gg"};
  fixed_width_column_wrapper<TypeParam> v2{1, 2, 3, 4, 5, 6, 7};
  table_view value{{v1, v2}};

  strings_column_wrapper e1({"", "", "a", "cc", "", "", "bb"}, {0, 0, 1, 1, 0, 0, 1});
  fixed_width_column_wrapper<TypeParam> e2({-1, 1, 3, 6, -1, 2, 4}, {0, 1, 1, 1, 0, 1, 1});
  table_view expected{{e1, e2}};

  std::vector<size_type> offset{2, 1};
  auto slr1 = cudf::make_default_constructed_scalar(column_view(v1).type());
  auto slr2 = cudf::make_default_constructed_scalar(column_view(v2).type());
  std::vector<std::reference_wrapper<const scalar>> fill_values{*slr1, *slr2};

  test_groupby_shift_multi(key, value, offset, fill_values, expected);
}

TYPED_TEST(groupby_shift_mixed_test, Fill)
{
  fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  strings_column_wrapper v1{"a", "bb", "cc", "d", "eee", "f", "gg"};
  fixed_width_column_wrapper<TypeParam> v2{1, 2, 3, 4, 5, 6, 7};
  table_view value{{v1, v2}};

  strings_column_wrapper e1({"cc", "f", "gg", "42", "d", "eee", "42"});
  fixed_width_column_wrapper<TypeParam> e2({6, 7, 42, 42, 5, 42, 42});
  table_view expected{{e1, e2}};

  std::vector<size_type> offset{-1, -2};

  auto slr1 = cudf::make_string_scalar("42");
  auto slr2 = cudf::scalar_type_t<TypeParam>(make_type_param_scalar<TypeParam>(42), true);
  std::vector<std::reference_wrapper<const scalar>> fill_values{*slr1, slr2};

  test_groupby_shift_multi(key, value, offset, fill_values, expected);
}

struct groupby_shift_fixed_point_type_test : public BaseFixture {
};

TEST_F(groupby_shift_fixed_point_type_test, Matching)
{
  fixed_width_column_wrapper<K> key{2, 3, 4, 4, 3, 2, 2, 4};
  fixed_point_column_wrapper<int32_t> v1{{10, 10, 40, 40, 20, 20, 30, 40}, numeric::scale_type{-1}};
  fixed_point_column_wrapper<int64_t> v2{{5, 5, 8, 8, 6, 7, 9, 7}, numeric::scale_type{3}};
  table_view value{{v1, v2}};

  std::vector<size_type> offset{-3, 1};
  auto slr1 = make_fixed_point_scalar<numeric::decimal32>(-42, numeric::scale_type{-1});
  auto slr2 = make_fixed_point_scalar<numeric::decimal64>(42, numeric::scale_type{3});
  std::vector<std::reference_wrapper<const scalar>> fill_values{*slr1, *slr2};

  fixed_point_column_wrapper<int32_t> e1{{-42, -42, -42, -42, -42, -42, -42, -42},
                                         numeric::scale_type{-1}};
  fixed_point_column_wrapper<int64_t> e2{{42, 5, 7, 42, 5, 42, 8, 8}, numeric::scale_type{3}};
  table_view expected{{e1, e2}};

  test_groupby_shift_multi(key, value, offset, fill_values, expected);
}

TEST_F(groupby_shift_fixed_point_type_test, MismatchScaleType)
{
  fixed_width_column_wrapper<K> key{2, 3, 4, 4, 3, 2, 2, 4};
  fixed_point_column_wrapper<int32_t> v1{{10, 10, 40, 40, 20, 20, 30, 40}, numeric::scale_type{-1}};

  std::vector<size_type> offset{-3};
  auto slr1 = make_fixed_point_scalar<numeric::decimal32>(-42, numeric::scale_type{-4});

  fixed_point_column_wrapper<int32_t> stub{{-42, -42, -42, -42, -42, -42, -42, -42},
                                           numeric::scale_type{-1}};

  EXPECT_THROW(test_groupby_shift_multi(key, table_view{{v1}}, offset, {*slr1}, table_view{{stub}}),
               logic_error);
}

TEST_F(groupby_shift_fixed_point_type_test, MismatchRepType)
{
  fixed_width_column_wrapper<K> key{2, 3, 4, 4, 3, 2, 2, 4};
  fixed_point_column_wrapper<int64_t> v1{{10, 10, 40, 40, 20, 20, 30, 40}, numeric::scale_type{-1}};

  std::vector<size_type> offset{-3};
  auto slr1 = make_fixed_point_scalar<numeric::decimal32>(-42, numeric::scale_type{-1});

  fixed_point_column_wrapper<int32_t> stub{{-42, -42, -42, -42, -42, -42, -42, -42},
                                           numeric::scale_type{-1}};

  EXPECT_THROW(test_groupby_shift_multi(key, table_view{{v1}}, offset, {*slr1}, table_view{{stub}}),
               logic_error);
}

}  // namespace test
}  // namespace cudf
