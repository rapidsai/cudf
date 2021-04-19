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
#include <cudf_test/type_lists.hpp>

#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace test {

using K = int32_t;
template <typename T>
struct groupby_shift_fixed_width_test : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(groupby_shift_fixed_width_test, cudf::test::FixedWidthTypes);

template <typename V>
void test_groupby_shift_fixed_width(cudf::test::fixed_width_column_wrapper<K> const& key,
                                    cudf::test::fixed_width_column_wrapper<V> const& value,
                                    size_type offset,
                                    scalar const& fill_value,
                                    cudf::test::fixed_width_column_wrapper<V> const& expected)
{
  groupby::groupby gb_obj(table_view({key}));
  auto got = gb_obj.shift(value, offset, fill_value);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got.second, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithoutNull_NullScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({-1, -1, 3, 5, -1, -1, 4},
                                                     {0, 0, 1, 1, 0, 0, 1});
  size_type offset = 2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithNull_NullScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> expected({-1, -1, -1, -1, -1, -1, -1},
                                                     {0, 0, 0, 0, 0, 0, 0});
  size_type offset = 2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithoutNull_ValidScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5});
  cudf::test::fixed_width_column_wrapper<V> expected({42, 42, 42, 3, 5, 8, 9, 42, 42, 42, 4, 6, 7});
  size_type offset = 3;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithNull_ValidScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},
                                                {1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<V> expected(
    {42, 42, 42, 3, 5, -1, -1, 42, 42, 42, -1, -1, 7}, {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1});
  size_type offset = 3;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithoutNull_NullScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({5, 8, 9, -1, 6, 7, -1},
                                                     {1, 1, 1, 0, 1, 1, 0});
  size_type offset = -1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithNull_NullScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> expected({-1, 8, 9, -1, 6, 7, -1},
                                                     {0, 1, 1, 0, 1, 1, 0});
  size_type offset = -1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithoutNull_ValidScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected(
    {3, 5, 42, 42, 42, 42, 42, 4, 42, 42, 42, 42, 42});
  size_type offset = -5;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithNull_ValidScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},
                                                {1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<V> expected({5, -1, -1, -1, 3, 5, 42, -1, 7, 0, 2, -1, 42},
                                                     {1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1});
  size_type offset = -1;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ZeroShiftNullScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({3, 5, 8, 9, 4, 6, 7});
  size_type offset = 0;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ZeroShiftValidScalar)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected({3, 5, 8, 9, 1, 3, 5, 4, 6, 7, 0, 2, 4});
  size_type offset = 0;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width<V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, VeryLargeForwardOffset)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected(
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  size_type offset = 1024;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, VeryLargeBackwardOffset)
{
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected(
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  size_type offset = -1024;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_fixed_width<V>(key, val, offset, *slr, expected);
}

struct groupby_shift_string_test : public cudf::test::BaseFixture {
};

void test_groupby_shift_string(cudf::test::fixed_width_column_wrapper<K> const& key,
                               cudf::test::strings_column_wrapper const& value,
                               size_type offset,
                               scalar const& fill_value,
                               cudf::test::strings_column_wrapper const& expected)
{
  groupby::groupby gb_obj(table_view({key}));
  auto got = gb_obj.shift(value, offset, fill_value);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got.second, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithoutNull_NullScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"", "a", "cc", "f", "", "bb", "d"},
                                              {0, 1, 1, 1, 0, 1, 1});
  size_type offset = 1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithNull_NullScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {1, 0, 1, 1, 0, 0, 0});
  cudf::test::strings_column_wrapper expected({"", "", "a", "cc", "", "", ""},
                                              {0, 0, 1, 1, 0, 0, 0});
  size_type offset = 2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithoutNull_ValidScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "a", "cc", "42", "42", "bb"});

  size_type offset = 2;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithNull_ValidScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {1, 1, 0, 0, 1, 0, 1});
  cudf::test::strings_column_wrapper expected({"42", "a", "", "", "42", "bb", ""},
                                              {1, 1, 0, 0, 1, 1, 0});

  size_type offset = 1;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithoutNull_NullScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"gg", "", "", "", "", "", ""},
                                              {1, 0, 0, 0, 0, 0, 0});

  size_type offset = -3;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithNull_NullScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {1, 0, 1, 1, 0, 0, 0});
  cudf::test::strings_column_wrapper expected({"cc", "", "", "", "d", "", ""},
                                              {1, 0, 0, 0, 1, 0, 0});

  size_type offset = -1;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithoutNull_ValidScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  size_type offset = -4;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithNull_ValidScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {1, 1, 0, 0, 1, 0, 1});
  cudf::test::strings_column_wrapper expected({"", "gg", "42", "42", "eee", "42", "42"},
                                              {0, 1, 1, 1, 1, 1, 1});

  size_type offset = -2;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ZeroShiftNullScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"a", "cc", "f", "gg", "bb", "d", "eee"});

  size_type offset = 0;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ZeroShiftValidScalar)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"a", "cc", "f", "gg", "bb", "d", "eee"});

  size_type offset = 0;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, VeryLargeForwardOffset)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  size_type offset = 1024;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, VeryLargeBackwardOffset)
{
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  size_type offset = -1024;
  auto slr         = cudf::make_string_scalar("42");

  test_groupby_shift_string(key, val, offset, *slr, expected);
}

}  // namespace test
}  // namespace cudf
