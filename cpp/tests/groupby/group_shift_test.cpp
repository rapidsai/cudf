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

#include <rmm/cuda_stream_view.hpp>
#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cudf/copying.hpp>

namespace cudf {
namespace test {

template <typename T>
struct GroupByShiftTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(GroupByShiftTest, cudf::test::FixedWidthTypes);

template <typename V>
void test_groupby_shift(cudf::test::fixed_width_column_wrapper<int32_t> const& key,
                        cudf::test::fixed_width_column_wrapper<V> const& value,
                        size_type offset,
                        scalar const& fill_value,
                        cudf::test::fixed_width_column_wrapper<V> const& expected)
{
  groupby::groupby gb_obj(table_view({key}));
  auto got = gb_obj.shift(value, offset, fill_value);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got.second, expected);
}

TYPED_TEST(GroupByShiftTest, ForwardShiftNullScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({-1, -1, 3, 5, -1, -1, 4},
                                                     {0, 0, 1, 1, 0, 0, 1});
  size_type offset = 2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(GroupByShiftTest, ForwardShiftValidScalar)
{
  using K = int32_t;
  using V = TypeParam;
  using cudf::scalar_type_t;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({42, 42, 3, 5, 42, 42, 4});
  size_type offset = 2;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift<V>(key, val, offset, slr, expected);
}

TYPED_TEST(GroupByShiftTest, BackwardShiftNullScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({8, 9, -1, -1, 7, -1, -1},
                                                     {1, 1, 0, 0, 1, 0, 0});
  size_type offset = -2;
  auto slr         = cudf::make_default_constructed_scalar(column_view(val).type());

  test_groupby_shift<V>(key, val, offset, *slr, expected);
}

TYPED_TEST(GroupByShiftTest, BackwardShiftValidScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({8, 9, 42, 42, 7, 42, 42});
  size_type offset = -2;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift<V>(key, val, offset, slr, expected);
}

}  // namespace test
}  // namespace cudf
