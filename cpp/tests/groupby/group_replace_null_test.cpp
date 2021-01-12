
/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

template <typename T>
struct GroupbyReplaceNullsTest : public cudf::test::BaseFixture {
};

using test_types = cudf::test::NumericTypes;

TYPED_TEST_CASE(GroupbyReplaceNullsTest, test_types);

template <typename T>
void TestReplaceNullsGroupby(cudf::test::fixed_width_column_wrapper<T> key,
                             cudf::test::fixed_width_column_wrapper<int32_t> input,
                             cudf::test::fixed_width_column_wrapper<T> expected,
                             cudf::replace_policy policy)
{
  cudf::groupby::groupby gb_obj(table_view({key}), false, false, {}, {});
  auto result = gb_obj.replace_nulls(input, policy);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TYPED_TEST(GroupbyReplaceNullsTest, PrecedingFill)
{
  using K = TypeParam;
  using V = int32_t;

  std::vector<K> key = cudf::test::make_type_param_vector<K>({0, 1, 0, 1, 0, 1});
  std::vector<V> val = cudf::test::make_type_param_vector<V>({42, 7, 24, 10, 1, 1000});
  std::vector<cudf::valid_type> mask =
    cudf::test::make_type_param_vector<cudf::valid_type>({1, 1, 1, 0, 0, 0});
  std::vector<K> expect_col = cudf::test::make_type_param_vector<K>({42, 7, 24, 7, 24, 7});

  TestReplaceNullsGroupby(
    cudf::test::fixed_width_column_wrapper<K>(key.begin(), key.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<K>(
      expect_col.begin(), expect_col.end(), cudf::test::all_valid()),
    cudf::replace_policy::PRECEDING);
}
