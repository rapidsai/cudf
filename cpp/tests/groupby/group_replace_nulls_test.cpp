
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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>
#include "cudf/table/table.hpp"
#include "cudf/utilities/error.hpp"
#include "cudf_test/table_utilities.hpp"

template <typename T>
struct GroupbyReplaceNullsTest : public cudf::test::BaseFixture {
};

using test_types = cudf::test::NumericTypes;

TYPED_TEST_CASE(GroupbyReplaceNullsTest, test_types);

template <typename K>
void TestReplaceNullsGroupby(cudf::test::fixed_width_column_wrapper<K> key,
                             cudf::test::fixed_width_column_wrapper<int32_t> input,
                             cudf::test::fixed_width_column_wrapper<K> expected_key,
                             cudf::test::fixed_width_column_wrapper<int32_t> expected_val,
                             cudf::replace_policy policy)
{
  cudf::groupby::groupby gb_obj(cudf::table_view({key}));
  auto p = gb_obj.replace_nulls(input, policy);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*p.first, cudf::table_view({expected_key}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*p.second.result, expected_val);
}

TYPED_TEST(GroupbyReplaceNullsTest, PrecedingFill)
{
  using K = TypeParam;
  using V = int32_t;

  // Group 0 value: {42, 24, null}  --> {42, 24, 24}
  // Group 1 value: {7, null, null} --> {7, 7, 7}
  std::vector<K> key = cudf::test::make_type_param_vector<K>({0, 1, 0, 1, 0, 1});
  std::vector<V> val = cudf::test::make_type_param_vector<V>({42, 7, 24, 10, 1, 1000});
  std::vector<cudf::valid_type> mask =
    cudf::test::make_type_param_vector<cudf::valid_type>({1, 1, 1, 0, 0, 0});
  std::vector<K> expect_key = cudf::test::make_type_param_vector<K>({0, 0, 0, 1, 1, 1});
  std::vector<V> expect_col = cudf::test::make_type_param_vector<V>({42, 24, 24, 7, 7, 7});

  TestReplaceNullsGroupby(
    cudf::test::fixed_width_column_wrapper<K>(key.begin(), key.end()),
    cudf::test::fixed_width_column_wrapper<V>(val.begin(), val.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<K>(expect_key.begin(), expect_key.end()),
    cudf::test::fixed_width_column_wrapper<V>(
      expect_col.begin(), expect_col.end(), cudf::test::all_valid()),
    cudf::replace_policy::PRECEDING);
}

TYPED_TEST(GroupbyReplaceNullsTest, FollowingFill)
{
  using K = TypeParam;
  using V = int32_t;

  // Group 0 value: {2, null, 32}               --> {2, 32, 32}
  // Group 1 value: {8, null, null, 128, 256}   --> {8, 128, 128, 128, 256}
  std::vector<K> key = cudf::test::make_type_param_vector<K>({0, 0, 1, 1, 0, 1, 1, 1});
  std::vector<V> val = cudf::test::make_type_param_vector<V>({2, 4, 8, 16, 32, 64, 128, 256});
  std::vector<cudf::valid_type> mask =
    cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 1, 0, 1, 0, 1, 1});
  std::vector<K> expect_key = cudf::test::make_type_param_vector<K>({0, 0, 0, 1, 1, 1, 1, 1});
  std::vector<V> expect_col =
    cudf::test::make_type_param_vector<V>({2, 32, 32, 8, 128, 128, 128, 256});

  TestReplaceNullsGroupby(
    cudf::test::fixed_width_column_wrapper<K>(key.begin(), key.end()),
    cudf::test::fixed_width_column_wrapper<V>(val.begin(), val.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<K>(expect_key.begin(), expect_key.end()),
    cudf::test::fixed_width_column_wrapper<V>(
      expect_col.begin(), expect_col.end(), cudf::test::all_valid()),
    cudf::replace_policy::FOLLOWING);
}

TYPED_TEST(GroupbyReplaceNullsTest, PrecedingFillLeadingNulls)
{
  using K = TypeParam;
  using V = int32_t;

  // Group 0 value: {null, 24, null}    --> {null, 24, 24}
  // Group 1 value: {null, null, null}  --> {null, null, null}
  std::vector<K> key = cudf::test::make_type_param_vector<K>({0, 1, 0, 1, 0, 1});
  std::vector<V> val = cudf::test::make_type_param_vector<V>({42, 7, 24, 10, 1, 1000});
  std::vector<cudf::valid_type> mask =
    cudf::test::make_type_param_vector<cudf::valid_type>({0, 0, 1, 0, 0, 0});
  std::vector<K> expect_key = cudf::test::make_type_param_vector<K>({0, 0, 0, 1, 1, 1});
  std::vector<V> expect_col = cudf::test::make_type_param_vector<V>({-1, 24, 24, -1, -1, -1});
  std::vector<cudf::valid_type> expect_valid =
    cudf::test::make_type_param_vector<cudf::valid_type>({0, 1, 1, 0, 0, 0});

  TestReplaceNullsGroupby(
    cudf::test::fixed_width_column_wrapper<K>(key.begin(), key.end()),
    cudf::test::fixed_width_column_wrapper<V>(val.begin(), val.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<K>(expect_key.begin(), expect_key.end()),
    cudf::test::fixed_width_column_wrapper<V>(
      expect_col.begin(), expect_col.end(), expect_valid.begin()),
    cudf::replace_policy::PRECEDING);
}

TYPED_TEST(GroupbyReplaceNullsTest, FollowingFillTrailingNulls)
{
  using K = TypeParam;
  using V = int32_t;

  // Group 0 value: {2, null, null}                 --> {2, null, null}
  // Group 1 value: {null, null, 64, null, null}    --> {64, 64, 64, null, null}
  std::vector<K> key = cudf::test::make_type_param_vector<K>({0, 0, 1, 1, 0, 1, 1, 1});
  std::vector<V> val = cudf::test::make_type_param_vector<V>({2, 4, 8, 16, 32, 64, 128, 256});
  std::vector<cudf::valid_type> mask =
    cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 0, 0, 0, 1, 0, 0});
  std::vector<K> expect_key = cudf::test::make_type_param_vector<K>({0, 0, 0, 1, 1, 1, 1, 1});
  std::vector<V> expect_col =
    cudf::test::make_type_param_vector<V>({2, -1, -1, 64, 64, 64, -1, -1});
  std::vector<cudf::valid_type> expect_valid =
    cudf::test::make_type_param_vector<cudf::valid_type>({1, 0, 0, 1, 1, 1, 0, 0});

  TestReplaceNullsGroupby(
    cudf::test::fixed_width_column_wrapper<K>(key.begin(), key.end()),
    cudf::test::fixed_width_column_wrapper<V>(val.begin(), val.end(), mask.begin()),
    cudf::test::fixed_width_column_wrapper<K>(expect_key.begin(), expect_key.end()),
    cudf::test::fixed_width_column_wrapper<V>(
      expect_col.begin(), expect_col.end(), expect_valid.begin()),
    cudf::replace_policy::FOLLOWING);
}
