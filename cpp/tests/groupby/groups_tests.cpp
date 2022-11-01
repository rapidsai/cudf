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

#include <cudf/types.hpp>

using namespace cudf::test::iterators;

namespace cudf {
namespace test {
struct groupby_group_keys_test : public BaseFixture {
};

template <typename V>
struct groupby_group_keys_and_values_test : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(groupby_group_keys_and_values_test, NumericTypes);

TEST_F(groupby_group_keys_test, basic)
{
  using K = int32_t;

  fixed_width_column_wrapper<K> keys{1, 1, 2, 1, 2, 3};
  fixed_width_column_wrapper<K> expect_grouped_keys{1, 1, 1, 2, 2, 3};
  std::vector<size_type> expect_group_offsets = {0, 3, 5, 6};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TEST_F(groupby_group_keys_test, empty_keys)
{
  using K = int32_t;

  fixed_width_column_wrapper<K> keys{};
  fixed_width_column_wrapper<K> expect_grouped_keys{};
  std::vector<size_type> expect_group_offsets = {0};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TEST_F(groupby_group_keys_test, all_null_keys)
{
  using K = int32_t;

  fixed_width_column_wrapper<K> keys({1, 1, 2, 3, 1, 2}, all_nulls());
  fixed_width_column_wrapper<K> expect_grouped_keys{};
  std::vector<size_type> expect_group_offsets = {0};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TYPED_TEST(groupby_group_keys_and_values_test, basic_with_values)
{
  using K = int32_t;
  using V = TypeParam;

  fixed_width_column_wrapper<K> keys({5, 4, 3, 2, 1, 0});
  fixed_width_column_wrapper<K> expect_grouped_keys{0, 1, 2, 3, 4, 5};
  fixed_width_column_wrapper<V> values({0, 0, 1, 1, 2, 2});
  fixed_width_column_wrapper<V> expect_grouped_values{2, 2, 1, 1, 0, 0};
  std::vector<size_type> expect_group_offsets = {0, 1, 2, 3, 4, 5, 6};
  test_groups(keys, expect_grouped_keys, expect_group_offsets, values, expect_grouped_values);
}

TYPED_TEST(groupby_group_keys_and_values_test, some_nulls)
{
  using K = int32_t;
  using V = TypeParam;

  fixed_width_column_wrapper<K> keys({1, 1, 3, 2, 1, 2}, {1, 0, 1, 0, 0, 1});
  fixed_width_column_wrapper<K> expect_grouped_keys({1, 2, 3}, no_nulls());
  fixed_width_column_wrapper<V> values({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<V> expect_grouped_values({1, 6, 3});
  std::vector<size_type> expect_group_offsets = {0, 1, 2, 3};
  test_groups(keys, expect_grouped_keys, expect_group_offsets, values, expect_grouped_values);
}

}  // namespace test
}  // namespace cudf
