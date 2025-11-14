/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

void test_groups(cudf::column_view const& keys,
                 cudf::column_view const& expect_grouped_keys,
                 std::vector<cudf::size_type> const& expect_group_offsets,
                 cudf::column_view const& values                = {},
                 cudf::column_view const& expect_grouped_values = {})
{
  cudf::groupby::groupby gb(cudf::table_view({keys}));
  cudf::groupby::groupby::groups gb_groups;

  if (values.size()) {
    gb_groups = gb.get_groups(cudf::table_view({values}));
  } else {
    gb_groups = gb.get_groups();
  }
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({expect_grouped_keys}), gb_groups.keys->view());

  auto got_offsets = gb_groups.offsets;
  EXPECT_EQ(expect_group_offsets.size(), got_offsets.size());
  for (auto i = 0u; i != expect_group_offsets.size(); ++i) {
    EXPECT_EQ(expect_group_offsets[i], got_offsets[i]);
  }

  if (values.size()) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({expect_grouped_values}),
                                  gb_groups.values->view());
  }
}

struct groupby_group_keys_test : public cudf::test::BaseFixture {};

template <typename V>
struct groupby_group_keys_and_values_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_group_keys_and_values_test, cudf::test::NumericTypes);

TEST_F(groupby_group_keys_test, basic)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 2, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{1, 1, 1, 2, 2, 3};
  std::vector<cudf::size_type> expect_group_offsets = {0, 3, 5, 6};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TEST_F(groupby_group_keys_test, empty_keys)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{};
  std::vector<cudf::size_type> expect_group_offsets = {0};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TEST_F(groupby_group_keys_test, all_null_keys)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 1, 2, 3, 1, 2},
                                                 cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{};
  std::vector<cudf::size_type> expect_group_offsets = {0};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TYPED_TEST(groupby_group_keys_and_values_test, basic_with_values)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({5, 4, 3, 2, 1, 0});
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> values({0, 0, 1, 1, 2, 2});
  cudf::test::fixed_width_column_wrapper<V> expect_grouped_values{2, 2, 1, 1, 0, 0};
  std::vector<cudf::size_type> expect_group_offsets = {0, 1, 2, 3, 4, 5, 6};
  test_groups(keys, expect_grouped_keys, expect_group_offsets, values, expect_grouped_values);
}

TYPED_TEST(groupby_group_keys_and_values_test, some_nulls)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 1, 3, 2, 1, 2},
                                                 {true, false, true, false, false, true});
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys({1, 2, 3},
                                                                cudf::test::iterators::no_nulls());
  cudf::test::fixed_width_column_wrapper<V> values({1, 2, 3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<V> expect_grouped_values({1, 6, 3});
  std::vector<cudf::size_type> expect_group_offsets = {0, 1, 2, 3};
  test_groups(keys, expect_grouped_keys, expect_group_offsets, values, expect_grouped_values);
}
