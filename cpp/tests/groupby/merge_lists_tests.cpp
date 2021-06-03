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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/table_view.hpp>

constexpr bool print_all{false};  // For debugging
constexpr int32_t null{0};        // Mark for null child elements
constexpr int32_t XXX{0};         // Mark for null struct elements

template <typename V>
struct GroupbyMergeListsTest : public cudf::test::BaseFixture {
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(GroupbyMergeListsTest, FixedWidthTypesNotBool);

namespace {

auto table_views(std::initializer_list<cudf::column_view> const& cols)
{
  std::vector<cudf::table_view> tables;
  for (auto const& col : cols) { tables.push_back(cudf::table_view{{col}}); }
  return tables;
}
//
}  // namespace

TYPED_TEST(GroupbyMergeListsTest, MergeWithoutNulls)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Keys and lists are the results of distributed collect_list.
  auto const keys1  = keys_col{1, 2};
  auto const keys2  = keys_col{1, 3};
  auto const keys3  = keys_col{2, 3, 4};
  auto const lists1 = lists_col{{1, 2, 3}, {4, 5, 6}};
  auto const lists2 = lists_col{{10, 11}, {11, 12}};
  auto const lists3 = lists_col{{20, 21, 22}, {23, 24, 25}, {24, 25, 26}};

  // Append all the keys and lists together.
  auto const keys  = cudf::concatenate(table_views({keys1, keys2, keys3}));
  auto const lists = cudf::concatenate(table_views({lists1, lists2, lists3}));

  //  printf("line: %d\n", __LINE__);
  //  cudf::test::print(keys->get_column(0));

  //  printf("line: %d\n", __LINE__);
  //  cudf::test::print(lists->get_column(0));

  auto const expected_keys = keys_col{1, 2, 3, 4};
  auto const expected_lists =
    lists_col{{1, 2, 3, 10, 11}, {4, 5, 6, 20, 21, 22}, {11, 12, 23, 24, 25}, {24, 25, 26}};

  cudf::test::test_single_agg(keys->get_column(0),
                              lists->get_column(0),
                              expected_keys,
                              expected_lists,
                              cudf::make_merge_lists_aggregation());

  //  exit(0);
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, result.first->get_column(0), print_all);
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *result.second, print_all);
}
