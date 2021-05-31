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

#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>

constexpr bool print_all{false};  // For debugging
constexpr int32_t null{0};        // Mark for null child elements
constexpr int32_t XXX{0};         // Mark for null struct elements

template <typename V>
struct GroupbyCollectListMergeTest : public cudf::test::BaseFixture {
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(GroupbyCollectListMergeTest, FixedWidthTypesNotBool);

namespace {
auto merge(std::vector<cudf::table_view> const& keys, std::vector<cudf::column_view> const& lists)
{
  // A new groupby object is constructed using an arbitrary keys table.
  // In practice, this object can be reused from the previous `collect_list` aggregation.
  // By using the same groupby object, we can make sure that the null handling parameters used for
  // merging are the same as the parameters that were used for `collect_list` aggregation.
  auto const gb_obj = cudf::groupby::groupby(keys.front());
  auto const agg    = cudf::make_collect_list_aggregation();
  return gb_obj.merge(*agg, keys, lists);
}

auto merge(std::vector<cudf::column_view> const& keys, std::vector<cudf::column_view> const& lists)
{
  std::vector<cudf::table_view> tbl_keys;
  for (auto const& key : keys) { tbl_keys.push_back(cudf::table_view{{key}}); }
  return merge(tbl_keys, lists);
}

}  // namespace

TYPED_TEST(GroupbyCollectListMergeTest, MergeWithoutNulls)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1  = keys_col{1, 2};
  auto const keys2  = keys_col{1, 3};
  auto const keys3  = keys_col{2, 3, 4};
  auto const lists1 = lists_col{{1, 2, 3}, {4, 5, 6}};
  auto const lists2 = lists_col{{10, 11}, {11, 12}};
  auto const lists3 = lists_col{{20, 21, 22}, {23, 24, 25}, {24, 25, 26}};

  auto const result        = merge({keys1, keys2, keys3}, {lists1, lists2, lists3});
  auto const expected_keys = keys_col{1, 2, 3, 4};
  auto const expected_lists =
    lists_col{{1, 2, 3, 10, 11}, {4, 5, 6, 20, 21, 22}, {11, 12, 23, 24, 25}, {24, 25, 26}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, result.first->get_column(0), print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *result.second, print_all);
}
