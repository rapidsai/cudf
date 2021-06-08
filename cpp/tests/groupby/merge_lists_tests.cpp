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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/table_view.hpp>

namespace {

constexpr bool print_all{false};  // For debugging
constexpr int32_t null{0};        // Mark for null child elements

using vcol_views = std::vector<cudf::column_view>;

auto merge_lists(vcol_views const& keys_cols, vcol_views const& values_cols)
{
  // Append all the keys and lists together.
  auto const keys   = cudf::concatenate(keys_cols);
  auto const values = cudf::concatenate(values_cols);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = *values;
  requests[0].aggregations.emplace_back(cudf::make_merge_lists_aggregation());

  cudf::groupby::groupby gb_obj(cudf::table_view({*keys}));
  auto result = gb_obj.aggregate(requests);
  return std::make_pair(std::move(result.first->release()[0]),
                        std::move(result.second[0].results[0]));
}

auto all_valids() { return cudf::test::iterator_no_null(); }

auto all_nulls() { return cudf::test::iterator_all_nulls(); }

auto null_at(cudf::size_type idx) { return cudf::test::iterator_with_null_at(idx); }

auto null_at(std::vector<cudf::size_type> const& indices)
{
  return cudf::test::iterator_with_null_at(cudf::host_span<cudf::size_type const>{indices});
}

}  // namespace

template <typename V>
struct GroupbyMergeListsTest : public cudf::test::BaseFixture {
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(GroupbyMergeListsTest, FixedWidthTypesNotBool);

TYPED_TEST(GroupbyMergeListsTest, InvalidInput)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys = keys_col{1, 2, 3};
  // The input lists column must NOT be nullable.
  auto const lists = lists_col{{lists_col{1}, lists_col{} /*NULL*/, lists_col{2}}, null_at(1)};

  EXPECT_THROW(merge_lists({keys}, {lists}), cudf::logic_error);
}

TYPED_TEST(GroupbyMergeListsTest, EmptyInput)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{};
  auto const keys2 = keys_col{};
  auto const keys3 = keys_col{};

  // Lists are all empty columns.
  auto const lists0 = lists_col{{1, 2, 3}, {4, 5, 6}};
  auto const lists1 = cudf::empty_like(lists0);
  auto const lists2 = cudf::empty_like(lists0);
  auto const lists3 = cudf::empty_like(lists0);

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{*lists1, *lists2, *lists3});
  auto const expected_keys  = keys_col{};
  auto const expected_lists = cudf::empty_like(lists0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_lists, *out_lists, print_all);
}

TYPED_TEST(GroupbyMergeListsTest, InputWithoutNull)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Keys and lists are the results of distributed collect_list.
  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{{1, 2, 3}, {4, 5, 6}};
  auto const lists2 = lists_col{{10, 11}, {11, 12}};
  auto const lists3 = lists_col{{20, 21, 22}, {23, 24, 25}, {24, 25, 26}};

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys = keys_col{1, 2, 3, 4};
  auto const expected_lists =
    lists_col{{1, 2, 3, 10, 11}, {4, 5, 6, 20, 21, 22}, {11, 12, 23, 24, 25}, {24, 25, 26}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, print_all);
}

TYPED_TEST(GroupbyMergeListsTest, InputHasNulls)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Keys and lists are the results of distributed collect_list.
  // Note that the null elements here are not sorted, while the results from current collect_list
  // are sorted.
  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{lists_col{{1, null, 3}, null_at(1)}, lists_col{4, 5, 6}};
  auto const lists2 = lists_col{lists_col{10, 11}, lists_col{{null, null, null}, all_nulls()}};
  auto const lists3 = lists_col{lists_col{20, 21, 22},
                                lists_col{{null, 24, null}, null_at({0, 2})},
                                lists_col{{24, 25, 26}, all_valids()}};

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys = keys_col{1, 2, 3, 4};
  auto const expected_lists =
    lists_col{lists_col{{1, null, 3, 10, 11}, null_at(1)},
              lists_col{4, 5, 6, 20, 21, 22},
              lists_col{{null, null, null, null, 24, null}, null_at({0, 1, 2, 3, 5})},
              lists_col{24, 25, 26}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, print_all);
}

TYPED_TEST(GroupbyMergeListsTest, InputHasEmptyLists)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{{1, 2, 3}, {}};
  auto const lists2 = lists_col{{}, {11, 12}};
  auto const lists3 = lists_col{{}, {}, {24, 25, 26}};

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{{1, 2, 3}, {}, {11, 12}, {24, 25, 26}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, print_all);
}

TYPED_TEST(GroupbyMergeListsTest, InputHasListsOfLists)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{
    lists_col{lists_col{1, 2, 3}, lists_col{4}, lists_col{5, 6}},  // key = 1
    lists_col{lists_col{}, lists_col{7}}                           // key = 2
  };
  auto const lists2 = lists_col{
    lists_col{lists_col{}, lists_col{8, 9}},     // key = 1
    lists_col{lists_col{11}, lists_col{12, 13}}  // key = 3
  };
  auto const lists3 = lists_col{
    lists_col{lists_col{14}, lists_col{15, 16, 17, 18}},             // key = 2
    lists_col{lists_col{}},                                          // key = 3
    lists_col{lists_col{17, 18, 19, 20, 21}, lists_col{18, 19, 20}}  // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    lists_col{lists_col{1, 2, 3}, lists_col{4}, lists_col{5, 6}, lists_col{}, lists_col{8, 9}},
    lists_col{lists_col{}, lists_col{7}, lists_col{14}, lists_col{15, 16, 17, 18}},
    lists_col{lists_col{11}, lists_col{12, 13}, lists_col{}},
    lists_col{lists_col{17, 18, 19, 20, 21}, lists_col{18, 19, 20}}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, print_all);
}
