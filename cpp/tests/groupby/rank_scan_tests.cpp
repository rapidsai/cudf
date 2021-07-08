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

#include <cudf/detail/aggregation/aggregation.hpp>

using namespace cudf::test::iterators;

namespace cudf {
namespace test {

struct groupby_rank_scan_test : public cudf::test::BaseFixture {
};

struct groupby_rank_scan_test_failures : public cudf::test::BaseFixture {
};

template <typename T>
struct typed_groupby_rank_scan_test : public cudf::test::BaseFixture {
};

using testing_type_set = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                            cudf::test::FloatingPointTypes,
                                            cudf::test::FixedPointTypes,
                                            cudf::test::ChronoTypes>;

TYPED_TEST_CASE(typed_groupby_rank_scan_test, testing_type_set);

TYPED_TEST(typed_groupby_rank_scan_test, empty_cols)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> vals{};
  fixed_width_column_wrapper<T> keys{};
  fixed_width_column_wrapper<T> order_col{};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{};
  fixed_width_column_wrapper<size_type> expected_rank_vals{};
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

TYPED_TEST(typed_groupby_rank_scan_test, zero_valid_keys)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> vals{3, 4, 5};
  fixed_width_column_wrapper<T> keys{{1, 2, 3}, all_nulls()};
  fixed_width_column_wrapper<T> order_col{3, 3, 1};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{1, 1, 2};
  fixed_width_column_wrapper<size_type> expected_rank_vals{1, 1, 3};

  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

TYPED_TEST(typed_groupby_rank_scan_test, zero_valid_orders)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> vals{3, 4, 5, 6};
  fixed_width_column_wrapper<T> keys{1, 1, 3, 3};
  fixed_width_column_wrapper<T> order_col{{5, 6, 7, 8}, all_nulls()};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{1, 1, 1, 1};
  fixed_width_column_wrapper<size_type> expected_rank_vals{1, 1, 1, 1};

  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

TYPED_TEST(typed_groupby_rank_scan_test, basic)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> vals{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50}};
  fixed_width_column_wrapper<T> keys{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  fixed_width_column_wrapper<T> order_col{5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals = {
    {1, 1, 1, 2, 2, 2, 3, 1, 2, 2, 3, 3}};
  fixed_width_column_wrapper<size_type> expected_rank_vals =
    fixed_width_column_wrapper<size_type>{{1, 1, 1, 4, 4, 4, 7, 1, 2, 2, 4, 4}};

  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

TYPED_TEST(typed_groupby_rank_scan_test, null_values_and_orders)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> vals{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                     {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> keys{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  fixed_width_column_wrapper<T> order_col{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -5},
                                          {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 2, 3, 4, 5, 5, 1, 1, 2, 3, 3, 3}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 2, 3, 4, 5, 5, 1, 1, 3, 4, 4, 4}};

  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

TYPED_TEST(typed_groupby_rank_scan_test, null_values_orders_and_keys)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 2};
  fixed_width_column_wrapper<T> keys = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  fixed_width_column_wrapper<T> order_col{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6},
                                          {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 2, 3, 4, 5, 5, 1, 1, 2, 1, 1, 2}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 2, 3, 4, 5, 5, 1, 1, 3, 1, 1, 3}};

  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

/* Struct support dependent on https://github.com/rapidsai/cudf/issues/8683
TYPED_TEST(typed_groupby_rank_scan_test, structs)
{
  using T = TypeParam;

  auto col = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto col2 = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto strings =  strings_column_wrapper{{"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9",
"9", "10d"}, null_at(8)}; auto strings2 = strings_column_wrapper{{"0a", "0a", "2a", "2a", "3b", "5",
"6c", "6c", "6c", "9", "9", "10d"}, null_at(8)}; auto struct_col = structs_column_wrapper{col,
strings}.release(); cudf::table_view
struct_order{std::vector<cudf::column_view>{struct_col->view()}}; cudf::table_view
col_order{std::vector<cudf::column_view>{col2, strings2}};

  strings_column_wrapper vals{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "4", "2"};
  strings_column_wrapper keys = {{"0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1"},
                                 {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = vals;
  requests[0].aggregations.push_back(make_dense_rank_aggregation(struct_order));
  requests[0].aggregations.push_back(make_dense_rank_aggregation(col_order));
  requests[0].aggregations.push_back(make_rank_aggregation(struct_order));
  requests[0].aggregations.push_back(make_rank_aggregation(col_order));

  groupby::groupby gb_obj(table_view({keys}), null_policy::INCLUDE,  sorted::YES);
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[0], *result.second[0].results[1],
true); CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[2],
*result.second[0].results[3], true);
}
*/

/* List support dependent on https://github.com/rapidsai/cudf/issues/8683
template <typename T>
struct list_groupby_rank_scan_test : public cudf::test::BaseFixture {
};

using list_test_type_set = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                            cudf::test::FloatingPointTypes,
                                            cudf::test::FixedPointTypes>;

TYPED_TEST_CASE(list_groupby_rank_scan_test, list_type_set);

TYPED_TEST(list_groupby_rank_scan_test, lists)
{
  using T = TypeParam;

  auto list_col = lists_column_wrapper<T>{{0, 0}, {0, 0}, {7, 2}, {7, 2}, {7, 3}, {5, 5}, {4, 6},
                                             {4, 6}, {4, 6}, {9, 9}, {9, 9}, {9, 10}}.release();
  fixed_width_column_wrapper<T> element1{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9};
  fixed_width_column_wrapper<T> element2{0, 0, 2, 2, 3, 5, 6, 6, 6, 9, 9, 10};
  cudf::table_view list_order{std::vector<cudf::column_view>{list_col->view()}};
  cudf::table_view col_order{std::vector<cudf::column_view>{element1, element2}};

  fixed_width_column_wrapper<uint32_t> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 2};
  fixed_width_column_wrapper<T> keys = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = vals;
  requests[0].aggregations.push_back(make_dense_rank_aggregation(list_order));
  requests[0].aggregations.push_back(make_dense_rank_aggregation(col_order));
  requests[0].aggregations.push_back(make_rank_aggregation(list_order));
  requests[0].aggregations.push_back(make_rank_aggregation(col_order));

  groupby::groupby gb_obj(table_view({keys}), null_policy::INCLUDE,  sorted::YES);
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[0], *result.second[0].results[1],
true); CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[2],
*result.second[0].results[3], true);
}
*/

TEST(groupby_rank_scan_test, bools)
{
  fixed_width_column_wrapper<bool> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 2};
  fixed_width_column_wrapper<bool> keys = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                                           {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  fixed_width_column_wrapper<bool> order_col{{0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1},
                                             {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 1, 2, 2, 2, 2, 1, 2, 3, 1, 1, 2}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 1, 3, 3, 3, 3, 1, 2, 3, 1, 1, 3}};

  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

TEST(groupby_rank_scan_test, strings)
{
  strings_column_wrapper vals{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "4", "2"};
  strings_column_wrapper keys = {{"0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1"},
                                 {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  strings_column_wrapper order_col{
    {"-1", "-2", "-2", "-2", "-3", "-3", "-4", "-4", "-4", "-5", "-5", "-6"},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 2, 3, 4, 5, 5, 1, 1, 2, 1, 1, 2}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 2, 3, 4, 5, 5, 1, 1, 3, 1, 1, 3}};

  test_single_scan(keys,
                   vals,
                   keys,
                   expected_dense_vals,
                   make_dense_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   vals,
                   keys,
                   expected_rank_vals,
                   make_rank_aggregation(order_table),
                   null_policy::INCLUDE,
                   sorted::YES);
}

TEST_F(groupby_rank_scan_test_failures, test_exception_triggers)
{
  using T = uint32_t;

  fixed_width_column_wrapper<T> vals{3, 4, 5};
  fixed_width_column_wrapper<T> keys{{1, 2, 3}, {1, 1, 0}};
  fixed_width_column_wrapper<T> order_col{3, 3, 1};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(keys,
                     vals,
                     keys,
                     vals,
                     make_dense_rank_aggregation(order_table),
                     null_policy::INCLUDE,
                     sorted::NO),
    "Dense rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(
      keys, vals, keys, vals, make_rank_aggregation(order_table), null_policy::INCLUDE, sorted::NO),
    "Rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(keys,
                     vals,
                     keys,
                     vals,
                     make_dense_rank_aggregation(order_table),
                     null_policy::EXCLUDE,
                     sorted::YES),
    "Dense rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys,
                                             vals,
                                             keys,
                                             vals,
                                             make_rank_aggregation(order_table),
                                             null_policy::EXCLUDE,
                                             sorted::YES),
                            "Rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(keys,
                     vals,
                     keys,
                     vals,
                     make_dense_rank_aggregation(order_table),
                     null_policy::EXCLUDE,
                     sorted::NO),
    "Dense rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(
      keys, vals, keys, vals, make_rank_aggregation(order_table), null_policy::EXCLUDE, sorted::NO),
    "Rank aggregate in groupby scan requires the keys to be presorted");

  fixed_width_column_wrapper<T> extended_order_col{3, 3, 1, 1};
  cudf::table_view extended_order_table{std::vector<cudf::column_view>{extended_order_col}};

  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys,
                                             vals,
                                             keys,
                                             vals,
                                             make_dense_rank_aggregation(extended_order_table),
                                             null_policy::INCLUDE,
                                             sorted::YES),
                            "Number of rows in the key and order tables do not match");

  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys,
                                             vals,
                                             keys,
                                             vals,
                                             make_rank_aggregation(extended_order_table),
                                             null_policy::INCLUDE,
                                             sorted::YES),
                            "Number of rows in the key and order tables do not match");
}

}  // namespace test
}  // namespace cudf
