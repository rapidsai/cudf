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

inline void test_pair_rank_scans(column_view const& keys,
                                 column_view const& order,
                                 column_view const& expected_dense,
                                 column_view const& expected_rank,
                                 null_policy include_null_keys = null_policy::INCLUDE,
                                 sorted keys_are_sorted        = sorted::YES)
{
  test_single_scan(keys,
                   order,
                   keys,
                   expected_dense,
                   make_dense_rank_aggregation<groupby_scan_aggregation>(),
                   null_policy::INCLUDE,
                   sorted::YES);
  test_single_scan(keys,
                   order,
                   keys,
                   expected_rank,
                   make_rank_aggregation<groupby_scan_aggregation>(),
                   null_policy::INCLUDE,
                   sorted::YES);
}

struct groupby_rank_scan_test : public BaseFixture {
};

struct groupby_rank_scan_test_failures : public BaseFixture {
};

template <typename T>
struct typed_groupby_rank_scan_test : public BaseFixture {
};

using testing_type_set =
  Concat<IntegralTypesNotBool, FloatingPointTypes, FixedPointTypes, ChronoTypes>;

TYPED_TEST_SUITE(typed_groupby_rank_scan_test, testing_type_set);

TYPED_TEST(typed_groupby_rank_scan_test, empty_cols)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> keys{};
  fixed_width_column_wrapper<T> order_col{};
  structs_column_wrapper struct_order{};

  fixed_width_column_wrapper<size_type> expected_dense_vals{};
  fixed_width_column_wrapper<size_type> expected_rank_vals{};

  test_pair_rank_scans(keys, order_col, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
}

TYPED_TEST(typed_groupby_rank_scan_test, zero_valid_keys)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> keys{{1, 2, 3}, all_nulls()};
  fixed_width_column_wrapper<T> order_col1{3, 3, 1};
  fixed_width_column_wrapper<T> order_col2{3, 3, 1};
  fixed_width_column_wrapper<T> order_col3{3, 3, 1};
  structs_column_wrapper struct_order{order_col2, order_col3};

  fixed_width_column_wrapper<size_type> expected_dense_vals{1, 1, 2};
  fixed_width_column_wrapper<size_type> expected_rank_vals{1, 1, 3};

  test_pair_rank_scans(keys, order_col1, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
}

TYPED_TEST(typed_groupby_rank_scan_test, zero_valid_orders)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> keys{1, 1, 3, 3};
  fixed_width_column_wrapper<T> order_col1{{5, 6, 7, 8}, all_nulls()};
  fixed_width_column_wrapper<T> order_col2{{5, 6, 7, 8}, all_nulls()};
  fixed_width_column_wrapper<T> order_col3{{5, 6, 7, 8}, all_nulls()};
  fixed_width_column_wrapper<T> order_col4{{5, 6, 7, 8}, all_nulls()};
  fixed_width_column_wrapper<T> order_col5{{5, 6, 7, 8}, all_nulls()};
  structs_column_wrapper struct_order{order_col2, order_col3};
  structs_column_wrapper struct_order_with_nulls{{order_col4, order_col5}, all_nulls()};

  fixed_width_column_wrapper<size_type> expected_dense_vals{1, 1, 1, 1};
  fixed_width_column_wrapper<size_type> expected_rank_vals{1, 1, 1, 1};

  test_pair_rank_scans(keys, order_col1, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order_with_nulls, expected_dense_vals, expected_rank_vals);
}

TYPED_TEST(typed_groupby_rank_scan_test, basic)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> keys{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  fixed_width_column_wrapper<T> order_col1{5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1};
  fixed_width_column_wrapper<T> order_col2{5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1};
  fixed_width_column_wrapper<T> order_col3{5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1};
  structs_column_wrapper struct_order{order_col2, order_col3};

  fixed_width_column_wrapper<size_type> expected_dense_vals = {
    {1, 1, 1, 2, 2, 2, 3, 1, 2, 2, 3, 3}};
  fixed_width_column_wrapper<size_type> expected_rank_vals =
    fixed_width_column_wrapper<size_type>{{1, 1, 1, 4, 4, 4, 7, 1, 2, 2, 4, 4}};

  test_pair_rank_scans(keys, order_col1, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
}

TYPED_TEST(typed_groupby_rank_scan_test, null_orders)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> keys{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  fixed_width_column_wrapper<T> order_col1{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -5},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col2{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -5},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col3{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -5},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col4{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -5},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col5{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -5},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  structs_column_wrapper struct_order{order_col2, order_col3};
  structs_column_wrapper struct_order_with_nulls{{order_col4, order_col5},
                                                 {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 2, 3, 4, 5, 5, 1, 1, 2, 3, 3, 3}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 2, 3, 4, 5, 5, 1, 1, 3, 4, 4, 4}};

  test_pair_rank_scans(keys, order_col1, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
}

TYPED_TEST(typed_groupby_rank_scan_test, null_orders_and_keys)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> keys = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  fixed_width_column_wrapper<T> order_col1{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col2{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col3{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col4{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<T> order_col5{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6},
                                           {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  structs_column_wrapper struct_order{order_col2, order_col3};
  structs_column_wrapper struct_order_with_nulls{{order_col4, order_col5},
                                                 {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 2, 3, 4, 5, 5, 1, 1, 2, 1, 1, 2}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 2, 3, 4, 5, 5, 1, 1, 3, 1, 1, 3}};

  test_pair_rank_scans(keys, order_col1, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order_with_nulls, expected_dense_vals, expected_rank_vals);
}

TYPED_TEST(typed_groupby_rank_scan_test, mixedStructs)
{
  auto col     = fixed_width_column_wrapper<int>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto strings = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  auto struct_col = structs_column_wrapper{{col, strings}, null_at(11)}.release();

  strings_column_wrapper keys = {{"0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1"},
                                 {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_dense_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 2, 2, 3, 4, 1, 1, 2, 1, 1, 2};
  auto expected_rank_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 3, 3, 5, 6, 1, 1, 3, 1, 1, 3};

  std::vector<groupby::scan_request> requests;
  requests.emplace_back(groupby::scan_request());
  requests[0].values = *struct_col;
  requests[0].aggregations.push_back(make_dense_rank_aggregation<groupby_scan_aggregation>());
  requests[0].aggregations.push_back(make_rank_aggregation<groupby_scan_aggregation>());

  groupby::groupby gb_obj(table_view({keys}), null_policy::INCLUDE, sorted::YES);
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[0], expected_dense_vals);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[1], expected_rank_vals);
}

TYPED_TEST(typed_groupby_rank_scan_test, nestedStructs)
{
  using T = TypeParam;

  auto col1     = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto col2     = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto col3     = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto col4     = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto strings1 = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  auto strings2 = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  auto struct_col    = structs_column_wrapper{col1, strings1};
  auto nested_col    = structs_column_wrapper{struct_col, col2}.release();
  auto flattened_col = structs_column_wrapper{col3, strings2, col4}.release();

  strings_column_wrapper keys = {{"0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1"},
                                 {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  std::vector<groupby::scan_request> requests;
  requests.emplace_back(groupby::scan_request());
  requests.emplace_back(groupby::scan_request());
  requests[0].values = *nested_col;
  requests[0].aggregations.push_back(make_dense_rank_aggregation<groupby_scan_aggregation>());
  requests[0].aggregations.push_back(make_rank_aggregation<groupby_scan_aggregation>());
  requests[1].values = *flattened_col;
  requests[1].aggregations.push_back(make_dense_rank_aggregation<groupby_scan_aggregation>());
  requests[1].aggregations.push_back(make_rank_aggregation<groupby_scan_aggregation>());

  groupby::groupby gb_obj(table_view({keys}), null_policy::INCLUDE, sorted::YES);
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[0], *result.second[1].results[0]);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[1], *result.second[1].results[1]);
}

TYPED_TEST(typed_groupby_rank_scan_test, structsWithNullPushdown)
{
  using T = TypeParam;

  auto col1     = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto col2     = fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
  auto strings1 = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  auto strings2 = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};

  std::vector<std::unique_ptr<column>> struct_columns;
  struct_columns.push_back(col1.release());
  struct_columns.push_back(strings1.release());
  auto struct_col =
    cudf::make_structs_column(12, std::move(struct_columns), 0, rmm::device_buffer{});
  auto const struct_nulls =
    thrust::host_vector<bool>(std::vector<bool>{1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  struct_col->set_null_mask(
    cudf::test::detail::make_null_mask(struct_nulls.begin(), struct_nulls.end()));

  std::vector<std::unique_ptr<column>> null_struct_columns;
  null_struct_columns.push_back(col2.release());
  null_struct_columns.push_back(strings2.release());
  auto null_col =
    cudf::make_structs_column(12, std::move(null_struct_columns), 0, rmm::device_buffer{});
  null_col->set_null_mask(create_null_mask(12, cudf::mask_state::ALL_NULL));

  strings_column_wrapper keys = {{"0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1"},
                                 {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  std::vector<groupby::scan_request> requests;
  requests.emplace_back(groupby::scan_request());
  requests.emplace_back(groupby::scan_request());
  requests[0].values = *struct_col;
  requests[0].aggregations.push_back(make_dense_rank_aggregation<groupby_scan_aggregation>());
  requests[0].aggregations.push_back(make_rank_aggregation<groupby_scan_aggregation>());
  requests[1].values = *null_col;
  requests[1].aggregations.push_back(make_dense_rank_aggregation<groupby_scan_aggregation>());
  requests[1].aggregations.push_back(make_rank_aggregation<groupby_scan_aggregation>());

  groupby::groupby gb_obj(table_view({keys}), null_policy::INCLUDE, sorted::YES);
  auto result = gb_obj.scan(requests);

  auto expected_dense_vals =
    fixed_width_column_wrapper<size_type>{1, 2, 2, 3, 4, 5, 1, 1, 2, 1, 1, 2};
  auto expected_rank_vals =
    fixed_width_column_wrapper<size_type>{1, 2, 2, 4, 5, 6, 1, 1, 3, 1, 1, 3};
  auto expected_null_result =
    fixed_width_column_wrapper<size_type>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[0], expected_dense_vals);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[1], expected_rank_vals);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[1].results[0], expected_null_result);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[1].results[1], expected_null_result);
}

/* List support dependent on https://github.com/rapidsai/cudf/issues/8683
template <typename T>
struct list_groupby_rank_scan_test : public BaseFixture {
};

using list_test_type_set = Concat<IntegralTypesNotBool,
                                              FloatingPointTypes,
                                              FixedPointTypes>;

TYPED_TEST_SUITE(list_groupby_rank_scan_test, list_test_type_set);

TYPED_TEST(list_groupby_rank_scan_test, lists)
{
  using T = TypeParam;

  auto list_col = lists_column_wrapper<T>{
    {0, 0},
    {0, 0},
    {7, 2},
    {7, 2},
    {7, 3},
    {5, 5},
    {4, 6},
    {4, 6},
    {4, 6},
    {9, 9},
    {9, 9},
    {9, 10}}.release();
  fixed_width_column_wrapper<T> element1{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9};
  fixed_width_column_wrapper<T> element2{0, 0, 2, 2, 3, 5, 6, 6, 6, 9, 9, 10};
  auto struct_col = structs_column_wrapper{element1, element2}.release();

  fixed_width_column_wrapper<T> keys = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  std::vector<groupby::scan_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = list_col;
  requests[0].aggregations.push_back(make_dense_rank_aggregation<groupby_scan_aggregation>());
  requests[0].aggregations.push_back(make_rank_aggregation<groupby_scan_aggregation>());
  requests[1].values = struct_col;
  requests[1].aggregations.push_back(make_dense_rank_aggregation<groupby_scan_aggregation>());
  requests[1].aggregations.push_back(make_rank_aggregation<groupby_scan_aggregation>());

  groupby::groupby gb_obj(table_view({keys}), null_policy::INCLUDE, sorted::YES);
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *result.second[0].results[0], *result.second[1].results[0]);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *result.second[0].results[2], *result.second[1].results[2]);
}
*/

TEST(groupby_rank_scan_test, bools)
{
  fixed_width_column_wrapper<bool> keys = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                                           {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  fixed_width_column_wrapper<bool> order_col1{{0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1},
                                              {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<bool> order_col2{{0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1},
                                              {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<bool> order_col3{{0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1},
                                              {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<bool> order_col4{{0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1},
                                              {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};
  fixed_width_column_wrapper<bool> order_col5{{0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1},
                                              {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};
  structs_column_wrapper struct_order{order_col2, order_col3};
  structs_column_wrapper struct_order_with_nulls{{order_col4, order_col5},
                                                 {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 1, 2, 2, 2, 2, 1, 2, 3, 1, 1, 2}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 1, 3, 3, 3, 3, 1, 2, 3, 1, 1, 3}};

  test_pair_rank_scans(keys, order_col1, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order_with_nulls, expected_dense_vals, expected_rank_vals);
}

TEST(groupby_rank_scan_test, strings)
{
  strings_column_wrapper keys = {{"0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1"},
                                 {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  strings_column_wrapper order_col1{
    {"-1", "-2", "-2", "-2", "-3", "-3", "-4", "-4", "-4", "-5", "-5", "-6"},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  strings_column_wrapper order_col2{
    {"-1", "-2", "-2", "-2", "-3", "-3", "-4", "-4", "-4", "-5", "-5", "-6"},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  strings_column_wrapper order_col3{
    {"-1", "-2", "-2", "-2", "-3", "-3", "-4", "-4", "-4", "-5", "-5", "-6"},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  strings_column_wrapper order_col4{
    {"-1", "-2", "-2", "-2", "-3", "-3", "-4", "-4", "-4", "-5", "-5", "-6"},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  strings_column_wrapper order_col5{
    {"-1", "-2", "-2", "-2", "-3", "-3", "-4", "-4", "-4", "-5", "-5", "-6"},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
  structs_column_wrapper struct_order{order_col2, order_col3};
  structs_column_wrapper struct_order_with_nulls{{order_col4, order_col5},
                                                 {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0}};

  fixed_width_column_wrapper<size_type> expected_dense_vals{{1, 2, 3, 4, 5, 5, 1, 1, 2, 1, 1, 2}};
  fixed_width_column_wrapper<size_type> expected_rank_vals{{1, 2, 3, 4, 5, 5, 1, 1, 3, 1, 1, 3}};

  test_pair_rank_scans(keys, order_col1, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order, expected_dense_vals, expected_rank_vals);
  test_pair_rank_scans(keys, struct_order_with_nulls, expected_dense_vals, expected_rank_vals);
}

TEST_F(groupby_rank_scan_test_failures, test_exception_triggers)
{
  using T = uint32_t;

  fixed_width_column_wrapper<T> keys{{1, 2, 3}, {1, 1, 0}};
  fixed_width_column_wrapper<T> col{3, 3, 1};

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(keys,
                     col,
                     keys,
                     col,
                     make_dense_rank_aggregation<groupby_scan_aggregation>(),
                     null_policy::INCLUDE,
                     sorted::NO),
    "Dense rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys,
                                             col,
                                             keys,
                                             col,
                                             make_rank_aggregation<groupby_scan_aggregation>(),
                                             null_policy::INCLUDE,
                                             sorted::NO),
                            "Rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(keys,
                     col,
                     keys,
                     col,
                     make_dense_rank_aggregation<groupby_scan_aggregation>(),
                     null_policy::EXCLUDE,
                     sorted::YES),
    "Dense rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys,
                                             col,
                                             keys,
                                             col,
                                             make_rank_aggregation<groupby_scan_aggregation>(),
                                             null_policy::EXCLUDE,
                                             sorted::YES),
                            "Rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(
    test_single_scan(keys,
                     col,
                     keys,
                     col,
                     make_dense_rank_aggregation<groupby_scan_aggregation>(),
                     null_policy::EXCLUDE,
                     sorted::NO),
    "Dense rank aggregate in groupby scan requires the keys to be presorted");

  CUDF_EXPECT_THROW_MESSAGE(test_single_scan(keys,
                                             col,
                                             keys,
                                             col,
                                             make_rank_aggregation<groupby_scan_aggregation>(),
                                             null_policy::EXCLUDE,
                                             sorted::NO),
                            "Rank aggregate in groupby scan requires the keys to be presorted");
}

}  // namespace test
}  // namespace cudf
