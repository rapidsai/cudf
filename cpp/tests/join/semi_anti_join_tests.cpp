/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/transform_iterator.h>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using column_vector  = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

struct JoinTest : public cudf::test::BaseFixture {};

namespace {
// This function is a wrapper around cudf's join APIs that takes the gather map
// from join APIs and materializes the table that would be created by gathering
// from the joined tables. Join APIs originally returned tables like this, but
// they were modified in https://github.com/rapidsai/cudf/pull/7454. This
// helper function allows us to avoid rewriting all our tests in terms of
// gather maps.
template <std::unique_ptr<rmm::device_uvector<cudf::size_type>> (*join_impl)(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)>
std::unique_ptr<cudf::table> join_and_gather(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto left_selected      = left_input.select(left_on);
  auto right_selected     = right_input.select(right_on);
  auto const join_indices = join_impl(left_selected, right_selected, compare_nulls, stream, mr);

  auto left_indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
  auto left_indices_col  = cudf::column_view{left_indices_span};
  return cudf::gather(left_input, left_indices_col);
}
}  // namespace

std::unique_ptr<cudf::table> left_semi_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::left_semi_join>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> left_anti_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::left_anti_join>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

TEST_F(JoinTest, TestSimple)
{
  column_wrapper<int32_t> left_col0{0, 1, 2};
  column_wrapper<int32_t> right_col0{0, 1, 3};

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto result    = left_semi_join(left, right);
  auto result_cv = cudf::column_view(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                     result->size(),
                                     result->data(),
                                     nullptr,
                                     0);
  column_wrapper<cudf::size_type> expected{0, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_cv);
}

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> get_saj_tables(
  std::vector<bool> const& left_is_human_nulls, std::vector<bool> const& right_is_human_nulls)
{
  column_wrapper<int32_t> col0_0{{99, 1, 2, 0, 2}, {false, true, true, true, true}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, left_is_human_nulls.begin()};

  auto col0_3 = cudf::test::structs_column_wrapper{
    {col0_names_col, col0_ages_col, col0_is_human_col}, {true, true, true, true, true}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, -99}, {true, true, true, true, false}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_names_col = strcol_wrapper{"Carrot Ironfoundersson",
                                       "Angua von Überwald",
                                       "Detritus",
                                       "Carrot Ironfoundersson",
                                       "Samuel Vimes"};
  auto col1_ages_col  = column_wrapper<int32_t>{{351, 25, 27, 31, 48}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, false, false, false, true}, right_is_human_nulls.begin()};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  column_vector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  return {std::make_unique<Table>(std::move(cols0)), std::make_unique<Table>(std::move(cols1))};
}

TEST_F(JoinTest, SemiJoinWithStructsAndNulls)
{
  auto tables = get_saj_tables({true, true, false, true, false}, {true, false, false, true, true});

  auto result =
    left_semi_join(*tables.first, *tables.second, {0, 1, 3}, {0, 1, 3}, cudf::null_equality::EQUAL);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{99, 2}, {false, true}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, SemiJoinWithStructsAndNullsNotEqual)
{
  auto tables = get_saj_tables({true, true, false, true, true}, {true, true, false, true, true});

  auto result = left_semi_join(
    *tables.first, *tables.second, {0, 1, 3}, {0, 1, 3}, cudf::null_equality::UNEQUAL);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{2}, {true}};
  strcol_wrapper col_gold_1({"s0"}, {true});
  column_wrapper<int32_t> col_gold_2{{1}};
  auto col_gold_3_names_col = strcol_wrapper{"Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{false}, {true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, AntiJoinWithStructsAndNulls)
{
  auto tables = get_saj_tables({true, true, false, true, false}, {true, false, false, true, true});

  auto result =
    left_anti_join(*tables.first, *tables.second, {0, 1, 3}, {0, 1, 3}, cudf::null_equality::EQUAL);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{1, 2, 0}, {true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s0", "s4"}, {true, false, true});
  column_wrapper<int32_t> col_gold_2{{1, 2, 4}};
  auto col_gold_3_names_col = strcol_wrapper{"Carrot Ironfoundersson", "Detritus", "Samuel Vimes"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{27, 351, 31}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false, false}, {true, false, true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, AntiJoinWithStructsAndNullsNotEqual)
{
  auto tables = get_saj_tables({true, true, false, true, true}, {true, true, false, true, true});

  auto result = left_anti_join(
    *tables.first, *tables.second, {0, 1, 3}, {0, 1, 3}, cudf::null_equality::UNEQUAL);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{99, 1, 2, 0}, {false, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s1", "s0", "s4"}, {true, true, false, true});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4}};
  auto col_gold_3_names_col =
    strcol_wrapper{"Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes"};
  auto col_gold_3_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31}};

  auto col_gold_3_is_human_col =
    column_wrapper<bool>{{true, true, false, false}, {true, true, false, true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, AntiJoinWithStructsAndNullsOnOneSide)
{
  auto constexpr null{0};
  auto left_col0 = [] {
    column_wrapper<int32_t> child1{{1, null}, cudf::test::iterators::null_at(1)};
    column_wrapper<int32_t> child2{11, 12};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();
  auto right_col0 = [] {
    column_wrapper<int32_t> child1{1, 2, 3, 4};
    column_wrapper<int32_t> child2{11, 12, 13, 14};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto result      = cudf::left_anti_join(left, right);
  auto result_span = cudf::device_span<cudf::size_type const>{*result};
  auto result_col  = cudf::column_view{result_span};
  auto expected    = column_wrapper<cudf::size_type>{1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_col);
}
