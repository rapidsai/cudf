/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <limits>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

struct DistinctJoinTest : public cudf::test::BaseFixture {
  void compare_to_reference(
    cudf::table_view const& build_table,
    cudf::table_view const& probe_table,
    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> const& result,
    cudf::table_view const& expected_table)
  {
    auto const& [build_join_indices, probe_join_indices] = result;

    auto build_indices_span = cudf::device_span<cudf::size_type const>{*build_join_indices};
    auto probe_indices_span = cudf::device_span<cudf::size_type const>{*probe_join_indices};

    auto build_indices_col = cudf::column_view{build_indices_span};
    auto probe_indices_col = cudf::column_view{probe_indices_span};

    auto constexpr oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
    auto joined_cols          = cudf::gather(build_table, build_indices_col, oob_policy)->release();
    auto right_cols           = cudf::gather(probe_table, probe_indices_col, oob_policy)->release();

    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(right_cols.begin()),
                       std::make_move_iterator(right_cols.end()));
    auto joined_table        = std::make_unique<cudf::table>(std::move(joined_cols));
    auto result_sort_order   = cudf::sorted_order(joined_table->view());
    auto sorted_joined_table = cudf::gather(joined_table->view(), *result_sort_order);

    auto expected_sort_order = cudf::sorted_order(expected_table);
    auto sorted_expected     = cudf::gather(expected_table, *expected_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_expected, *sorted_joined_table);
  }
};

TEST_F(DistinctJoinTest, IntegerInnerJoin)
{
  auto constexpr size = 2024;

  auto const init = cudf::numeric_scalar<int32_t>{0};

  auto build = cudf::sequence(size, init, cudf::numeric_scalar<int32_t>{1});
  auto probe = cudf::sequence(size, init, cudf::numeric_scalar<int32_t>{2});

  auto build_table = cudf::table_view{{build->view()}};
  auto probe_table = cudf::table_view{{probe->view()}};

  auto distinct_join = cudf::distinct_hash_join<cudf::has_nested::NO>{
    build_table, probe_table, cudf::nullable_join::NO};

  auto result = distinct_join.inner_join();

  auto constexpr gold_size = size / 2;
  auto gold                = cudf::sequence(gold_size, init, cudf::numeric_scalar<int32_t>{2});
  this->compare_to_reference(build_table, probe_table, result, cudf::table_view{{gold->view()}});
}

TEST_F(DistinctJoinTest, InnerJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{1, 2, 3, 4, 5}};
  strcol_wrapper col0_1({"s0", "s0", "s3", "s4", "s5"});
  column_wrapper<int32_t> col0_2{{9, 9, 9, 9, 9}};

  column_wrapper<int32_t> col1_0{{1, 2, 3, 4, 9}};
  strcol_wrapper col1_1({"s0", "s0", "s0", "s4", "s4"});
  column_wrapper<int32_t> col1_2{{9, 9, 9, 0, 9}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table build(std::move(cols0));
  Table probe(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join<cudf::has_nested::YES>{build.view(), probe.view()};
  auto result        = distinct_join.inner_join();

  column_wrapper<int32_t> col_gold_0{{1, 2}};
  strcol_wrapper col_gold_1({"s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{9, 9}};
  column_wrapper<int32_t> col_gold_3{{1, 2}};
  strcol_wrapper col_gold_4({"s0", "s0"});
  column_wrapper<int32_t> col_gold_5{{9, 9}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(build.view(), probe.view(), result, gold.view());
}

TEST_F(DistinctJoinTest, InnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{1, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{1, 2, 0, 2, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s0", "s1"});
  column_wrapper<int32_t> col1_2{{1, 1, 1, 1, 1}, {0, 1, 1, 0, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table build(std::move(cols0));
  Table probe(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join<cudf::has_nested::YES>{build.view(), probe.view()};
  auto result        = distinct_join.inner_join();

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_2{{1, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2}};
  strcol_wrapper col_gold_4({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_5{{1, 1}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(build.view(), probe.view(), result, gold.view());
}

TEST_F(DistinctJoinTest, InnerJoinWithStructsAndNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 4}, {1, 1, 1, 1, 0}};
  std::initializer_list<std::string> col0_names = {
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_names_col = strcol_wrapper{col0_names.begin(), col0_names.end()};
  auto col0_ages_col  = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col = column_wrapper<bool>{{true, true, false, false, false}, {1, 1, 0, 1, 0}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 1, 1, 2, 0}, {1, 0, 1, 1, 1}};
  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Detritus",
                                                   "Carrot Ironfoundersson",
                                                   "Samuel Vimes"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{31, 25, 351, 27, 48}};

  auto col1_is_human_col = column_wrapper<bool>{{true, false, false, false, true}, {1, 0, 0, 1, 1}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table probe(std::move(cols0));
  Table build(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join<cudf::has_nested::YES>{build.view(), probe.view()};
  auto result        = distinct_join.inner_join();

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 4}, {1, 0}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {1, 0}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_wrapper<int32_t> col_gold_4{{3, 2}};
  strcol_wrapper col_gold_5({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_6{{0, -1}, {1, 0}};
  auto col_gold_7_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_7_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_7_is_human_col = column_wrapper<bool>{{true, false}, {1, 0}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {col_gold_7_names_col, col_gold_7_ages_col, col_gold_7_is_human_col}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  cols_gold.push_back(col_gold_6.release());
  cols_gold.push_back(col_gold_7.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(build.view(), probe.view(), result, gold.view());
}

TEST_F(DistinctJoinTest, EmptyBuildTableInnerJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table build(std::move(cols0));
  Table probe(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join<cudf::has_nested::NO>{build.view(), probe.view()};
  auto result        = distinct_join.inner_join();

  this->compare_to_reference(build.view(), probe.view(), result, build.view());
}

TEST_F(DistinctJoinTest, EmptyProbeTableInnerJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table build(std::move(cols0));
  Table probe(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join<cudf::has_nested::NO>{build.view(), probe.view()};
  auto result        = distinct_join.inner_join();

  this->compare_to_reference(build.view(), probe.view(), result, probe.view());
}
