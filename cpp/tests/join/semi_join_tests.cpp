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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <thrust/iterator/transform_iterator.h>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using column_vector  = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

struct JoinTest : public cudf::test::BaseFixture {
};

TEST_F(JoinTest, SemiJoinWithStructsAndNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col = column_wrapper<bool>{{true, true, false, false, false}, {1, 1, 0, 1, 0}};

  auto col0_3 = cudf::test::structs_column_wrapper{
    {col0_names_col, col0_ages_col, col0_is_human_col}, {1, 1, 1, 1, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};
  auto col1_names_col = strcol_wrapper{"Carrot Ironfoundersson",
                                       "Angua von Überwald",
                                       "Detritus",
                                       "Carrot Ironfoundersson",
                                       "Samuel Vimes"};
  auto col1_ages_col  = column_wrapper<int32_t>{{351, 25, 27, 31, 48}};

  auto col1_is_human_col = column_wrapper<bool>{{true, false, false, false, true}, {1, 0, 0, 1, 1}};

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

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = cudf::left_semi_join(t0, t1, {0, 1, 3}, {0, 1, 3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {1, 0}};

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
