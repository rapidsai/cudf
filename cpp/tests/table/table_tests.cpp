/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <random>
#include <memory>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

using s_col_wrapper  = cudf::test::strings_column_wrapper;

using CVector     = std::vector<std::unique_ptr<cudf::column>>;
using column      = cudf::column;
using column_view = cudf::column_view;
using TView       = cudf::table_view;
using Table       = cudf::experimental::table;

struct TableTest : public cudf::test::BaseFixture {};

TEST_F(TableTest, EmptyColumnedTable)
{
    std::vector<column_view> cols{};

    TView input(cols);
    cudf::size_type expected = 0;

    EXPECT_EQ(input.num_columns(), expected);
}

TEST_F(TableTest, ValidateConstructorTableViewToTable)
{
    column_wrapper <int8_t>  col1{{1,2,3,4}};
    column_wrapper <int8_t>  col2{{1,2,3,4}};

    CVector cols;
    cols.push_back(col1.release());
    cols.push_back(col2.release());

    Table input_table(std::move(cols));

    Table out_table(input_table.view());

    EXPECT_EQ(input_table.num_columns(), out_table.num_columns());
    EXPECT_EQ(input_table.num_rows(), out_table.num_rows());
}

TEST_F(TableTest, GetTableWithSelectedColumns)
{
  column_wrapper <int8_t>  col1{{1,2,3,4}};
  column_wrapper <int16_t> col2{{1,2,3,4}};
  column_wrapper <int32_t> col3{{4,5,6,7}};
  column_wrapper <int64_t> col4{{4,5,6,7}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());

  Table t(std::move(cols));

  cudf::table_view selected_tview = t.select(std::vector<cudf::size_type>{2,3});
  cudf::test::expect_columns_equal(t.view().column(2), selected_tview.column(0));
  cudf::test::expect_columns_equal(t.view().column(3), selected_tview.column(1));
}

TEST_F(TableTest, SelectingOutOfBounds)
{
  column_wrapper <int8_t > col1{{1,2,3,4}};
  column_wrapper <int16_t> col2{{1,2,3,4}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());

  Table t(std::move(cols));

  EXPECT_THROW (t.select(std::vector<cudf::size_type>{0,1,2}), std::out_of_range);
}

TEST_F(TableTest, SelectingNoColumns)
{
  column_wrapper <int8_t > col1{{1,2,3,4}};
  column_wrapper <int16_t> col2{{1,2,3,4}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  Table t(std::move(cols));
  TView selected_table = t.select(std::vector<cudf::size_type>{});

  EXPECT_EQ(selected_table.num_columns(), 0);
}

TEST_F(TableTest, CreateFromViewVector)
{
  column_wrapper <int8_t > col1{{1,2,3,4}};
  column_wrapper <int16_t> col2{{1,2,3,4}};

  std::vector<TView> views;
  views.emplace_back(std::vector<column_view>{col1});
  views.emplace_back(std::vector<column_view>{col2});
  TView final_view{views};
  cudf::test::expect_columns_equal(final_view.column(0), views[0].column(0));
  cudf::test::expect_columns_equal(final_view.column(1), views[1].column(0));
}

TEST_F(TableTest, CreateFromViewVectorRowsMismatch)
{
  column_wrapper <int8_t > col1{{1,2,3,4}};
  column_wrapper <int16_t> col2{{1,2,3}};

  std::vector<TView> views;
  views.emplace_back(std::vector<column_view>{col1});
  views.emplace_back(std::vector<column_view>{col2});
  EXPECT_THROW (TView{views}, cudf::logic_error);
}

TEST_F(TableTest, CreateFromViewVectorEmptyTables)
{
  std::vector<TView> views;
  views.emplace_back(std::vector<column_view>{});
  views.emplace_back(std::vector<column_view>{});
  TView final_view{views};
  EXPECT_EQ(final_view.num_columns(), 0);
}

TEST_F(TableTest, ConcatenateTables)
{
  std::vector<const char*> h_strings{
    "Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit" };

  CVector cols_gold;
  column_wrapper <int8_t > col1_gold{{1,2,3,4,5,6,7,8}};
  column_wrapper <int16_t> col2_gold{{1,2,3,4,5,6,7,8}};
  s_col_wrapper            col3_gold(h_strings.data(), h_strings.data() + h_strings.size());
  cols_gold.push_back(col1_gold.release());
  cols_gold.push_back(col2_gold.release());
  cols_gold.push_back(col3_gold.release());
  Table gold_table(std::move(cols_gold));

  CVector cols_table1;
  column_wrapper <int8_t > col1_table1{{1,2,3,4}};
  column_wrapper <int16_t> col2_table1{{1,2,3,4}};
  s_col_wrapper            col3_table1(h_strings.data(), h_strings.data() + 4);
  cols_table1.push_back(col1_table1.release());
  cols_table1.push_back(col2_table1.release());
  cols_table1.push_back(col3_table1.release());
  Table t1(std::move(cols_table1));

  CVector cols_table2;
  column_wrapper <int8_t > col1_table2{{5,6,7,8}};
  column_wrapper <int16_t> col2_table2{{5,6,7,8}};
  s_col_wrapper            col3_table2(h_strings.data() + 4, h_strings.data() + h_strings.size());
  cols_table2.push_back(col1_table2.release());
  cols_table2.push_back(col2_table2.release());
  cols_table2.push_back(col3_table2.release());
  Table t2(std::move(cols_table2));

  auto concat_table = cudf::experimental::concatenate({t1.view(), t2.view()});

  cudf::test::expect_tables_equal(*concat_table, gold_table);
}

TEST_F(TableTest, ConcatenateTablesWithOffsets)
{
  column_wrapper<int32_t> col1_1{{5, 4, 3, 5, 8, 5, 6}};
  cudf::test::strings_column_wrapper col2_1({"dada", "egg", "avocado", "dada", "kite", "dog", "ln"});
  cudf::table_view table_view_in1 {{col1_1, col2_1}};

  column_wrapper<int32_t> col1_2{{5, 8, 5, 6, 15, 14, 13}};
  cudf::test::strings_column_wrapper col2_2({"dada", "kite", "dog", "ln", "dado", "greg", "spinach"});
  cudf::table_view table_view_in2 {{col1_2, col2_2}};

  std::vector<cudf::size_type> split_indexes1{3};
  std::vector<cudf::table_view> partitioned1 =
    cudf::experimental::split( table_view_in1, split_indexes1);

  std::vector<cudf::size_type> split_indexes2{3};
  std::vector<cudf::table_view> partitioned2 =
    cudf::experimental::split( table_view_in2, split_indexes2);

  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::experimental::table> concatenated_tables =
      cudf::experimental::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 6, 15, 14, 13}};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "ln", "dado", "greg", "spinach"});
    cudf::table_view table_view_exp1 {{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal( concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[0]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::experimental::table> concatenated_tables =
      cudf::experimental::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 4, 3, 6, 15, 14, 13}};
    cudf::test::strings_column_wrapper exp2_1({"dada", "egg", "avocado", "ln", "dado", "greg", "spinach"});
    cudf::table_view table_view_exp1 {{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal( concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[0]);
    std::unique_ptr<cudf::experimental::table> concatenated_tables =
      cudf::experimental::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 5, 8, 5}};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "dada", "kite", "dog"});
    cudf::table_view table_view_exp {{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal( concatenated_tables->view(), table_view_exp);
  }
}

TEST_F(TableTest, ConcatenateTablesWithOffsetsAndNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1_1{{5, 4, 3, 5, 8, 5, 6},{0,1,1,1,1,1,1}};
  cudf::test::strings_column_wrapper col2_1({"dada", "egg", "avocado", "dada", "kite", "dog", "ln"},{1,1,1,0,1,1,1});
  cudf::table_view table_view_in1 {{col1_1, col2_1}};

  cudf::test::fixed_width_column_wrapper<int32_t> col1_2{{5, 8, 5, 6, 15, 14, 13},{1,1,1,1,1,1,0}};
  cudf::test::strings_column_wrapper col2_2({"dada", "kite", "dog", "ln", "dado", "greg", "spinach"},{1,0,1,1,1,1,1});
  cudf::table_view table_view_in2 {{col1_2, col2_2}};

  std::vector<cudf::size_type> split_indexes1{3};
  std::vector<cudf::table_view> partitioned1 =
    cudf::experimental::split( table_view_in1, split_indexes1);

  std::vector<cudf::size_type> split_indexes2{3};
  std::vector<cudf::table_view> partitioned2 =
    cudf::experimental::split( table_view_in2, split_indexes2);

  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::experimental::table> concatenated_tables =
      cudf::experimental::concatenate(table_views_to_concat);

    cudf::test::fixed_width_column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 6, 15, 14, 13},{1,1,1,1,1,1,1,0}};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "ln", "dado", "greg", "spinach"},{0,1,1,1,1,1,1,1});
    cudf::table_view table_view_exp1 {{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal( concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[0]);
    std::unique_ptr<cudf::experimental::table> concatenated_tables =
      cudf::experimental::concatenate(table_views_to_concat);

    cudf::test::fixed_width_column_wrapper<int32_t> exp1_1{5, 8, 5, 6, 5, 8, 5};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "dada", "kite", "dog"},{0,1,1,1,1,0,1});
    cudf::table_view table_view_exp1 {{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal( concatenated_tables->view(), table_view_exp1);
  }
}
